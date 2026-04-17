from dolfin import *
import numpy as np
import os
try:
    from ufl import tanh
except ModuleNotFoundError:
    from ufl_legacy import tanh

from mma import mmasub
from TurbulenceModel_SpalartAllmaras_TO_Frozen import (
    SpalartAllmarasSteadyState,
    sa_turbulent_viscosity,
)
from Utilities_SharedTO import (
    append_optimization_log_entry,
    as_list,
    build_pressure_pin_expression_from_config,
    compute_filter_base_length_from_config,
    create_design_mesh_from_config,
    ensure_clean_dir,
    initialize_optimization_log,
    load_config_module_from_cli,
    ResilientVTKFile,
)
from Utilities_TurbulentTO_Frozen import (
    build_penalized_wall_distance_solver,
    enforce_scalar_floor,
    nu_tilde_from_viscosity_ratio,
    positive_part,
)

# ====================================================================================
# Turbulent topology optimization with a frozen-turbulence adjoint.
# The flow is advanced with IPCS pressure correction, the SA model is updated in
# an outer Picard loop, and the adjoint is solved only for the flow variables (u, p).
# The continuation schedule gradually sharpens the design while reducing MMA moves.
# ====================================================================================

# The DG Helmholtz filter uses interior-facet terms (dS), which require
# ghosted mesh entities for parallel assembly.
parameters["ghost_mode"] = "shared_facet"
COMM = MPI.comm_world
IS_ROOT = MPI.rank(COMM) == 0


def root_print(message):
    if IS_ROOT:
        print(message)

CONFIG_MODULE_NAME, CONFIG = load_config_module_from_cli()
for _name, _value in vars(CONFIG).items():
    if not _name.startswith("_"):
        globals()[_name] = _value
root_print("Using config module: {}".format(CONFIG_MODULE_NAME))

SHOW_SOLVE_LABELS = bool(globals().get("SHOW_SOLVE_LABELS", True))
SHOW_DOLFIN_SOLVER_LOGS = bool(globals().get("SHOW_DOLFIN_SOLVER_LOGS", False))
FORWARD_IPCS_LOG_EVERY = max(1, int(globals().get("FORWARD_IPCS_LOG_EVERY", 25)))
LINEAR_SOLVER_NAME = str(globals().get("LINEAR_SOLVER", globals().get("SNES_LINEAR_SOLVER", "mumps")))

if not SHOW_DOLFIN_SOLVER_LOGS:
    try:
        set_log_level(LogLevel.WARNING)
    except NameError:
        set_log_level(30)


def solver_log(message):
    if SHOW_SOLVE_LABELS and IS_ROOT:
        print(message)

BETA_PROJ = Constant(float(BETA_PROJ_VALUE))
if "BETA_PROJ_SCHEDULE" in globals():
    BETA_PROJ_SCHEDULE = [float(b) for b in BETA_PROJ_SCHEDULE]
else:
    BETA_PROJ_SCHEDULE = [float(BETA_PROJ_VALUE)] * len(Q_PENAL_SCHEDULE)

# Brinkman limits for fluid and solid regions.
mu_fluid = Constant(MU_FLUID_VALUE)
rho_fluid = Constant(RHO_FLUID_VALUE)
brinkman_fluid_length = float(globals().get("BRINKMAN_FLUID_LENGTH", 100.0))
brinkman_solid_length = float(globals().get("BRINKMAN_SOLID_LENGTH", 0.01))
alpha_fluid = Constant(float(globals().get("ALPHA_FLUID", 2.5 * MU_FLUID_VALUE / brinkman_fluid_length**2.0)))
alpha_solid = Constant(float(globals().get("ALPHA_SOLID", 2.5 * MU_FLUID_VALUE / brinkman_solid_length**2.0)))
q_penal = Constant(0.1)  # updated each continuation stage


def projection(rho_design, eta_proj):
    """Projection used to sharpen the filtered design."""
    return (
        tanh(BETA_PROJ * Constant(eta_proj))
        + tanh(BETA_PROJ * (rho_design - Constant(eta_proj)))
    ) / (
        tanh(BETA_PROJ * Constant(eta_proj))
        + tanh(BETA_PROJ * (Constant(1.0) - Constant(eta_proj)))
    )

def alpha(brinkman_density):
    """Interpolate the Brinkman penalty between fluid and solid."""
    return alpha_solid + (alpha_fluid - alpha_solid) * brinkman_density * (1 + q_penal) / (
        brinkman_density + q_penal
    )

# CONVENTION
# brinkman_density = 0 for solid region
# brinkman_density = 1 for fluid region

# alpha = alpha_solid (>>) in solid region
# alpha = alpha_fluid (<<) in fluid region

def sa_positive_viscosity(state_nu_tilde):
    """Return the SA turbulent viscosity with negative values clipped away."""
    nu_lam = mu_fluid / rho_fluid
    return sa_turbulent_viscosity(
        state_nu_tilde,
        nu_lam,
        smooth_abs_eps=float(globals().get("SA_SMOOTH_ABS_EPS", 1.0e-12)),
    )


def effective_dynamic_viscosity(frozen_nu_tilde):
    """Return the effective viscosity used in the flow solve."""
    return mu_fluid + rho_fluid * sa_positive_viscosity(frozen_nu_tilde)


def build_state_form(state_u, state_p, adj_u, adj_p, rho_eff, custom_dx, frozen_nu_tilde):
    """Build the flow weak form using the frozen turbulent viscosity."""
    mu_effective = effective_dynamic_viscosity(frozen_nu_tilde)
    return (
        rho_fluid * inner(dot(state_u, nabla_grad(state_u)), adj_u) * custom_dx
        + mu_effective * inner(grad(state_u), grad(adj_u)) * custom_dx
        + inner(grad(state_p), adj_u) * custom_dx
        + inner(div(state_u), adj_p) * custom_dx
        + alpha(rho_eff) * inner(state_u, adj_u) * custom_dx
    )

# ---------------------------------------------------------------
# Mesh, spaces, and boundaries.
# ---------------------------------------------------------------
mesh = create_design_mesh_from_config(globals())

# Flow uses Taylor-Hood, SA uses CG1, and density uses DG0.
U_h = VectorElement("CG", mesh.ufl_cell(), 2)
P_h = FiniteElement("CG", mesh.ufl_cell(), 1)
T_h = FiniteElement("CG", mesh.ufl_cell(), 1)
A_h = FiniteElement("DG", mesh.ufl_cell(), 0)

FlowSpace = FunctionSpace(mesh, U_h * P_h)     # forward (u, p)
FlowSpaceAdj = FunctionSpace(mesh, U_h * P_h)  # adjoint (lambda_u, lambda_p); same topology, kept separate for clarity
TurbulenceSpace = FunctionSpace(mesh, T_h)
DensitySpace = FunctionSpace(mesh, A_h)

w_fwd = Function(FlowSpace)       # forward state: (u, p)
(u, p) = split(w_fwd)
w_adj = Function(FlowSpaceAdj)    # adjoint state: (lambda_u, lambda_p)
(v, q) = split(w_adj)

rho = Function(DensitySpace)           # MMA design variable (unfiltered)
rho_f = Function(DensitySpace)         # PDE-filtered density
nu_tilde_frozen = Function(TurbulenceSpace)  # SA working variable, frozen during NS/adjoint

rho_proj_plot = Function(DensitySpace)    # projected rho for visualisation only
unfiltered_gradient = Function(DensitySpace)
filtered_gradient = Function(DensitySpace)
unfiltered_s_vol = Function(DensitySpace)
filtered_s_vol = Function(DensitySpace)
unfiltered_constraint_gradient = Function(DensitySpace)
filtered_constraint_gradient = Function(DensitySpace)


def _as_density_function(value, density_space):
    """Convert a scalar / expression / function into a DG0 Function."""
    if isinstance(value, Function):
        return value

    out = Function(density_space)
    if np.isscalar(value):
        assign(out, interpolate(Constant(float(value)), density_space))
    else:
        assign(out, interpolate(value, density_space))
    return out


def build_density_bounds_from_config():
    """Return per-cell lower/upper bounds for rho, defaulting to [0, 1]."""
    density_bounds_builder = globals().get("build_density_bounds")
    if callable(density_bounds_builder):
        lower_spec, upper_spec = density_bounds_builder(mesh, DensitySpace)
    else:
        lower_spec, upper_spec = 0.0, 1.0

    lower_bound = _as_density_function(lower_spec, DensitySpace)
    upper_bound = _as_density_function(upper_spec, DensitySpace)

    lower_values = lower_bound.vector().get_local()
    upper_values = upper_bound.vector().get_local()

    if np.any(lower_values > upper_values + 1.0e-12):
        raise ValueError("Density lower bound exceeds upper bound in at least one cell.")

    lower_values = np.clip(lower_values, 0.0, 1.0)
    upper_values = np.clip(upper_values, 0.0, 1.0)

    lower_bound.vector().set_local(lower_values)
    lower_bound.vector().apply("insert")
    upper_bound.vector().set_local(upper_values)
    upper_bound.vector().apply("insert")

    fixed_cells = np.abs(upper_values - lower_values) < 1.0e-12
    num_fixed = int(np.count_nonzero(fixed_cells))
    if num_fixed > 0:
        num_fixed_fluid = int(np.count_nonzero(fixed_cells & (lower_values > 0.5)))
        num_fixed_solid = int(np.count_nonzero(fixed_cells & (upper_values < 0.5)))
        root_print(
            "Density bounds: {} passive cells ({} fluid, {} solid).".format(
                num_fixed, num_fixed_fluid, num_fixed_solid,
            )
        )

    return lower_bound, upper_bound


def build_region_function_from_config(builder_name, default_value=1.0):
    """Return a DG0 mask used to restrict objective/volume integrations."""
    region_builder = globals().get(builder_name)
    if callable(region_builder):
        region_spec = region_builder(mesh, DensitySpace)
    else:
        region_spec = default_value

    region_function = _as_density_function(region_spec, DensitySpace)
    region_values = np.clip(region_function.vector().get_local(), 0.0, 1.0)
    region_function.vector().set_local(region_values)
    region_function.vector().apply("insert")
    return region_function

mark = MARK
boundaries = globals()["mark_boundaries"](mesh)
wall_markers = as_list(mark["walls"])
inlet_markers = as_list(mark["inlet"])
outlet_markers = as_list(mark["outlet"])
density_lower_bound, density_upper_bound = build_density_bounds_from_config()
density_lower_values = density_lower_bound.vector().get_local()
density_upper_values = density_upper_bound.vector().get_local()
ObjectiveRegion = build_region_function_from_config("build_objective_region", 1.0)
VolumeRegion = build_region_function_from_config("build_volume_region", 1.0)

if "QUADRATURE_DEGREE" in globals():
    quadrature_degree = int(QUADRATURE_DEGREE)
    parameters["form_compiler"]["quadrature_degree"] = quadrature_degree
    measure_metadata = {"quadrature_degree": quadrature_degree}
    dx = Measure("dx", domain=mesh, metadata=measure_metadata)
    ds = Measure("ds", domain=mesh, subdomain_data=boundaries, metadata=measure_metadata)
    dS = Measure("dS", domain=mesh, metadata=measure_metadata)
else:
    dx = Measure("dx", domain=mesh)
    ds = Measure("ds", domain=mesh, subdomain_data=boundaries)
    dS = Measure("dS", domain=mesh)
n = FacetNormal(mesh)

inlet_profiles, outlet_profiles = globals()["build_velocity_profile_sets"]()
inlet_profiles = as_list(inlet_profiles)
outlet_profiles = as_list(outlet_profiles)

u_noslip = Constant((0.0, 0.0))
# This config toggle decides whether the outlet gets a velocity Dirichlet BC or a pressure Dirichlet BC.
outlet_bc_type = str(globals().get("OUTLET_BC_TYPE", "velocity")).strip().lower()
if outlet_bc_type not in ("velocity", "pressure"):
    raise ValueError("OUTLET_BC_TYPE must be either 'velocity' or 'pressure'.")
use_outlet_velocity_bc = outlet_bc_type == "velocity"
use_outlet_pressure_bc = outlet_bc_type == "pressure"
use_pressure_pin = bool(globals().get("ENABLE_PRESSURE_PIN", True))
if use_outlet_pressure_bc and use_pressure_pin:
    root_print("Outlet pressure BC requested; disabling the redundant pointwise pressure pin.")
    use_pressure_pin = False
outlet_pressure_value_float = float(globals().get("OUTLET_PRESSURE_VALUE", 0.0))
outlet_pressure_value = Constant(outlet_pressure_value_float)

if len(inlet_profiles) != len(inlet_markers):
    raise ValueError(
        "Expected {} inlet velocity profiles, got {}.".format(
            len(inlet_markers), len(inlet_profiles),
        )
    )
if use_outlet_velocity_bc and len(outlet_profiles) != len(outlet_markers):
    raise ValueError(
        "Expected {} outlet velocity profiles, got {}.".format(
            len(outlet_markers), len(outlet_profiles),
        )
    )

if use_outlet_pressure_bc:
    root_print("Outlet BC type: pressure (p = {:.3e} on outlet).".format(outlet_pressure_value_float))
else:
    root_print("Outlet BC type: velocity profile.")

bcu_walls = [DirichletBC(FlowSpace.sub(0), u_noslip, boundaries, m) for m in wall_markers]
bcu_inlet = [DirichletBC(FlowSpace.sub(0), prof, boundaries, m) for prof, m in zip(inlet_profiles, inlet_markers)]
# Velocity outlet mode: prescribe u on the outlet marker.
bcu_outlet = (
    [DirichletBC(FlowSpace.sub(0), prof, boundaries, m) for prof, m in zip(outlet_profiles, outlet_markers)]
    if use_outlet_velocity_bc else []
)
# Pressure outlet mode: prescribe p on the outlet marker instead of an outlet velocity profile.
bcp_outlet = (
    [DirichletBC(FlowSpace.sub(1), outlet_pressure_value, boundaries, m) for m in outlet_markers]
    if use_outlet_pressure_bc else []
)
bc_NS = bcu_walls + bcu_inlet + bcu_outlet + bcp_outlet
if use_pressure_pin:
    bc_NS.append(DirichletBC(
        FlowSpace.sub(1), Constant(0.0),
        build_pressure_pin_expression_from_config(globals()), "pointwise",
    ))

# Use zero adjoint velocity where the forward velocity is prescribed.
adjoint_velocity_bc_builder = globals().get("build_adjoint_velocity_bcs")
if callable(adjoint_velocity_bc_builder):
    bc_NS_adj = list(adjoint_velocity_bc_builder(
        FlowSpaceAdj, boundaries, wall_markers, inlet_markers, outlet_markers,
        u_noslip, inlet_profiles, outlet_profiles,
    ))
else:
    bc_NS_adj = (
        [DirichletBC(FlowSpaceAdj.sub(0), u_noslip, boundaries, m) for m in wall_markers]
        + [DirichletBC(FlowSpaceAdj.sub(0), u_noslip, boundaries, m) for m in inlet_markers]
    )
    if use_outlet_velocity_bc:
        bc_NS_adj += [DirichletBC(FlowSpaceAdj.sub(0), u_noslip, boundaries, m) for m in outlet_markers]
if use_outlet_pressure_bc:
    bc_NS_adj += [DirichletBC(FlowSpaceAdj.sub(1), outlet_pressure_value, boundaries, m) for m in outlet_markers]
if use_pressure_pin:
    bc_NS_adj.append(DirichletBC(
        FlowSpaceAdj.sub(1), Constant(0.0),
        build_pressure_pin_expression_from_config(globals()), "pointwise",
    ))


def compute_fixed_inlet_flow_rate():
    """Return the positive inlet flow magnitude implied by the prescribed inlet profiles."""
    inlet_flux = 0.0
    for profile, marker in zip(inlet_profiles, inlet_markers):
        inlet_flux += assemble(dot(profile, n) * ds(marker))
    return -float(inlet_flux)


mass_flow_constraint_markers = []
mass_flow_target_fractions = []
mass_flow_constraint_tolerance = float(globals().get("MASS_FLOW_CONSTRAINT_TOLERANCE", 1.0e-4))
mass_flow_constraint_mode = str(globals().get("MASS_FLOW_CONSTRAINT_MODE", "upper")).strip().lower()
fixed_inlet_flow_value = None
use_two_sided_mass_flow_constraints = mass_flow_constraint_mode in {"band", "equality", "two_sided"}

if mass_flow_constraint_mode not in {"upper", "band", "equality", "two_sided"}:
    raise ValueError(
        "MASS_FLOW_CONSTRAINT_MODE must be one of 'upper', 'band', 'equality', or 'two_sided'."
    )

if "MASS_FLOW_TARGET_FRACTIONS" in globals():
    mass_flow_target_fractions = [float(value) for value in as_list(MASS_FLOW_TARGET_FRACTIONS)]
    mass_flow_constraint_markers = [
        int(value) for value in as_list(globals().get("MASS_FLOW_CONSTRAINT_MARKERS", outlet_markers))
    ]
    if len(mass_flow_constraint_markers) != len(mass_flow_target_fractions):
        raise ValueError(
            "Expected the same number of MASS_FLOW_CONSTRAINT_MARKERS and MASS_FLOW_TARGET_FRACTIONS."
        )
    if use_outlet_velocity_bc:
        raise ValueError("Mass-flow outlet constraints require OUTLET_BC_TYPE = 'pressure'.")

    fixed_inlet_flow_value = compute_fixed_inlet_flow_rate()
    if fixed_inlet_flow_value <= 0.0:
        raise ValueError(
            "The prescribed inlet profiles yield a non-positive fixed inlet flow rate ({:.4e}).".format(
                fixed_inlet_flow_value
            )
        )

    root_print(
        "Mass-flow constraints [{}]: markers {} with target fractions {} (Fin = {:.4e}, eps = {:.1e}).".format(
            mass_flow_constraint_mode,
            mass_flow_constraint_markers,
            ["{:.3f}".format(value) for value in mass_flow_target_fractions],
            fixed_inlet_flow_value,
            mass_flow_constraint_tolerance,
        )
    )

# Build SA inlet values from either custom inlet profiles supplied by the
# config or, by default, from uniform viscosity-ratio targets / scalar values.
_nu_lam = MU_FLUID_VALUE / RHO_FLUID_VALUE
custom_turbulence_inlet_builder = globals().get("build_turbulence_inlet_profile_sets")
if callable(custom_turbulence_inlet_builder):
    nu_tilde_inlet_bc_values = as_list(custom_turbulence_inlet_builder())
    if len(nu_tilde_inlet_bc_values) != len(inlet_markers):
        raise ValueError(
            "Expected {} custom SA inlet profiles, got {}.".format(
                len(inlet_markers), len(nu_tilde_inlet_bc_values),
            )
        )
    root_print("Using custom SA inlet profiles from config (nu_lam = {:.3e}).".format(_nu_lam))
else:
    if "SA_MUT_RATIOS" in globals():
        _sa_nu_tilde_targets = [nu_tilde_from_viscosity_ratio(r, _nu_lam) for r in as_list(SA_MUT_RATIOS)]
    elif "SA_MUT_RATIO" in globals():
        _sa_nu_tilde_targets = [nu_tilde_from_viscosity_ratio(SA_MUT_RATIO, _nu_lam)] * len(inlet_markers)
    elif "SA_NU_TILDE_INLETS" in globals():
        _sa_nu_tilde_targets = [float(value) for value in as_list(SA_NU_TILDE_INLETS)]
    else:
        _sa_nu_tilde_targets = [float(SA_NU_TILDE_INLET)]
    if len(_sa_nu_tilde_targets) != len(inlet_markers):
        raise ValueError(
            "Expected {} SA inlet nu_tilde values, got {}.".format(
                len(inlet_markers), len(_sa_nu_tilde_targets),
            )
        )
    root_print("SA inlet nu_tilde: {}  (nu_lam = {:.3e})".format(
        ["  {:.4e}".format(v) for v in _sa_nu_tilde_targets], _nu_lam,
    ))
    nu_tilde_inlet_bc_values = [Constant(value) for value in _sa_nu_tilde_targets]
bcn_turbulence = (
    [DirichletBC(TurbulenceSpace, bc_val, boundaries, m) for bc_val, m in zip(nu_tilde_inlet_bc_values, inlet_markers)]
    + [DirichletBC(TurbulenceSpace, Constant(0.0), boundaries, m) for m in wall_markers]
)

# ---------------------------------------------------------------
# PDE density filter.
# ---------------------------------------------------------------
r_filter = (
    compute_filter_base_length_from_config(globals())
    * float(globals().get("FILTER_RADIUS_IN_CELLS", 3.0))
)
r = r_filter / (2.0 * 3.0**0.5)  # convert cell-count radius to PDE length scale

u_filter = TrialFunction(DensitySpace)
v_filter = TestFunction(DensitySpace)
filter_in = Function(DensitySpace)
h = CellDiameter(mesh)
h_avg = (h("+") + h("-")) / 2.0


def pde_filter(input_field, output_field):
    alpha_dg = 4.0
    helmholtz = (
        r**2 * (alpha_dg / h_avg * dot(jump(v_filter, n), jump(u_filter, n))) * dS
        + u_filter * v_filter * dx
        - filter_in * v_filter * dx
    )
    assign(filter_in, input_field)
    solve(lhs(helmholtz) == rhs(helmholtz), output_field)
    return output_field


def enforce_density_bounds_inplace(density_field):
    values = density_field.vector().get_local()
    values = np.clip(values, density_lower_values, density_upper_values)
    density_field.vector().set_local(values)
    density_field.vector().apply("insert")
    return density_field

# ---------------------------------------------------------------
# SA model setup.
# ---------------------------------------------------------------
rho_projected = projection(rho_f, ETA_I) # projected design variable
rho_effective = density_lower_bound + (density_upper_bound - density_lower_bound) * rho_projected

custom_wall_distance_builder = globals().get("build_wall_distance_field")
custom_initial_wall_distance = None
if callable(custom_wall_distance_builder):
    custom_initial_wall_distance = custom_wall_distance_builder(
        TurbulenceSpace, mesh, boundaries, wall_markers, dx
    )

sa_wall_density_source = str(globals().get("SA_WALL_DENSITY_SOURCE", "design")).strip().lower()
if sa_wall_density_source not in {"design", "passive"}:
    raise ValueError(
        "SA_WALL_DENSITY_SOURCE must be either 'design' or 'passive'. "
        "Got {!r}.".format(sa_wall_density_source)
    )
sa_wall_sigma = float(globals().get("SA_WALL_SIGMA", globals().get("SA_DISTANCE_RELAXATION", 0.01)))
sa_wall_g0 = float(globals().get("SA_WALL_G0", 20.0))
sa_wall_penalty_alpha = float(globals().get("SA_WALL_PENALTY_ALPHA", 1.0e3))
sa_wall_penalty_power = float(globals().get("SA_WALL_PENALTY_N", 3.0))
sa_wall_g_floor = float(globals().get("SA_WALL_G_FLOOR", 1.0e-8))
sa_wall_newton_rtol = float(globals().get("SA_WALL_NEWTON_RTOL", 1.0e-8))
sa_wall_newton_atol = float(globals().get("SA_WALL_NEWTON_ATOL", 1.0e-10))
sa_wall_newton_max_iters = int(globals().get("SA_WALL_NEWTON_MAX_ITERS", 80))
sa_wall_newton_relax = float(globals().get("SA_WALL_NEWTON_RELAXATION", 0.5))
sa_wall_penalty_homotopy = globals().get("SA_WALL_PENALTY_HOMOTOPY", (0.0, 0.1, 0.25, 0.5, 1.0))
sa_wall_initial_solid_guess = float(globals().get("SA_WALL_INITIAL_SOLID_GUESS", 1.0))
sa_wall_extra_relaxations = globals().get("SA_WALL_NEWTON_RELAXATION_CANDIDATES", None)
sa_wall_solid_threshold = float(globals().get("SA_WALL_SOLID_THRESHOLD", 1.0))
sa_wall_prefer_pseudo_time = bool(globals().get("SA_WALL_PREFER_PSEUDO_TIME", False))
if sa_wall_density_source == "passive":
    wall_penalty_fluid_indicator = density_upper_bound
else:
    wall_penalty_fluid_indicator = rho_effective

(
    wall_distance,
    update_wall_distance_field,
) = build_penalized_wall_distance_solver(
    TurbulenceSpace, boundaries, wall_markers, dx, wall_penalty_fluid_indicator,
    sa_wall_sigma, sa_wall_g0, sa_wall_penalty_alpha, sa_wall_penalty_power, sa_wall_g_floor,
    newton_rtol=sa_wall_newton_rtol,
    newton_atol=sa_wall_newton_atol,
    newton_max_iters=sa_wall_newton_max_iters,
    newton_relax=sa_wall_newton_relax,
    penalty_homotopy=sa_wall_penalty_homotopy,
    solid_guess_weight=sa_wall_initial_solid_guess,
    extra_relaxations=sa_wall_extra_relaxations,
    solid_threshold=sa_wall_solid_threshold,
    initial_wall_distance=custom_initial_wall_distance,
    prefer_pseudo_time=sa_wall_prefer_pseudo_time,
)
root_print("Penalized SA wall-distance enabled: source={}, init_G={}, pseudo_G={}, sigma={}, G0={}, alpha_G={}, n_G={}, rho_cut_G={}, relax_G={}, maxit_G={}".format(
    sa_wall_density_source, "custom" if custom_initial_wall_distance is not None else "geometric",
    sa_wall_prefer_pseudo_time,
    sa_wall_sigma, sa_wall_g0, sa_wall_penalty_alpha, sa_wall_penalty_power,
    sa_wall_solid_threshold, sa_wall_newton_relax, sa_wall_newton_max_iters,
))

# Initial SA field.
sa_nu_tilde_init = float(globals().get("SA_NU_TILDE_INITIAL",
    nu_tilde_from_viscosity_ratio(SA_MUT_RATIO, _nu_lam) if "SA_MUT_RATIO" in globals()
    else float(_sa_nu_tilde_targets[0])
))

nu_laminar = Constant(MU_FLUID_VALUE / RHO_FLUID_VALUE)
sa_nu_tilde_penalty_reaction = Constant(float(globals().get("SA_NU_TILDE_PENALTY_ALPHA", 1.0e3))) * (
    positive_part(Constant(1.0) - rho_effective) ** float(globals().get("SA_NU_TILDE_PENALTY_N", 3.0))
)

sa_model = SpalartAllmarasSteadyState(
    TurbulenceSpace, bcn_turbulence, sa_nu_tilde_init, nu_laminar,
    dx, wall_distance,
    nu_tilde_penalty_reaction=sa_nu_tilde_penalty_reaction,
    wall_distance_floor=float(globals().get("SA_WALL_DISTANCE_FLOOR", 0.0)),
)

# ---------------------------------------------------------------
# Objective, state, adjoint, and sensitivity forms.
# The turbulent viscosity is treated as a frozen coefficient here.
# ---------------------------------------------------------------
dissipation_density_builder = globals().get("build_dissipation_density")
mu_effective = effective_dynamic_viscosity(nu_tilde_frozen)
if callable(dissipation_density_builder):
    dissipation_density = dissipation_density_builder(u, mu_effective)
else:
    deformation = nabla_grad(u) + nabla_grad(u).T
    dissipation_density = 0.5 * mu_effective * inner(deformation, deformation)

# Objective: dissipation plus Brinkman drag.
ObjFunctional = ObjectiveRegion * (
    dissipation_density
    + alpha(rho_effective) * inner(u, u)
) * dx

state_form = build_state_form(u, p, v, q, rho_effective, dx, nu_tilde_frozen)
lagrangian_form = ObjFunctional + state_form

objective_adjoint_form = derivative(lagrangian_form, w_fwd, TestFunction(FlowSpaceAdj))

# Sensitivity with respect to the filtered density.
objective_ddx = derivative(lagrangian_form, rho_f)

# Volume constraint on the projected density.
vol_constraint = VolumeRegion * rho_effective * dx - VolumeRegion * VOL_FRAC * dx
sensitivities_vol_constraint = derivative(vol_constraint, rho_f)

mass_flow_constraints = []
if mass_flow_target_fractions:
    for marker, target_fraction in zip(mass_flow_constraint_markers, mass_flow_target_fractions):
        target_flux_value = target_fraction * fixed_inlet_flow_value
        if target_flux_value <= 0.0:
            raise ValueError("Mass-flow target fraction must be positive.")
        outlet_flux_functional = dot(u, n) * ds(marker)
        normalized_outlet_flux_functional = Constant(1.0 / target_flux_value) * outlet_flux_functional
        upper_constraint_form = normalized_outlet_flux_functional
        upper_constraint_lagrangian_form = upper_constraint_form + state_form
        mass_flow_constraints.append(
            {
                "marker": marker,
                "target_fraction": target_fraction,
                "bound": "upper",
                "functional": upper_constraint_form,
                "offset": -(1.0 + mass_flow_constraint_tolerance),
                "adjoint_form": derivative(
                    upper_constraint_lagrangian_form, w_fwd, TestFunction(FlowSpaceAdj)
                ),
                "gradient_form": derivative(upper_constraint_lagrangian_form, rho_f),
            }
        )
        if use_two_sided_mass_flow_constraints:
            lower_constraint_form = -normalized_outlet_flux_functional
            lower_constraint_lagrangian_form = lower_constraint_form + state_form
            mass_flow_constraints.append(
                {
                    "marker": marker,
                    "target_fraction": target_fraction,
                    "bound": "lower",
                    "functional": lower_constraint_form,
                    "offset": 1.0 - mass_flow_constraint_tolerance,
                    "adjoint_form": derivative(
                        lower_constraint_lagrangian_form, w_fwd, TestFunction(FlowSpaceAdj)
                    ),
                    "gradient_form": derivative(lower_constraint_lagrangian_form, rho_f),
                }
            )

# Stokes-Brinkman warm start used on the first iteration.
u_lin, p_lin = TrialFunctions(FlowSpace)
v_lin, q_lin = TestFunctions(FlowSpace)
a_stokes = (
    mu_fluid * inner(grad(u_lin), grad(v_lin))
    + inner(grad(p_lin), v_lin)
    + inner(div(u_lin), q_lin)
    + alpha(rho_effective) * inner(u_lin, v_lin)
) * dx
l_stokes = Constant(0.0) * q_lin * dx

# ---------------------------------------------------------------
# IPCS pressure-correction setup for the forward flow solve.
# ---------------------------------------------------------------
VelocitySpace = FlowSpace.sub(0).collapse()
PressureSpace = FlowSpace.sub(1).collapse()

u_pc_tr  = TrialFunction(VelocitySpace)
z_pc     = TestFunction(VelocitySpace)
p_pc_tr  = TrialFunction(PressureSpace)
r_pc     = TestFunction(PressureSpace)

u_pc_old  = Function(VelocitySpace)   # previous velocity
u_pc_star = Function(VelocitySpace)   # tentative velocity
u_pc_new  = Function(VelocitySpace)   # corrected velocity
p_pc_old  = Function(PressureSpace)
p_pc_new  = Function(PressureSpace)

bcu_pc  = ([DirichletBC(VelocitySpace, u_noslip, boundaries, m) for m in wall_markers]
         + [DirichletBC(VelocitySpace, prof, boundaries, m) for prof, m in zip(inlet_profiles, inlet_markers)]
         + ([DirichletBC(VelocitySpace, prof, boundaries, m) for prof, m in zip(outlet_profiles, outlet_markers)]
            if use_outlet_velocity_bc else []))
bcp_pc  = (
    ([DirichletBC(PressureSpace, outlet_pressure_value, boundaries, m) for m in outlet_markers]
     if use_outlet_pressure_bc else [])
    + ([DirichletBC(PressureSpace, Constant(0.0),
         build_pressure_pin_expression_from_config(globals()), "pointwise")]
       if use_pressure_pin else [])
)

dt_pc = Constant(float(globals().get("FORWARD_IPCS_DT", 2.0e-4)))

# Tentative velocity step.
a_pc_u    = (rho_fluid / dt_pc * inner(u_pc_tr, z_pc)
           + rho_fluid * inner(dot(u_pc_old, nabla_grad(u_pc_tr)), z_pc)
           + mu_effective * inner(grad(u_pc_tr), grad(z_pc))
           + alpha(rho_effective) * inner(u_pc_tr, z_pc)) * dx
L_pc_u    = (rho_fluid / dt_pc * inner(u_pc_old, z_pc)
           - inner(grad(p_pc_old), z_pc)) * dx

# Pressure correction step.
a_pc_p    = inner(grad(p_pc_tr), grad(r_pc)) * dx
L_pc_p    = (inner(grad(p_pc_old), grad(r_pc))
           - rho_fluid / dt_pc * div(u_pc_star) * r_pc) * dx

# Velocity correction step.
a_pc_corr = inner(u_pc_tr, z_pc) * dx
L_pc_corr = (inner(u_pc_star, z_pc)
           - dt_pc / rho_fluid * inner(grad(p_pc_new - p_pc_old), z_pc)) * dx

# ---------------------------------------------------------------
# Solver helpers.
# ---------------------------------------------------------------


def sanitize_output_label(raw_label):
    text = str(raw_label).strip()
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in text)
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_") or "solve"


def write_ipcs_residual_outputs(
    output_dir,
    base_name,
    u_history,
    p_history,
    rtol_u,
    rtol_p,
    write_svg=False,
):
    os.makedirs(output_dir, exist_ok=True)
    txt_path = os.path.join(output_dir, base_name + ".txt")
    with open(txt_path, "w") as handle:
        handle.write("# step du_rel dp_rel\n")
        handle.write("# u_rtol {:.16e}\n".format(rtol_u))
        handle.write("# p_rtol {:.16e}\n".format(rtol_p))
        for step_idx, (u_val, p_val) in enumerate(zip(u_history, p_history), start=1):
            handle.write("{:04d} {:.16e} {:.16e}\n".format(step_idx, u_val, p_val))

    if (not write_svg) or (not u_history) or (not p_history):
        return

    width = 920
    height = 520
    margin_left = 85
    margin_right = 30
    margin_top = 65
    margin_bottom = 65
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    steps = np.arange(1, len(u_history) + 1, dtype=float)
    u_vals = np.maximum(np.asarray(u_history, dtype=float), 1.0e-16)
    p_vals = np.maximum(np.asarray(p_history, dtype=float), 1.0e-16)
    all_logs = np.log10(np.concatenate((u_vals, p_vals)))

    threshold_values = []
    if rtol_u > 0.0:
        threshold_values.append(max(float(rtol_u), 1.0e-16))
    if rtol_p > 0.0:
        threshold_values.append(max(float(rtol_p), 1.0e-16))
    if threshold_values:
        all_logs = np.concatenate((all_logs, np.log10(np.asarray(threshold_values, dtype=float))))

    y_min = float(np.floor(np.min(all_logs)))
    y_max = float(np.ceil(np.max(all_logs)))
    if y_min == y_max:
        y_min -= 1.0
        y_max += 1.0

    if len(steps) == 1:
        x_positions = np.array([margin_left + 0.5 * plot_width])
    else:
        x_positions = margin_left + (steps - steps[0]) / (steps[-1] - steps[0]) * plot_width

    def y_from_log(log_values):
        return margin_top + (y_max - log_values) / (y_max - y_min) * plot_height

    y_u = y_from_log(np.log10(u_vals))
    y_p = y_from_log(np.log10(p_vals))
    u_points = " ".join("{:.2f},{:.2f}".format(x_val, y_val) for x_val, y_val in zip(x_positions, y_u))
    p_points = " ".join("{:.2f},{:.2f}".format(x_val, y_val) for x_val, y_val in zip(x_positions, y_p))

    x_tick_count = min(max(len(steps), 2), 6)
    if len(steps) == 1:
        x_tick_values = np.array([1.0])
    else:
        x_tick_values = np.linspace(1.0, float(len(steps)), num=x_tick_count)

    y_tick_logs = np.arange(y_min, y_max + 1.0, 1.0)

    svg_lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}" viewBox="0 0 {} {}">'.format(
            width, height, width, height,
        ),
        '  <rect x="0" y="0" width="{}" height="{}" fill="white"/>'.format(width, height),
        '  <text x="{}" y="32" font-size="20" font-family="monospace" fill="#111111">{}</text>'.format(
            margin_left, base_name,
        ),
        '  <text x="{}" y="52" font-size="13" font-family="monospace" fill="#555555">IPCS residual history (log10 scale)</text>'.format(
            margin_left,
        ),
        '  <rect x="{}" y="{}" width="{}" height="{}" fill="none" stroke="#222222" stroke-width="1.2"/>'.format(
            margin_left, margin_top, plot_width, plot_height,
        ),
    ]

    for tick_log in y_tick_logs:
        y_tick = y_from_log(tick_log)
        tick_label = "{:.1e}".format(10.0 ** tick_log)
        svg_lines.append(
            '  <line x1="{}" y1="{:.2f}" x2="{}" y2="{:.2f}" stroke="#d9d9d9" stroke-width="1"/>'.format(
                margin_left, y_tick, margin_left + plot_width, y_tick,
            )
        )
        svg_lines.append(
            '  <text x="{}" y="{:.2f}" font-size="12" font-family="monospace" fill="#333333" text-anchor="end" dominant-baseline="middle">{}</text>'.format(
                margin_left - 10, y_tick, tick_label,
            )
        )

    for tick_value in x_tick_values:
        if len(steps) == 1:
            x_tick = x_positions[0]
        else:
            x_tick = margin_left + (tick_value - 1.0) / (len(steps) - 1.0) * plot_width
        svg_lines.append(
            '  <line x1="{:.2f}" y1="{}" x2="{:.2f}" y2="{}" stroke="#e6e6e6" stroke-width="1"/>'.format(
                x_tick, margin_top, x_tick, margin_top + plot_height,
            )
        )
        svg_lines.append(
            '  <text x="{:.2f}" y="{}" font-size="12" font-family="monospace" fill="#333333" text-anchor="middle">{}</text>'.format(
                x_tick, margin_top + plot_height + 24, int(round(tick_value)),
            )
        )

    if rtol_u > 0.0:
        y_tol_u = y_from_log(np.log10(max(float(rtol_u), 1.0e-16)))
        svg_lines.append(
            '  <line x1="{}" y1="{:.2f}" x2="{}" y2="{:.2f}" stroke="#0b6efd" stroke-width="1.2" stroke-dasharray="6 4"/>'.format(
                margin_left, y_tol_u, margin_left + plot_width, y_tol_u,
            )
        )
    if rtol_p > 0.0:
        y_tol_p = y_from_log(np.log10(max(float(rtol_p), 1.0e-16)))
        svg_lines.append(
            '  <line x1="{}" y1="{:.2f}" x2="{}" y2="{:.2f}" stroke="#d63384" stroke-width="1.2" stroke-dasharray="6 4"/>'.format(
                margin_left, y_tol_p, margin_left + plot_width, y_tol_p,
            )
        )

    svg_lines.extend([
        '  <polyline fill="none" stroke="#0b6efd" stroke-width="2.2" points="{}"/>'.format(u_points),
        '  <polyline fill="none" stroke="#d63384" stroke-width="2.2" points="{}"/>'.format(p_points),
        '  <text x="{}" y="{}" font-size="12" font-family="monospace" fill="#222222">step</text>'.format(
            margin_left + plot_width - 8, margin_top + plot_height + 48,
        ),
        '  <text x="24" y="{}" font-size="12" font-family="monospace" fill="#222222" transform="rotate(-90 24,{})">residual</text>'.format(
            margin_top + 0.5 * plot_height, margin_top + 0.5 * plot_height,
        ),
        '  <line x1="{}" y1="{}" x2="{}" y2="{}" stroke="#0b6efd" stroke-width="2.2"/>'.format(
            margin_left, height - 26, margin_left + 36, height - 26,
        ),
        '  <text x="{}" y="{}" font-size="12" font-family="monospace" fill="#222222">u residual</text>'.format(
            margin_left + 44, height - 22,
        ),
        '  <line x1="{}" y1="{}" x2="{}" y2="{}" stroke="#d63384" stroke-width="2.2"/>'.format(
            margin_left + 170, height - 26, margin_left + 206, height - 26,
        ),
        '  <text x="{}" y="{}" font-size="12" font-family="monospace" fill="#222222">p residual</text>'.format(
            margin_left + 214, height - 22,
        ),
        '</svg>',
    ])

    svg_path = os.path.join(output_dir, base_name + ".svg")
    with open(svg_path, "w") as handle:
        handle.write("\n".join(svg_lines) + "\n")

# Stokes initialization.
def initialize_forward_guess_with_stokes():
    """Build the initial flow and SA guesses from a Stokes-Brinkman solve."""
    A = assemble(a_stokes)
    b = assemble(l_stokes)
    for bc in bc_NS:
        bc.apply(A, b)
    solve(A, w_fwd.vector(), b, LINEAR_SOLVER_NAME)

    wall_dist_scale = max(float(globals().get("SA_INIT_WALL_DIST_SCALE", 0.05 * float(globals().get("L", 1.0)))), 1.0e-12)
    nu_guess = project(
        Constant(sa_nu_tilde_init) * wall_distance / (wall_distance + Constant(wall_dist_scale)),
        TurbulenceSpace,
    )
    nu_tilde_frozen.assign(nu_guess)
    sa_model.nu_tilde0.assign(nu_guess)
    sa_model.nu_tilde1.assign(nu_guess)

# Forward solver.
def solve_forward(solve_label=None):
    """Run IPCS to a steady state and store the result in w_fwd."""
    global ipcs_solve_counter
    global save_ipcs_residual_plots
    max_it  = int(globals().get("FORWARD_IPCS_MAX_ITERS", 200))
    rtol_u  = float(globals().get("FORWARD_IPCS_VELOCITY_RTOL", globals().get("FORWARD_IPCS_RTOL", 1.0e-3)))
    rtol_p  = float(globals().get("FORWARD_IPCS_PRESSURE_RTOL", 2.0e-2))
    omega_u_base = float(globals().get("FORWARD_IPCS_VEL_RELAXATION", globals().get("FORWARD_IPCS_U_RELAXATION", 0.5)))
    omega_p_base = float(globals().get("FORWARD_IPCS_P_RELAXATION", 0.2))
    min_omega_u = float(globals().get("FORWARD_IPCS_MIN_U_RELAXATION", 0.05))
    min_omega_p = float(globals().get("FORWARD_IPCS_MIN_P_RELAXATION", 0.02))
    base_dt = float(globals().get("FORWARD_IPCS_DT", 2.0e-4))
    max_restarts = max(0, int(globals().get("FORWARD_IPCS_MAX_RESTARTS", 0)))
    dt_reduction = float(globals().get("FORWARD_IPCS_DT_REDUCTION_FACTOR", 0.5))
    relax_reduction = float(globals().get("FORWARD_IPCS_RELAXATION_REDUCTION_FACTOR", 0.7))
    error_on_nonconvergence = bool(globals().get("FORWARD_IPCS_ERROR_ON_NONCONVERGENCE", True))
    vel_solver_name = str(globals().get("FORWARD_IPCS_VEL_SOLVER", "bicgstab"))
    p_solver_name = str(globals().get("FORWARD_IPCS_P_SOLVER", "cg"))
    direct_solver_names = {"lu", "mumps", "umfpack", "superlu", "superlu_dist"}

    ksp_u = {"linear_solver": vel_solver_name}
    if vel_solver_name.lower() not in direct_solver_names:
        ksp_u["preconditioner"] = globals().get("FORWARD_IPCS_VEL_PRECONDITIONER", "ilu")

    ksp_p = {"linear_solver": p_solver_name}
    if p_solver_name.lower() not in direct_solver_names:
        ksp_p["preconditioner"] = globals().get("FORWARD_IPCS_P_PRECONDITIONER", "ilu")

    u_start = w_fwd.sub(0, deepcopy=True)
    p_start = w_fwd.sub(1, deepcopy=True)
    best_u = Function(VelocitySpace)
    best_p = Function(PressureSpace)
    assign(best_u, u_start)
    assign(best_p, p_start)
    best_score = np.inf
    best_du = np.inf
    best_dp = np.inf
    best_attempt_idx = 0
    best_step_idx = 0

    attempt_summaries = []
    solver_log("      [IPCS] tentative velocity, pressure correction, velocity correction")
    for attempt_idx in range(max_restarts + 1):
        current_dt = base_dt * (dt_reduction ** attempt_idx)
        current_omega_u = max(min_omega_u, omega_u_base * (relax_reduction ** attempt_idx))
        current_omega_p = max(min_omega_p, omega_p_base * (relax_reduction ** attempt_idx))
        dt_pc.assign(current_dt)

        assign(u_pc_old, u_start)
        assign(p_pc_old, p_start)

        u_history = []
        p_history = []
        du_rel = dp_rel = np.inf
        converged = False
        attempt_best_u = Function(VelocitySpace)
        attempt_best_p = Function(PressureSpace)
        assign(attempt_best_u, u_start)
        assign(attempt_best_p, p_start)
        attempt_best_score = np.inf
        attempt_best_du = np.inf
        attempt_best_dp = np.inf
        attempt_best_step_idx = 0

        solver_log(
            "      [IPCS] attempt {}/{}: dt={:.2e}, omega_u={:.2f}, omega_p={:.2f}".format(
                attempt_idx + 1, max_restarts + 1, current_dt, current_omega_u, current_omega_p,
            )
        )

        for step_idx in range(1, max_it + 1):
            solve(a_pc_u    == L_pc_u,    u_pc_star, bcu_pc, solver_parameters=ksp_u)
            solve(a_pc_p    == L_pc_p,    p_pc_new,  bcp_pc, solver_parameters=ksp_p)
            solve(a_pc_corr == L_pc_corr, u_pc_new,  bcu_pc, solver_parameters=ksp_u)

            du_rel = (np.linalg.norm(u_pc_new.vector()[:] - u_pc_old.vector()[:])
                      / max(np.linalg.norm(u_pc_new.vector()[:]), 1.0e-12))
            dp_rel = (np.linalg.norm(p_pc_new.vector()[:] - p_pc_old.vector()[:])
                      / max(np.linalg.norm(p_pc_new.vector()[:]), 1.0e-12))
            u_history.append(float(du_rel))
            p_history.append(float(dp_rel))

            score = max(
                du_rel / max(rtol_u, 1.0e-16),
                dp_rel / max(rtol_p, 1.0e-16),
            )
            if score < attempt_best_score:
                attempt_best_score = score
                attempt_best_du = float(du_rel)
                attempt_best_dp = float(dp_rel)
                attempt_best_step_idx = step_idx
                assign(attempt_best_u, u_pc_new)
                assign(attempt_best_p, p_pc_new)
                if score < best_score:
                    best_score = score
                    best_du = attempt_best_du
                    best_dp = attempt_best_dp
                    best_attempt_idx = attempt_idx + 1
                    best_step_idx = step_idx
                    assign(best_u, u_pc_new)
                    assign(best_p, p_pc_new)

            if step_idx == 1 or step_idx % FORWARD_IPCS_LOG_EVERY == 0:
                solver_log(
                    "      [IPCS] step {:03d}: du={:.2e}, dp={:.2e}".format(
                        step_idx, du_rel, dp_rel,
                    )
                )
            if du_rel < rtol_u and dp_rel < rtol_p:
                # Keep the fully corrected IPCS state on convergence instead of the
                # lagged under-relaxed iterate used to drive non-converged updates.
                assign(u_pc_old, u_pc_new)
                assign(p_pc_old, p_pc_new)
                converged = True
                solver_log(
                    "      [IPCS] converged in {:03d} steps: du={:.2e}, dp={:.2e}".format(
                        step_idx, du_rel, dp_rel,
                    )
                )
                break

            u_pc_old.vector()[:] = (
                current_omega_u * u_pc_new.vector()[:] + (1.0 - current_omega_u) * u_pc_old.vector()[:]
            )
            p_pc_old.vector()[:] = (
                current_omega_p * p_pc_new.vector()[:] + (1.0 - current_omega_p) * p_pc_old.vector()[:]
            )

        if save_ipcs_residual_plots:
            ipcs_solve_counter += 1
            label = solve_label if solve_label is not None else "solve"
            if max_restarts > 0:
                label = "{}_attempt{:02d}".format(label, attempt_idx + 1)
            base_name = "{:04d}_{}".format(ipcs_solve_counter, sanitize_output_label(label))
            try:
                write_ipcs_residual_outputs(
                    ipcs_residual_dir,
                    base_name,
                    u_history,
                    p_history,
                    rtol_u,
                    rtol_p,
                    write_svg=save_ipcs_residual_svgs,
                )
            except OSError as exc:
                if getattr(exc, "errno", None) == 28:
                    save_ipcs_residual_plots = False
                    root_print(
                        "Warning: disabling IPCS residual file dumps after running out of disk space at {}.".format(
                            os.path.join(ipcs_residual_dir, base_name + ".txt")
                        )
                    )
                else:
                    raise

        attempt_summaries.append(
            "attempt {}: dt={:.2e}, best@{:03d} du={:.2e} (target {:.2e}), dp={:.2e} (target {:.2e}); final du={:.2e}, dp={:.2e}".format(
                attempt_idx + 1, current_dt, attempt_best_step_idx, attempt_best_du, rtol_u,
                attempt_best_dp, rtol_p, du_rel, dp_rel,
            )
        )
        if converged:
            assign(w_fwd.sub(0), u_pc_old)
            assign(w_fwd.sub(1), p_pc_old)
            dt_pc.assign(base_dt)
            return float(du_rel), float(dp_rel)

        # Retry from the best iterate seen in this attempt, not from the
        # potentially degraded final iterate after a long failed march.
        assign(u_start, attempt_best_u)
        assign(p_start, attempt_best_p)

        solver_log(
            "      [IPCS] attempt {}/{} reached the {}-step limit without meeting tolerances: "
            "du={:.2e} (target {:.2e}), dp={:.2e} (target {:.2e})".format(
                attempt_idx + 1, max_restarts + 1, max_it, du_rel, rtol_u, dp_rel, rtol_p,
            )
        )

    dt_pc.assign(base_dt)
    assign(w_fwd.sub(0), best_u)
    assign(w_fwd.sub(1), best_p)
    message = (
        "IPCS did not meet the steady-state tolerances after {} attempt(s). "
        "Best iterate was attempt {} step {:03d}: du={:.2e} (target {:.2e}), "
        "dp={:.2e} (target {:.2e}). Attempt summaries: {}"
    ).format(
        max_restarts + 1, best_attempt_idx, best_step_idx, best_du, rtol_u,
        best_dp, rtol_p, "; ".join(attempt_summaries),
    )
    if error_on_nonconvergence:
        raise RuntimeError(message)
    root_print("Warning: {}".format(message))
    return float(best_du), float(best_dp)


def solve_forward_with_recovery(solve_label=None):
    """Retry a failed IPCS solve from a fresh Stokes-Brinkman warm start."""
    try:
        return solve_forward(solve_label)
    except RuntimeError:
        if not bool(globals().get("FORWARD_IPCS_RESTART_WITH_STOKES", True)):
            raise
        if int(globals().get("FORWARD_IPCS_MAX_RESTARTS", 0)) > 0:
            solver_log(
                "      [IPCS] adaptive attempts already exhausted; skipping duplicate Stokes rerun"
            )
            raise
        solver_log("      [IPCS] rebuilding Stokes-Brinkman warm start and retrying once")
        initialize_forward_guess_with_stokes()
        retry_label = solve_label if solve_label is not None else "solve"
        return solve_forward("{}_stokes".format(retry_label))

# Adjoint solver.
def solve_adjoint(adjoint_residual_form=objective_adjoint_form):
    """Assemble and solve the linear adjoint system."""
    solver_log("    [Adjoint] linear system")
    w_adj.vector().zero()
    A_adj = assemble(derivative(adjoint_residual_form, w_adj))
    b_adj = assemble(adjoint_residual_form)
    b_adj *= -1.0
    for bc in bc_NS_adj:
        bc.apply(A_adj, b_adj)
    solve(A_adj, w_adj.vector(), b_adj, LINEAR_SOLVER_NAME)

# ---------------------------------------------------------------
# Output setup.
# ---------------------------------------------------------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
results_root = os.path.join(THIS_DIR, globals().get("RESULTS_ROOT_NAME", "Results_Frozen/Results_TurbulentTO_Frozen"))
rho_dir = os.path.join(results_root, "rho")
rho_p_dir = os.path.join(results_root, "rho_projected")
u_dir = os.path.join(results_root, "u")
p_dir = os.path.join(results_root, "p")
nu_tilde_dir = os.path.join(results_root, "nu_tilde")
design_dir = os.path.join(results_root, "design")
ipcs_residual_dir = os.path.join(results_root, "ipcs_residuals")
save_ipcs_residual_plots = bool(globals().get("SAVE_IPCS_RESIDUAL_PLOTS", False))
save_ipcs_residual_svgs = bool(globals().get("SAVE_IPCS_RESIDUAL_SVGS", False))
ipcs_solve_counter = 0

ensure_clean_dir(results_root)
ensure_clean_dir(rho_dir)
ensure_clean_dir(rho_p_dir)
ensure_clean_dir(u_dir)
ensure_clean_dir(p_dir)
ensure_clean_dir(nu_tilde_dir)
ensure_clean_dir(design_dir)
if save_ipcs_residual_plots:
    ensure_clean_dir(ipcs_residual_dir)

rho_out = ResilientVTKFile(os.path.join(rho_dir, "plot_rho.pvd"), COMM)
rhop_out = ResilientVTKFile(os.path.join(rho_p_dir, "plot_rho_projected.pvd"), COMM)
u_out = ResilientVTKFile(os.path.join(u_dir, "plot_u.pvd"), COMM)
p_out = ResilientVTKFile(os.path.join(p_dir, "plot_p.pvd"), COMM)
nu_tilde_out = ResilientVTKFile(os.path.join(nu_tilde_dir, "plot_nu_tilde.pvd"), COMM)

log_path = os.path.join(results_root, "OptimizationLog.txt")
initialize_optimization_log(log_path, include_ipcs_residuals=True)

# ---------------------------------------------------------------
# MMA setup.
# ---------------------------------------------------------------
initial_density = float(globals().get("INITIAL_DENSITY_VALUE", VOL_FRAC))
assign(rho, interpolate(Constant(initial_density), DensitySpace))
enforce_density_bounds_inplace(rho)

iter_count = 0
previous_objective = 0.0

num_mma = mesh.num_cells()
xval = np.zeros((num_mma, 1))
xval[:, 0] = rho.vector()
xold1 = np.zeros((num_mma, 1))
xold2 = np.zeros((num_mma, 1))
low = np.zeros((num_mma, 1))
upp = np.zeros((num_mma, 1))

mmma = 1 + len(mass_flow_constraints)
a0 = 1.0
a = np.zeros((mmma, 1))
c = 1.0e4 * np.ones((mmma, 1))
d = np.ones((mmma, 1))

xmin = np.zeros((num_mma, 1))
xmax = np.ones((num_mma, 1))

df0dx = np.zeros((num_mma, 1))
fval = np.zeros((mmma, 1))
dfdx = np.zeros((mmma, num_mma))

volume = assemble(VolumeRegion * dx)
if volume <= 0.0:
    raise ValueError("The volume-constrained design region has zero measure.")

if len(MOVE_LIMIT_SCHEDULE) != len(Q_PENAL_SCHEDULE):
    raise ValueError("MOVE_LIMIT_SCHEDULE must match Q_PENAL_SCHEDULE length.")
if len(BETA_PROJ_SCHEDULE) != len(Q_PENAL_SCHEDULE):
    raise ValueError("BETA_PROJ_SCHEDULE must match Q_PENAL_SCHEDULE length.")

# Allow either one stage limit or one value per stage.
_max_iters_raw = globals().get("MAX_INNER_ITERATIONS_SCHEDULE", globals().get("MAX_INNER_ITERATIONS", 150))
if isinstance(_max_iters_raw, (list, tuple)):
    MAX_INNER_ITERATIONS_SCHEDULE = [int(x) for x in _max_iters_raw]
else:
    MAX_INNER_ITERATIONS_SCHEDULE = [int(_max_iters_raw)] * len(Q_PENAL_SCHEDULE)

# ---------------------------------------------------------------
# Optimization loop.
# ---------------------------------------------------------------
for stage_idx, q_val in enumerate(Q_PENAL_SCHEDULE):
    beta_val = float(BETA_PROJ_SCHEDULE[stage_idx])
    BETA_PROJ.assign(beta_val)
    move_limit_now = MOVE_LIMIT_SCHEDULE[stage_idx]
    max_iters_now = MAX_INNER_ITERATIONS_SCHEDULE[stage_idx]
    q_penal.assign(q_val)
    inner_count = 0
    convergence_history = 0
    objective_converged = False
    root_print("Starting continuation stage {}/{}: q = {:.3f}, beta = {:.2f}, move = {:.4f}".format(
        stage_idx + 1, len(Q_PENAL_SCHEDULE), q_val, beta_val, move_limit_now,
    ))

    while inner_count < max_iters_now and not objective_converged:
        root_print("--- Stage {}/{} | iter {:03d} (global {:03d}) ---".format(
            stage_idx + 1, len(Q_PENAL_SCHEDULE), inner_count, iter_count,
        ))

        # Update the filtered and projected design.
        solver_log("  [Filter] design density")
        rho_f = pde_filter(rho, rho_f) # filtered design variable
        rho_proj_plot.vector()[:] = project(rho_effective, DensitySpace).vector()[:]
        solver_log("  [Wall distance] update")
        update_wall_distance_field()

        rho_out << rho
        rhop_out << rho_proj_plot

        # Build the initial guess once.
        if iter_count == 0:
            solver_log("  [Warm start] Stokes-Brinkman and initial SA field")
            initialize_forward_guess_with_stokes()

        # Forward flow and SA updates.
        root_print("  [Forward solve]")
        picard_steps = max(1, int(globals().get("PICARD_STEPS", globals().get("FROZEN_PICARD_STEPS", 1))))
        for picard_idx in range(picard_steps):
            solver_log("    [Picard {}/{}] flow".format(picard_idx + 1, picard_steps))
            solve_forward_with_recovery("stage{:02d}_iter{:03d}_picard{:02d}".format(
                stage_idx + 1, inner_count, picard_idx + 1,
            ))
            velocity_for_sa = w_fwd.sub(0, deepcopy=True)
            sa_model.construct_forms(velocity_for_sa)
            solver_log("    [Picard {}/{}] SA transport".format(picard_idx + 1, picard_steps))
            sa_model.solve_turbulence_model()
            sa_model.update_variables(
                relaxation=float(globals().get("TURBULENCE_RELAXATION", globals().get("NUT_RELAXATION_FACTOR", 1.0)))
            )
            nu_tilde_frozen.assign(sa_model.nu_tilde0)
            enforce_scalar_floor(nu_tilde_frozen, float(globals().get("SA_NU_TILDE_FLOOR", 1.0e-12)))
            sa_model.nu_tilde0.assign(nu_tilde_frozen)
            sa_model.nu_tilde1.assign(nu_tilde_frozen)

        # Final flow solve with the updated turbulent viscosity.
        solver_log("    [Final flow] IPCS with updated turbulent viscosity")
        final_flow_du_ipcs, final_flow_dp_ipcs = solve_forward_with_recovery("stage{:02d}_iter{:03d}_final".format(
            stage_idx + 1, inner_count,
        ))

        # Adjoint solve.
        root_print("  [Adjoint solve]")
        solve_adjoint(objective_adjoint_form)

        u_out << w_fwd.sub(0)
        p_out << w_fwd.sub(1)
        nu_tilde_out << nu_tilde_frozen

        f0val = assemble(ObjFunctional)
        obj_conv = abs((f0val - previous_objective) / max(abs(f0val), 1e-12))

        if obj_conv < OBJECTIVE_CONVERGENCE_TOL:
            convergence_history += 1
            if convergence_history >= OBJECTIVE_STREAK_TO_STOP:
                objective_converged = True
        else:
            convergence_history = 0
        previous_objective = f0val

        # Objective gradient.
        unfiltered_gradient.vector()[:] = assemble(objective_ddx)[:]
        filtered_gradient = pde_filter(unfiltered_gradient, filtered_gradient)
        np.savetxt(os.path.join(design_dir, "rho_{:03}.txt".format(iter_count)), rho.vector()[:])

        # Volume constraint and gradient.
        fval[0, 0] = assemble(vol_constraint)
        unfiltered_s_vol.vector()[:] = assemble(sensitivities_vol_constraint)[:]
        filtered_s_vol = pde_filter(unfiltered_s_vol, filtered_s_vol)
        vol_fraction_now = assemble(VolumeRegion * rho_effective * dx) / volume
        vol_residual_now = float(fval[0, 0]) / max(volume, 1.0e-12)

        df0dx[:, 0] = filtered_gradient.vector()[:]
        dfdx[0, :] = filtered_s_vol.vector()[:]

        mass_flow_status = []
        mass_flow_status_markers = set()
        for constraint_idx, constraint_spec in enumerate(mass_flow_constraints, start=1):
            solve_adjoint(constraint_spec["adjoint_form"])
            fval[constraint_idx, 0] = (
                assemble(constraint_spec["functional"]) + constraint_spec["offset"]
            )
            unfiltered_constraint_gradient.vector()[:] = assemble(constraint_spec["gradient_form"])[:]
            filtered_constraint_gradient = pde_filter(
                unfiltered_constraint_gradient, filtered_constraint_gradient
            )
            dfdx[constraint_idx, :] = filtered_constraint_gradient.vector()[:]

            marker = constraint_spec["marker"]
            if marker not in mass_flow_status_markers:
                outlet_flow_value = assemble(dot(u, n) * ds(marker))
                outlet_fraction = outlet_flow_value / fixed_inlet_flow_value
                mass_flow_status.append(
                    "m{}={:.3f}/{:.3f}".format(
                        marker,
                        outlet_fraction,
                        constraint_spec["target_fraction"],
                    )
                )
                mass_flow_status_markers.add(marker)

        # MMA update.
        root_print("  [MMA update]")
        (xmma, _ymma, _zmma, _lam, _xsi, _eta, _mu_mma, _zet, _s, low, upp) = mmasub(
            mmma, num_mma, iter_count, xval, xmin, xmax, xold1, xold2,
            f0val, df0dx, fval, dfdx, low, upp, a0, a, c, d, move_limit_now,
        )

        xold2 = xold1.copy()
        xold1 = xval.copy()
        xval = xmma.copy()
        rho_values = np.clip(xmma[:, 0].copy(), density_lower_values, density_upper_values)
        xval[:, 0] = rho_values
        rho.vector()[:] = rho_values
        rho.vector().apply("insert")

        append_optimization_log_entry(
            log_path,
            stage_idx + 1,
            q_val,
            beta_val,
            inner_count,
            iter_count,
            f0val,
            obj_conv,
            vol_fraction_now,
            vol_residual_now,
            du_ipcs=final_flow_du_ipcs,
            dp_ipcs=final_flow_dp_ipcs,
        )

        constraint_status_text = ""
        if mass_flow_status:
            constraint_status_text = " " + " ".join(mass_flow_status)

        root_print("q={:.3f} beta={:.2f} move={:.3f} iter={:03d} J={:.4e} conv={:.3e} vol={:.4f} streak={}/{}{}".format(
            q_val, float(BETA_PROJ.values()[0]), move_limit_now,
            inner_count, f0val, obj_conv, vol_fraction_now,
            convergence_history, OBJECTIVE_STREAK_TO_STOP,
            constraint_status_text,
        ))

        inner_count += 1
        iter_count += 1

    if objective_converged:
        root_print("Stage {}/{} converged after {} iterations.".format(
            stage_idx + 1, len(Q_PENAL_SCHEDULE), inner_count,
        ))
    else:
        root_print("Stage {}/{} reached max iterations ({}).".format(
            stage_idx + 1, len(Q_PENAL_SCHEDULE), max_iters_now,
        ))

root_print("Writing final post-update density output.")
rho_f = pde_filter(rho, rho_f)
rho_proj_plot.vector()[:] = project(rho_effective, DensitySpace).vector()[:]
rho_out << rho
rhop_out << rho_proj_plot
np.savetxt(os.path.join(design_dir, "rho_{:03}.txt".format(iter_count)), rho.vector()[:])

root_print("Optimization finished. Results written to {}".format(results_root))
