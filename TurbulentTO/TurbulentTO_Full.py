from dolfin import *
import numpy as np
import os
from ufl import tanh

from mma import mmasub
from TurbulenceModel_SpalartAllmaras_TO_Full import (
    build_spalart_allmaras_residual,
    sa_turbulent_viscosity,
)
from Utilities_LaminarTO import (
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
from Utilities_TurbulentTO_Full import (
    build_penalized_wall_distance_solver,
    calculate_distance_field,
    nu_tilde_from_viscosity_ratio,
    positive_part,
)

# ====================================================================================
# Monolithic turbulent topology optimization with a full (u, p, nu_tilde) adjoint.
# This first "full" step still keeps the wall-distance field external to the state.
# ====================================================================================

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

mu_fluid = Constant(MU_FLUID_VALUE)
rho_fluid = Constant(RHO_FLUID_VALUE)
brinkman_fluid_length = float(globals().get("BRINKMAN_FLUID_LENGTH", 100.0))
brinkman_solid_length = float(globals().get("BRINKMAN_SOLID_LENGTH", 0.01))
alpha_fluid = Constant(float(globals().get("ALPHA_FLUID", 2.5 * MU_FLUID_VALUE / brinkman_fluid_length ** 2.0)))
alpha_solid = Constant(float(globals().get("ALPHA_SOLID", 2.5 * MU_FLUID_VALUE / brinkman_solid_length ** 2.0)))
q_penal = Constant(0.1)


def projection(rho_design, eta_proj):
    return (
        tanh(BETA_PROJ * Constant(eta_proj))
        + tanh(BETA_PROJ * (rho_design - Constant(eta_proj)))
    ) / (
        tanh(BETA_PROJ * Constant(eta_proj))
        + tanh(BETA_PROJ * (Constant(1.0) - Constant(eta_proj)))
    )


def alpha(brinkman_density):
    return alpha_solid + (alpha_fluid - alpha_solid) * brinkman_density * (1 + q_penal) / (
        brinkman_density + q_penal
    )


def sa_positive_viscosity(state_nu_tilde):
    nu_lam = mu_fluid / rho_fluid
    return sa_turbulent_viscosity(
        state_nu_tilde,
        nu_lam,
        smooth_abs_eps=float(globals().get("SA_SMOOTH_ABS_EPS", 1.0e-12)),
    )


def effective_dynamic_viscosity(state_nu_tilde):
    return mu_fluid + rho_fluid * sa_positive_viscosity(state_nu_tilde)


def build_flow_residual(state_u, state_p, test_u, test_p, rho_eff, custom_dx, state_nu_tilde):
    mu_effective = effective_dynamic_viscosity(state_nu_tilde)
    return (
        rho_fluid * inner(dot(state_u, nabla_grad(state_u)), test_u) * custom_dx
        + mu_effective * inner(grad(state_u), grad(test_u)) * custom_dx
        + inner(grad(state_p), test_u) * custom_dx
        + inner(div(state_u), test_p) * custom_dx
        + alpha(rho_eff) * inner(state_u, test_u) * custom_dx
    )


mesh = create_design_mesh_from_config(globals())

U_h = VectorElement("CG", mesh.ufl_cell(), 2)
P_h = FiniteElement("CG", mesh.ufl_cell(), 1)
T_h = FiniteElement("CG", mesh.ufl_cell(), 1)
A_h = FiniteElement("DG", mesh.ufl_cell(), 0)

StateElement = MixedElement([U_h, P_h, T_h])
StateSpace = FunctionSpace(mesh, StateElement)
StateSpaceAdj = FunctionSpace(mesh, StateElement)
FlowWarmSpace = FunctionSpace(mesh, U_h * P_h)
TurbulenceSpace = FunctionSpace(mesh, T_h)
DensitySpace = FunctionSpace(mesh, A_h)

w_state = Function(StateSpace)
(u, p, nu_tilde) = split(w_state)
w_adj = Function(StateSpaceAdj)
(v, q, xi) = split(w_adj)

rho = Function(DensitySpace)
rho_f = Function(DensitySpace)
rho_proj_plot = Function(DensitySpace)
unfiltered_gradient = Function(DensitySpace)
filtered_gradient = Function(DensitySpace)
unfiltered_s_vol = Function(DensitySpace)
filtered_s_vol = Function(DensitySpace)
unfiltered_constraint_gradient = Function(DensitySpace)
filtered_constraint_gradient = Function(DensitySpace)


def _as_density_function(value, density_space):
    if isinstance(value, Function):
        return value

    out = Function(density_space)
    if np.isscalar(value):
        assign(out, interpolate(Constant(float(value)), density_space))
    else:
        assign(out, interpolate(value, density_space))
    return out


def build_density_bounds_from_config():
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

bcu_walls = [DirichletBC(StateSpace.sub(0), u_noslip, boundaries, m) for m in wall_markers]
bcu_inlet = [DirichletBC(StateSpace.sub(0), prof, boundaries, m) for prof, m in zip(inlet_profiles, inlet_markers)]
bcu_outlet = (
    [DirichletBC(StateSpace.sub(0), prof, boundaries, m) for prof, m in zip(outlet_profiles, outlet_markers)]
    if use_outlet_velocity_bc else []
)
bcp_outlet = (
    [DirichletBC(StateSpace.sub(1), outlet_pressure_value, boundaries, m) for m in outlet_markers]
    if use_outlet_pressure_bc else []
)
bcp_pin = (
    [DirichletBC(
        StateSpace.sub(1), Constant(0.0),
        build_pressure_pin_expression_from_config(globals()), "pointwise",
    )]
    if use_pressure_pin else []
)

adjoint_velocity_bc_builder = globals().get("build_adjoint_velocity_bcs")
if callable(adjoint_velocity_bc_builder):
    bc_state_adj = list(adjoint_velocity_bc_builder(
        StateSpaceAdj, boundaries, wall_markers, inlet_markers, outlet_markers,
        u_noslip, inlet_profiles, outlet_profiles,
    ))
else:
    bc_state_adj = (
        [DirichletBC(StateSpaceAdj.sub(0), u_noslip, boundaries, m) for m in wall_markers]
        + [DirichletBC(StateSpaceAdj.sub(0), u_noslip, boundaries, m) for m in inlet_markers]
    )
    if use_outlet_velocity_bc:
        bc_state_adj += [DirichletBC(StateSpaceAdj.sub(0), u_noslip, boundaries, m) for m in outlet_markers]
if use_outlet_pressure_bc:
    bc_state_adj += [DirichletBC(StateSpaceAdj.sub(1), Constant(0.0), boundaries, m) for m in outlet_markers]
if use_pressure_pin:
    bc_state_adj += [DirichletBC(
        StateSpaceAdj.sub(1), Constant(0.0),
        build_pressure_pin_expression_from_config(globals()), "pointwise",
    )]


def compute_fixed_inlet_flow_rate():
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
    root_print(
        "SA inlet nu_tilde: {}  (nu_lam = {:.3e})".format(
            ["  {:.4e}".format(v) for v in _sa_nu_tilde_targets],
            _nu_lam,
        )
    )
    nu_tilde_inlet_bc_values = [Constant(value) for value in _sa_nu_tilde_targets]

bcn_turbulence_state = (
    [DirichletBC(StateSpace.sub(2), bc_val, boundaries, m) for bc_val, m in zip(nu_tilde_inlet_bc_values, inlet_markers)]
    + [DirichletBC(StateSpace.sub(2), Constant(0.0), boundaries, m) for m in wall_markers]
)
bcn_turbulence_adj = (
    [DirichletBC(StateSpaceAdj.sub(2), Constant(0.0), boundaries, m) for m in inlet_markers]
    + [DirichletBC(StateSpaceAdj.sub(2), Constant(0.0), boundaries, m) for m in wall_markers]
)

bc_state = bcu_walls + bcu_inlet + bcu_outlet + bcp_outlet + bcp_pin + bcn_turbulence_state
bc_state_adj += bcn_turbulence_adj

r_filter = (
    compute_filter_base_length_from_config(globals())
    * float(globals().get("FILTER_RADIUS_IN_CELLS", 3.0))
)
r = r_filter / (2.0 * 3.0 ** 0.5)

u_filter = TrialFunction(DensitySpace)
v_filter = TestFunction(DensitySpace)
filter_in = Function(DensitySpace)
h = CellDiameter(mesh)
h_avg = (h("+") + h("-")) / 2.0


def pde_filter(input_field, output_field):
    alpha_dg = 4.0
    helmholtz = (
        r ** 2 * (alpha_dg / h_avg * dot(jump(v_filter, n), jump(u_filter, n))) * dS
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


rho_projected = projection(rho_f, ETA_I)
rho_effective = density_lower_bound + (density_upper_bound - density_lower_bound) * rho_projected
use_penalized_wall_distance = bool(globals().get("SA_USE_PENALIZED_WALL_DISTANCE", True))
if use_penalized_wall_distance:
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

    wall_distance, update_wall_distance_field = build_penalized_wall_distance_solver(
        TurbulenceSpace,
        boundaries,
        wall_markers,
        dx,
        rho_effective,
        sa_wall_sigma,
        sa_wall_g0,
        sa_wall_penalty_alpha,
        sa_wall_penalty_power,
        sa_wall_g_floor,
        newton_rtol=sa_wall_newton_rtol,
        newton_atol=sa_wall_newton_atol,
        newton_max_iters=sa_wall_newton_max_iters,
        newton_relax=sa_wall_newton_relax,
        penalty_homotopy=sa_wall_penalty_homotopy,
        solid_guess_weight=sa_wall_initial_solid_guess,
        extra_relaxations=sa_wall_extra_relaxations,
    )
    root_print(
        "Penalized SA wall-distance enabled (frozen G): sigma={}, G0={}, alpha_G={}, n_G={}, relax_G={}, maxit_G={}".format(
            sa_wall_sigma,
            sa_wall_g0,
            sa_wall_penalty_alpha,
            sa_wall_penalty_power,
            sa_wall_newton_relax,
            sa_wall_newton_max_iters,
        )
    )
else:
    wall_distance = calculate_distance_field(
        TurbulenceSpace,
        boundaries,
        wall_markers,
        dx,
        relaxation=float(globals().get("SA_DISTANCE_RELAXATION", 0.01)),
    )

    def update_wall_distance_field():
        return None


nu_laminar = Constant(MU_FLUID_VALUE / RHO_FLUID_VALUE)
sa_nu_tilde_penalty_reaction = Constant(float(globals().get("SA_NU_TILDE_PENALTY_ALPHA", 1.0e3))) * (
    positive_part(Constant(1.0) - rho_effective)
    ** float(globals().get("SA_NU_TILDE_PENALTY_N", 3.0))
)

dissipation_density_builder = globals().get("build_dissipation_density")
mu_effective = effective_dynamic_viscosity(nu_tilde)
if callable(dissipation_density_builder):
    dissipation_density = dissipation_density_builder(u, mu_effective)
else:
    deformation = nabla_grad(u) + nabla_grad(u).T
    dissipation_density = 0.5 * mu_effective * inner(deformation, deformation)

ObjFunctional = ObjectiveRegion * (
    dissipation_density + alpha(rho_effective) * inner(u, u)
) * dx

state_form = (
    build_flow_residual(u, p, v, q, rho_effective, dx, nu_tilde)
    + build_spalart_allmaras_residual(
        u,
        nu_tilde,
        xi,
        nu_laminar,
        wall_distance,
        dx,
        nu_tilde_penalty_reaction=sa_nu_tilde_penalty_reaction,
        smooth_abs_eps=float(globals().get("SA_SMOOTH_ABS_EPS", 1.0e-12)),
    )
)
lagrangian_form = ObjFunctional + state_form

state_residual_form = derivative(state_form, w_adj, TestFunction(StateSpace))
objective_adjoint_form = derivative(lagrangian_form, w_state, TestFunction(StateSpaceAdj))
objective_ddx = derivative(lagrangian_form, rho_f)

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
                    upper_constraint_lagrangian_form, w_state, TestFunction(StateSpaceAdj)
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
                        lower_constraint_lagrangian_form, w_state, TestFunction(StateSpaceAdj)
                    ),
                    "gradient_form": derivative(lower_constraint_lagrangian_form, rho_f),
                }
            )

u_warm, p_warm = TrialFunctions(FlowWarmSpace)
v_warm, q_warm = TestFunctions(FlowWarmSpace)
a_stokes = (
    mu_fluid * inner(grad(u_warm), grad(v_warm))
    + inner(grad(p_warm), v_warm)
    + inner(div(u_warm), q_warm)
    + alpha(rho_effective) * inner(u_warm, v_warm)
) * dx
l_stokes = Constant(0.0) * q_warm * dx
w_warm = Function(FlowWarmSpace)

bcu_walls_warm = [DirichletBC(FlowWarmSpace.sub(0), u_noslip, boundaries, m) for m in wall_markers]
bcu_inlet_warm = [DirichletBC(FlowWarmSpace.sub(0), prof, boundaries, m) for prof, m in zip(inlet_profiles, inlet_markers)]
bcu_outlet_warm = (
    [DirichletBC(FlowWarmSpace.sub(0), prof, boundaries, m) for prof, m in zip(outlet_profiles, outlet_markers)]
    if use_outlet_velocity_bc else []
)
bcp_outlet_warm = (
    [DirichletBC(FlowWarmSpace.sub(1), outlet_pressure_value, boundaries, m) for m in outlet_markers]
    if use_outlet_pressure_bc else []
)
bcp_pin_warm = (
    [DirichletBC(
        FlowWarmSpace.sub(1), Constant(0.0),
        build_pressure_pin_expression_from_config(globals()), "pointwise",
    )]
    if use_pressure_pin else []
)
bc_warm = bcu_walls_warm + bcu_inlet_warm + bcu_outlet_warm + bcp_outlet_warm + bcp_pin_warm


def initialize_state_guess_with_stokes():
    solve(
        a_stokes == l_stokes,
        w_warm,
        bc_warm,
        solver_parameters={"linear_solver": SNES_LINEAR_SOLVER},
    )
    assign(w_state.sub(0), w_warm.sub(0))
    assign(w_state.sub(1), w_warm.sub(1))

    sa_nu_tilde_init = float(globals().get(
        "SA_NU_TILDE_INITIAL",
        nu_tilde_from_viscosity_ratio(SA_MUT_RATIO, _nu_lam)
        if "SA_MUT_RATIO" in globals()
        else float(_sa_nu_tilde_targets[0]),
    ))
    wall_dist_scale = max(
        float(globals().get("SA_INIT_WALL_DIST_SCALE", 0.05 * float(globals().get("L", 1.0)))),
        1.0e-12,
    )
    nu_guess = project(
        Constant(sa_nu_tilde_init) * wall_distance / (wall_distance + Constant(wall_dist_scale)),
        TurbulenceSpace,
    )
    assign(w_state.sub(2), nu_guess)


def solve_state_once(method_override=None, line_search_override=None, max_iters_override=None):
    jac_state = derivative(state_residual_form, w_state)
    problem_state = NonlinearVariationalProblem(state_residual_form, w_state, bc_state, jac_state)
    solver_state = NonlinearVariationalSolver(problem_state)
    solver_state.parameters["nonlinear_solver"] = "snes"
    solver_state.parameters["snes_solver"]["linear_solver"] = globals().get(
        "FULL_STATE_LINEAR_SOLVER",
        globals().get("SNES_LINEAR_SOLVER", "mumps"),
    )
    method = method_override or globals().get("FULL_STATE_SNES_METHOD", "newtonls")
    solver_state.parameters["snes_solver"]["method"] = method
    if method == "newtonls":
        solver_state.parameters["snes_solver"]["line_search"] = globals().get(
            "FULL_STATE_SNES_LINE_SEARCH" if line_search_override is None else "__unused__",
            "bt" if line_search_override is None else line_search_override,
        )
    solver_state.parameters["snes_solver"]["relative_tolerance"] = float(
        globals().get("FULL_STATE_SNES_RTOL", globals().get("FORWARD_SNES_RTOL", 1.0e-6))
    )
    solver_state.parameters["snes_solver"]["absolute_tolerance"] = float(
        globals().get("FULL_STATE_SNES_ATOL", globals().get("FORWARD_SNES_ATOL", 1.0e-8))
    )
    solver_state.parameters["snes_solver"]["maximum_iterations"] = int(
        globals().get("FULL_STATE_SNES_MAX_ITERS", globals().get("SNES_MAX_ITERS", 80))
        if max_iters_override is None
        else max_iters_override
    )
    solver_state.parameters["snes_solver"]["error_on_nonconvergence"] = True
    solver_state.solve()


def solve_state_with_recovery():
    try:
        solve_state_once()
    except RuntimeError:
        if not bool(globals().get("FULL_STATE_RESTART_WITH_STOKES", True)):
            raise
        root_print("  State solve diverged; rebuilding Stokes-Brinkman warm start and retrying once.")
        initialize_state_guess_with_stokes()
        fallback_method = globals().get(
            "FULL_STATE_SNES_FALLBACK_METHOD",
            globals().get("FULL_STATE_SNES_METHOD", "newtonls"),
        )
        fallback_line_search = globals().get(
            "FULL_STATE_SNES_FALLBACK_LINE_SEARCH",
            globals().get("FULL_STATE_SNES_LINE_SEARCH", "bt"),
        )
        fallback_max_iters = int(globals().get(
            "FULL_STATE_SNES_FALLBACK_MAX_ITERS",
            globals().get("FULL_STATE_SNES_MAX_ITERS", globals().get("SNES_MAX_ITERS", 80)),
        ))
        solve_state_once(
            method_override=fallback_method,
            line_search_override=fallback_line_search,
            max_iters_override=fallback_max_iters,
        )


def solve_adjoint(adjoint_residual_form=objective_adjoint_form):
    solver_log("    [Adjoint] linear system")
    w_adj.vector().zero()
    A_adj = assemble(derivative(adjoint_residual_form, w_adj))
    b_adj = assemble(adjoint_residual_form)
    b_adj *= -1.0
    for bc in bc_state_adj:
        bc.apply(A_adj, b_adj)
    solve(A_adj, w_adj.vector(), b_adj, globals().get("SNES_LINEAR_SOLVER", "mumps"))


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
results_root_name = globals().get("RESULTS_ROOT_NAME_FULL")
if results_root_name is None:
    results_root_name = "{}_Full".format(
        globals().get("RESULTS_ROOT_NAME", "Results_Full/Results_TurbulentTO")
    )
results_root = os.path.join(THIS_DIR, results_root_name)
rho_dir = os.path.join(results_root, "rho")
rho_p_dir = os.path.join(results_root, "rho_projected")
u_dir = os.path.join(results_root, "u")
p_dir = os.path.join(results_root, "p")
nu_tilde_dir = os.path.join(results_root, "nu_tilde")
design_dir = os.path.join(results_root, "design")

ensure_clean_dir(results_root)
ensure_clean_dir(rho_dir)
ensure_clean_dir(rho_p_dir)
ensure_clean_dir(u_dir)
ensure_clean_dir(p_dir)
ensure_clean_dir(nu_tilde_dir)
ensure_clean_dir(design_dir)

rho_out = ResilientVTKFile(os.path.join(rho_dir, "plot_rho.pvd"), COMM)
rhop_out = ResilientVTKFile(os.path.join(rho_p_dir, "plot_rho_projected.pvd"), COMM)
u_out = ResilientVTKFile(os.path.join(u_dir, "plot_u.pvd"), COMM)
p_out = ResilientVTKFile(os.path.join(p_dir, "plot_p.pvd"), COMM)
nu_tilde_out = ResilientVTKFile(os.path.join(nu_tilde_dir, "plot_nu_tilde.pvd"), COMM)

log_path = os.path.join(results_root, "OptimizationLog.txt")
initialize_optimization_log(log_path)

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

_max_iters_raw = globals().get("MAX_INNER_ITERATIONS_SCHEDULE", globals().get("MAX_INNER_ITERATIONS", 150))
if isinstance(_max_iters_raw, (list, tuple)):
    MAX_INNER_ITERATIONS_SCHEDULE = [int(x) for x in _max_iters_raw]
else:
    MAX_INNER_ITERATIONS_SCHEDULE = [int(_max_iters_raw)] * len(Q_PENAL_SCHEDULE)


for stage_idx, q_val in enumerate(Q_PENAL_SCHEDULE):
    beta_val = float(BETA_PROJ_SCHEDULE[stage_idx])
    BETA_PROJ.assign(beta_val)
    move_limit_now = MOVE_LIMIT_SCHEDULE[stage_idx]
    max_iters_now = MAX_INNER_ITERATIONS_SCHEDULE[stage_idx]
    q_penal.assign(q_val)
    inner_count = 0
    convergence_history = 0
    objective_converged = False
    root_print(
        "Starting continuation stage {}/{}: q = {:.3f}, beta = {:.2f}, move = {:.4f}".format(
            stage_idx + 1, len(Q_PENAL_SCHEDULE), q_val, beta_val, move_limit_now,
        )
    )

    while inner_count < max_iters_now and not objective_converged:
        root_print(
            "--- Stage {}/{} | iter {:03d} (global {:03d}) ---".format(
                stage_idx + 1, len(Q_PENAL_SCHEDULE), inner_count, iter_count,
            )
        )

        solver_log("  [Filter] design density")
        rho_f = pde_filter(rho, rho_f)
        rho_proj_plot.vector()[:] = project(rho_effective, DensitySpace).vector()[:]
        solver_log("  [Wall distance] update")
        update_wall_distance_field()

        rho_out << rho
        rhop_out << rho_proj_plot

        if iter_count == 0:
            solver_log("  [Warm start] Stokes-Brinkman plus SA initialization")
            initialize_state_guess_with_stokes()

        root_print("  [State solve]")
        solve_state_with_recovery()

        root_print("  [Adjoint solve]")
        solve_adjoint(objective_adjoint_form)

        u_out << w_state.sub(0)
        p_out << w_state.sub(1)
        nu_tilde_out << w_state.sub(2)

        f0val = assemble(ObjFunctional)
        obj_conv = abs((f0val - previous_objective) / max(abs(f0val), 1.0e-12))

        if obj_conv < OBJECTIVE_CONVERGENCE_TOL:
            convergence_history += 1
            if convergence_history >= OBJECTIVE_STREAK_TO_STOP:
                objective_converged = True
        else:
            convergence_history = 0
        previous_objective = f0val

        unfiltered_gradient.vector()[:] = assemble(objective_ddx)[:]
        filtered_gradient = pde_filter(unfiltered_gradient, filtered_gradient)

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
        )

        constraint_status_text = ""
        if mass_flow_status:
            constraint_status_text = " " + " ".join(mass_flow_status)

        root_print(
            "q={:.3f} beta={:.2f} move={:.3f} iter={:03d} J={:.4e} conv={:.3e} vol={:.4f} streak={}/{}{}".format(
                q_val, float(BETA_PROJ.values()[0]), move_limit_now,
                inner_count, f0val, obj_conv, vol_fraction_now,
                convergence_history, OBJECTIVE_STREAK_TO_STOP,
                constraint_status_text,
            )
        )

        inner_count += 1
        iter_count += 1

    if objective_converged:
        root_print(
            "Stage {}/{} converged after {} iterations.".format(
                stage_idx + 1, len(Q_PENAL_SCHEDULE), inner_count,
            )
        )
    else:
        root_print(
            "Stage {}/{} reached max iterations ({}).".format(
                stage_idx + 1, len(Q_PENAL_SCHEDULE), max_iters_now,
            )
        )

root_print("Optimization finished. Results written to {}".format(results_root))
