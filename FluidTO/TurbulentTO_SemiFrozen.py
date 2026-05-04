import os
from Runtime_Setup import configure_writable_runtime_environment

configure_writable_runtime_environment(base_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".runtime"))

from dolfin import *
import numpy as np
import time
try:
    from ufl import tanh
except ModuleNotFoundError:
    from ufl_legacy import tanh

from mma import mmasub
from TurbulenceModel_SpalartAllmaras_TO import (
    build_spalart_allmaras_residual,
    sa_transport_terms,
    sa_turbulent_viscosity,
)
from Utilities_SharedTO import (
    append_optimization_log_entry,
    as_list,
    build_design_pressure_drop_markers,
    build_sa_inlet_nu_tilde_targets,
    build_pressure_pin_expression_from_config,
    compute_filter_base_length_from_config,
    create_design_mesh_from_config,
    ensure_clean_dir,
    initialize_optimization_log,
    load_config_module_from_cli,
    pressure_drop_between_boundaries,
    pressure_drop_between_internal_facets,
    ResilientVTKFile,
)
from Utilities_TurbulentTO import (
    build_penalized_reciprocal_distance_residual,
    build_penalized_wall_distance_solver,
    enforce_scalar_floor,
    initialize_penalized_reciprocal_distance,
    nu_tilde_from_viscosity_ratio,
    positive_part,
    wall_distance_from_reciprocal_distance,
)

# ====================================================================================
# Shared implementation for the SemiFrozen and Full turbulent adjoints.
# By default the reciprocal wall-distance G is updated externally, yielding the
# SemiFrozen adjoint. The TurbulentTO_Full entrypoint activates the 4-field
# variant where G is promoted into the primal/adjoint state too.
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
# The TurbulentTO_Full.py wrapper flips this environment variable so the main
# implementation can stay shared in one file.
FULL_STATE_INCLUDE_G = str(os.environ.get("TURBULENTTO_INCLUDE_G_STATE", "0")).strip().lower() in {
    "1", "true", "yes", "on",
}

SHOW_SOLVE_LABELS = bool(globals().get("SHOW_SOLVE_LABELS", True))
SHOW_DOLFIN_SOLVER_LOGS = bool(globals().get("SHOW_DOLFIN_SOLVER_LOGS", False))
LINEAR_SOLVER_NAME = str(globals().get("LINEAR_SOLVER", globals().get("SNES_LINEAR_SOLVER", "mumps")))


def get_state_option(name, legacy_name, default):
    return globals().get(name, globals().get(legacy_name, default))

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
state_turbulence_coupling_weight = Constant(
    float(get_state_option("STATE_TURBULENCE_COUPLING_WEIGHT", "FULL_STATE_TURBULENCE_COUPLING_WEIGHT", 1.0))
)
state_convection_coupling_weight = Constant(
    float(get_state_option("STATE_CONVECTION_COUPLING_WEIGHT", "FULL_STATE_CONVECTION_COUPLING_WEIGHT", 1.0))
)


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
    return mu_fluid + state_turbulence_coupling_weight * rho_fluid * sa_positive_viscosity(state_nu_tilde)


def build_flow_residual(state_u, state_p, test_u, test_p, rho_eff, custom_dx, state_nu_tilde):
    mu_effective = effective_dynamic_viscosity(state_nu_tilde)
    return (
        state_convection_coupling_weight * rho_fluid * inner(dot(state_u, nabla_grad(state_u)), test_u) * custom_dx
        + mu_effective * inner(grad(state_u), grad(test_u)) * custom_dx
        + inner(grad(state_p), test_u) * custom_dx
        + inner(div(state_u), test_p) * custom_dx
        + alpha(rho_eff) * inner(state_u, test_u) * custom_dx
    )


mesh = create_design_mesh_from_config(globals())

U_h = VectorElement("CG", mesh.ufl_cell(), 2)
P_h = FiniteElement("CG", mesh.ufl_cell(), 1)
T_h = FiniteElement("CG", mesh.ufl_cell(), 1)
G_h = FiniteElement("CG", mesh.ufl_cell(), 1)
A_h = FiniteElement("DG", mesh.ufl_cell(), 0)

# SemiFrozen state: (u, p, nu_tilde)
# Full state:       (u, p, nu_tilde, G)
if FULL_STATE_INCLUDE_G:
    StateElement = MixedElement([U_h, P_h, T_h, G_h])
else:
    StateElement = MixedElement([U_h, P_h, T_h])
StateSpace = FunctionSpace(mesh, StateElement)
StateSpaceAdj = FunctionSpace(mesh, StateElement)
FlowWarmSpace = FunctionSpace(mesh, U_h * P_h)
TurbulenceSpace = FunctionSpace(mesh, T_h)
WallDistanceSpace = FunctionSpace(mesh, G_h) if FULL_STATE_INCLUDE_G else TurbulenceSpace
DensitySpace = FunctionSpace(mesh, A_h)

STATE_VEL_IDX = 0
STATE_P_IDX = 1
STATE_TURB_IDX = 2
STATE_G_IDX = 3 if FULL_STATE_INCLUDE_G else None

w_state = Function(StateSpace)
w_adj = Function(StateSpaceAdj)
if FULL_STATE_INCLUDE_G:
    (u, p, nu_tilde, wall_reciprocal_distance) = split(w_state)
    (v, q, xi, zeta_g) = split(w_adj)
else:
    (u, p, nu_tilde) = split(w_state)
    (v, q, xi) = split(w_adj)
    wall_reciprocal_distance = None
    zeta_g = None

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


def _as_scalar_dirichlet_value(value):
    if np.isscalar(value):
        return Constant(float(value))
    return value


def _normalize_velocity_component_bc_specs(raw_specs, marker_lookup):
    if raw_specs is None:
        return []

    normalized_specs = []
    for spec in as_list(raw_specs):
        if not isinstance(spec, dict):
            raise TypeError("Velocity component BC specs must be dictionaries.")

        if "component" not in spec:
            raise ValueError("Velocity component BC specs must define 'component'.")

        component = int(spec["component"])
        if component not in (0, 1):
            raise ValueError("Velocity component BC 'component' must be 0 or 1.")

        marker_spec = spec.get("marker", "outlet")
        if isinstance(marker_spec, str):
            if marker_spec not in marker_lookup:
                raise ValueError(
                    "Unknown marker name '{}' in velocity component BC.".format(marker_spec)
                )
            marker_values = as_list(marker_lookup[marker_spec])
        else:
            marker_values = as_list(marker_spec)

        value = _as_scalar_dirichlet_value(spec.get("value", 0.0))
        for marker_value in marker_values:
            normalized_specs.append(
                {
                    "marker": int(marker_value),
                    "component": component,
                    "value": value,
                }
            )

    return normalized_specs


mark = MARK
boundaries = globals()["mark_boundaries"](mesh)
wall_markers = as_list(mark["walls"])
inlet_markers = as_list(mark["inlet"])
outlet_markers = as_list(mark["outlet"])
density_lower_bound, density_upper_bound = build_density_bounds_from_config()
density_lower_values = density_lower_bound.vector().get_local()
density_upper_values = density_upper_bound.vector().get_local()
VolumeRegion = build_region_function_from_config("build_volume_region", 1.0)
if callable(globals().get("build_objective_region")):
    ObjectiveRegion = build_region_function_from_config("build_objective_region", 1.0)
else:
    ObjectiveRegion = VolumeRegion

if "QUADRATURE_DEGREE" in globals():
    quadrature_degree = int(QUADRATURE_DEGREE)
    parameters["form_compiler"]["quadrature_degree"] = quadrature_degree
    measure_metadata = {"quadrature_degree": quadrature_degree}
    dx = Measure("dx", domain=mesh, metadata=measure_metadata)
    ds = Measure("ds", domain=mesh, subdomain_data=boundaries, metadata=measure_metadata)
    dS = Measure("dS", domain=mesh, metadata=measure_metadata)
else:
    measure_metadata = {}
    dx = Measure("dx", domain=mesh)
    ds = Measure("ds", domain=mesh, subdomain_data=boundaries)
    dS = Measure("dS", domain=mesh)
design_pressure_drop_facets, design_pressure_drop_mark = build_design_pressure_drop_markers(mesh, globals())
dS_design_pressure = Measure(
    "dS",
    domain=mesh,
    subdomain_data=design_pressure_drop_facets,
    metadata=measure_metadata,
)
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
    root_print("Outlet BC type: pressure")
else:
    root_print("Outlet BC type: velocity")

pressure_outlet_component_bcs = _normalize_velocity_component_bc_specs(
    globals().get("PRESSURE_OUTLET_COMPONENT_BCS"),
    mark,
)
if pressure_outlet_component_bcs and not use_outlet_pressure_bc:
    raise ValueError("PRESSURE_OUTLET_COMPONENT_BCS require OUTLET_BC_TYPE = 'pressure'.")
if pressure_outlet_component_bcs:
    root_print(
        "Pressure-outlet velocity component BCs: {}".format(
            ", ".join(
                "marker {} -> u[{}] = {}".format(
                    spec["marker"], spec["component"], spec["value"]
                )
                for spec in pressure_outlet_component_bcs
            )
        )
    )

bcu_walls = [DirichletBC(StateSpace.sub(STATE_VEL_IDX), u_noslip, boundaries, m) for m in wall_markers]
bcu_inlet = [DirichletBC(StateSpace.sub(STATE_VEL_IDX), prof, boundaries, m) for prof, m in zip(inlet_profiles, inlet_markers)]
bcu_outlet = (
    [DirichletBC(StateSpace.sub(STATE_VEL_IDX), prof, boundaries, m) for prof, m in zip(outlet_profiles, outlet_markers)]
    if use_outlet_velocity_bc else []
)
bcu_pressure_outlet_components = [
    DirichletBC(
        StateSpace.sub(STATE_VEL_IDX).sub(spec["component"]),
        spec["value"],
        boundaries,
        spec["marker"],
    )
    for spec in pressure_outlet_component_bcs
]
bcp_outlet = (
    [DirichletBC(StateSpace.sub(STATE_P_IDX), outlet_pressure_value, boundaries, m) for m in outlet_markers]
    if use_outlet_pressure_bc else []
)
bcp_pin = (
    [DirichletBC(
        StateSpace.sub(STATE_P_IDX), Constant(0.0),
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
        [DirichletBC(StateSpaceAdj.sub(STATE_VEL_IDX), u_noslip, boundaries, m) for m in wall_markers]
        + [DirichletBC(StateSpaceAdj.sub(STATE_VEL_IDX), u_noslip, boundaries, m) for m in inlet_markers]
    )
    if use_outlet_velocity_bc:
        bc_state_adj += [DirichletBC(StateSpaceAdj.sub(STATE_VEL_IDX), u_noslip, boundaries, m) for m in outlet_markers]
    bc_state_adj += [
        DirichletBC(
            StateSpaceAdj.sub(STATE_VEL_IDX).sub(spec["component"]),
            Constant(0.0),
            boundaries,
            spec["marker"],
        )
        for spec in pressure_outlet_component_bcs
    ]
if use_outlet_pressure_bc:
    bc_state_adj += [DirichletBC(StateSpaceAdj.sub(STATE_P_IDX), Constant(0.0), boundaries, m) for m in outlet_markers]
if use_pressure_pin:
    bc_state_adj += [DirichletBC(
        StateSpaceAdj.sub(STATE_P_IDX), Constant(0.0),
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

# Build SA inlet values from custom profiles, turbulence-intensity inputs,
# direct eddy-viscosity ratios, or direct nu_tilde values.
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
    root_print("SA inlet nu_tilde: custom profiles  (nu_lam = {:.3e})".format(_nu_lam))
    _sa_nu_tilde_initial_default = _nu_lam
else:
    _sa_nu_tilde_targets, _sa_inlet_description = build_sa_inlet_nu_tilde_targets(
        globals(),
        len(inlet_markers),
        _nu_lam,
        nu_tilde_from_viscosity_ratio,
    )
    _sa_nu_tilde_initial_default = float(_sa_nu_tilde_targets[0])
    root_print("SA inlet setup: {}".format(_sa_inlet_description))
    root_print(
        "SA inlet nu_tilde: {}  (nu_lam = {:.3e})".format(
            ", ".join("{:.4e}".format(v) for v in _sa_nu_tilde_targets),
            _nu_lam,
        )
    )
    nu_tilde_inlet_bc_values = [Constant(value) for value in _sa_nu_tilde_targets]

bcn_turbulence_state = (
    [DirichletBC(StateSpace.sub(STATE_TURB_IDX), bc_val, boundaries, m) for bc_val, m in zip(nu_tilde_inlet_bc_values, inlet_markers)]
    + [DirichletBC(StateSpace.sub(STATE_TURB_IDX), Constant(0.0), boundaries, m) for m in wall_markers]
)
bcn_turbulence_warm = (
    [DirichletBC(TurbulenceSpace, bc_val, boundaries, m) for bc_val, m in zip(nu_tilde_inlet_bc_values, inlet_markers)]
    + [DirichletBC(TurbulenceSpace, Constant(0.0), boundaries, m) for m in wall_markers]
)
bcn_turbulence_adj = (
    [DirichletBC(StateSpaceAdj.sub(STATE_TURB_IDX), Constant(0.0), boundaries, m) for m in inlet_markers]
    + [DirichletBC(StateSpaceAdj.sub(STATE_TURB_IDX), Constant(0.0), boundaries, m) for m in wall_markers]
)

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
# Build the wall-distance model for the current design. In SemiFrozen this
# remains an external update; the Full entrypoint promotes penalized G into the
# monolithic state so the adjoint sees it too.
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

wall_distance_state_residual = None
g_state_initial_guess = None
bcg_state = []
bcg_adj = []

if FULL_STATE_INCLUDE_G:
    # Full activation point:
    # - G is initialized as an actual state variable
    # - wall_distance becomes y(G) inside the monolithic residual
    # - the G-equation and its BCs are appended to the primal/adjoint system
    g_state_initial_guess = initialize_penalized_reciprocal_distance(
        WallDistanceSpace,
        boundaries,
        wall_markers,
        dx,
        sa_wall_sigma,
        sa_wall_g0,
        sa_wall_g_floor,
        initial_wall_distance=custom_initial_wall_distance,
    )
    wall_distance = wall_distance_from_reciprocal_distance(
        wall_reciprocal_distance,
        sa_wall_g0,
        sa_wall_g_floor,
        smooth_abs_eps=float(globals().get("SA_WALL_G_SMOOTH_ABS_EPS", 1.0e-12)),
    )
    wall_distance_state_residual = build_penalized_reciprocal_distance_residual(
        wall_reciprocal_distance,
        zeta_g,
        dx,
        wall_penalty_fluid_indicator,
        sa_wall_sigma,
        sa_wall_g0,
        sa_wall_penalty_alpha,
        sa_wall_penalty_power,
        solid_threshold=sa_wall_solid_threshold,
    )
    bcg_state = [
        DirichletBC(StateSpace.sub(STATE_G_IDX), Constant(sa_wall_g0), boundaries, marker)
        for marker in as_list(wall_markers)
    ]
    bcg_adj = [
        DirichletBC(StateSpaceAdj.sub(STATE_G_IDX), Constant(0.0), boundaries, marker)
        for marker in as_list(wall_markers)
    ]

    def update_wall_distance_field():
        return None

else:
    # SemiFrozen mode: solve G externally, reconstruct y from that solve, and
    # keep the wall-distance update outside the adjointed state system.
    wall_distance, update_wall_distance_field = build_penalized_wall_distance_solver(
        TurbulenceSpace,
        boundaries,
        wall_markers,
        dx,
        wall_penalty_fluid_indicator,
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
        solid_threshold=sa_wall_solid_threshold,
        initial_wall_distance=custom_initial_wall_distance,
        prefer_pseudo_time=sa_wall_prefer_pseudo_time,
    )

bc_state = (
    bcu_walls
    + bcu_inlet
    + bcu_outlet
    + bcu_pressure_outlet_components
    + bcp_outlet
    + bcp_pin
    + bcn_turbulence_state
    + bcg_state
)
bc_state_adj += bcn_turbulence_adj + bcg_adj


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
DissipationFunctional = ObjectiveRegion * dissipation_density * dx

# Monolithic primal residual for the current design. Full simply adds the
# reciprocal wall-distance equation to this same state system.
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
        wall_distance_floor=float(globals().get("SA_WALL_DISTANCE_FLOOR", 0.0)),
        nu_tilde_floor=float(globals().get("SA_NU_TILDE_FLOOR", 0.0)),
    )
)
if FULL_STATE_INCLUDE_G:
    state_form += wall_distance_state_residual
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
bcu_pressure_outlet_components_warm = [
    DirichletBC(
        FlowWarmSpace.sub(0).sub(spec["component"]),
        spec["value"],
        boundaries,
        spec["marker"],
    )
    for spec in pressure_outlet_component_bcs
]
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
bc_warm = (
    bcu_walls_warm
    + bcu_inlet_warm
    + bcu_outlet_warm
    + bcu_pressure_outlet_components_warm
    + bcp_outlet_warm
    + bcp_pin_warm
)
sa_warm_trial = TrialFunction(TurbulenceSpace)
sa_warm_test = TestFunction(TurbulenceSpace)
sa_warm_previous = Function(TurbulenceSpace)
sa_warm_next = Function(TurbulenceSpace)


def run_sa_warm_start_sweeps(external_velocity):
    num_sweeps = max(0, int(get_state_option("STATE_INITIAL_SA_SWEEPS", "FULL_STATE_INITIAL_SA_SWEEPS", 0)))
    if num_sweeps <= 0:
        return

    relaxation = min(
        max(float(get_state_option("STATE_INITIAL_SA_RELAXATION", "FULL_STATE_INITIAL_SA_RELAXATION", 0.5)), 0.0),
        1.0,
    )
    linear_solver = str(
        get_state_option(
            "STATE_INITIAL_SA_LINEAR_SOLVER",
            "FULL_STATE_INITIAL_SA_LINEAR_SOLVER",
            get_state_option("STATE_LINEAR_SOLVER", "FULL_STATE_LINEAR_SOLVER", LINEAR_SOLVER_NAME),
        )
    )
    nu_floor = float(globals().get("SA_NU_TILDE_FLOOR", 1.0e-12))
    smooth_abs_eps = float(globals().get("SA_SMOOTH_ABS_EPS", 1.0e-12))
    wall_distance_floor = float(globals().get("SA_WALL_DISTANCE_FLOOR", 0.0))

    assign(sa_warm_previous, w_state.sub(STATE_TURB_IDX))
    enforce_scalar_floor(sa_warm_previous, nu_floor)

    for sweep_idx in range(num_sweeps):
        sigma, react_nt, source_nt, sa_warm_safe = sa_transport_terms(
            external_velocity,
            sa_warm_previous,
            nu_laminar,
            wall_distance,
            smooth_abs_eps=smooth_abs_eps,
            wall_distance_floor=wall_distance_floor,
            nu_tilde_floor=nu_floor,
        )
        sa_warm_form = (
            dot(external_velocity, nabla_grad(sa_warm_trial)) * sa_warm_test * dx
            + inner((nu_laminar + sa_warm_safe) / sigma * grad(sa_warm_trial), grad(sa_warm_test)) * dx
            + (react_nt + sa_nu_tilde_penalty_reaction) * sa_warm_trial * sa_warm_test * dx
            - source_nt * sa_warm_test * dx
        )
        sa_warm_matrix = assemble(lhs(sa_warm_form))
        sa_warm_rhs = assemble(rhs(sa_warm_form))
        for bc in bcn_turbulence_warm:
            bc.apply(sa_warm_matrix, sa_warm_rhs)
        solve(sa_warm_matrix, sa_warm_next.vector(), sa_warm_rhs, linear_solver)
        enforce_scalar_floor(sa_warm_next, nu_floor)

        if relaxation < 1.0:
            relaxed_values = (
                relaxation * sa_warm_next.vector().get_local()
                + (1.0 - relaxation) * sa_warm_previous.vector().get_local()
            )
            sa_warm_previous.vector().set_local(relaxed_values)
            sa_warm_previous.vector().apply("insert")
        else:
            sa_warm_previous.assign(sa_warm_next)
        enforce_scalar_floor(sa_warm_previous, nu_floor)
        solver_log("    [Warm SA] sweep {}/{}".format(sweep_idx + 1, num_sweeps))

    assign(w_state.sub(STATE_TURB_IDX), sa_warm_previous)


# Warm-start velocity/pressure from Stokes-Brinkman, then seed nu_tilde from
# the current wall distance before the first monolithic nonlinear solve.
def initialize_state_guess_with_stokes(reset_g=False):
    solve(
        a_stokes == l_stokes,
        w_warm,
        bc_warm,
        solver_parameters={"linear_solver": LINEAR_SOLVER_NAME},
    )
    assign(w_state.sub(STATE_VEL_IDX), w_warm.sub(STATE_VEL_IDX))
    assign(w_state.sub(STATE_P_IDX), w_warm.sub(STATE_P_IDX))

    if FULL_STATE_INCLUDE_G and g_state_initial_guess is not None:
        current_g = w_state.sub(STATE_G_IDX, deepcopy=True)
        current_g_norm = float(current_g.vector().norm("linf"))
        should_reset_g = bool(reset_g) or current_g_norm <= 1.0e-14
        if should_reset_g:
            assign(w_state.sub(STATE_G_IDX), g_state_initial_guess)
            state_g_projected = w_state.sub(STATE_G_IDX, deepcopy=True)
            enforce_scalar_floor(
                state_g_projected,
                float(globals().get("SA_WALL_G_FLOOR", 1.0e-8)),
            )
            assign(w_state.sub(STATE_G_IDX), state_g_projected)

    sa_nu_tilde_init = float(globals().get("SA_NU_TILDE_INITIAL", _sa_nu_tilde_initial_default))
    wall_dist_scale = max(
        float(globals().get("SA_INIT_WALL_DIST_SCALE", 0.05 * float(globals().get("L", 1.0)))),
        1.0e-12,
    )
    nu_guess = project(
        Constant(sa_nu_tilde_init) * wall_distance / (wall_distance + Constant(wall_dist_scale)),
        TurbulenceSpace,
    )
    assign(w_state.sub(STATE_TURB_IDX), nu_guess)
    run_sa_warm_start_sweeps(w_state.sub(STATE_VEL_IDX, deepcopy=True))


# ===============================================================
# Nonlinear state solve
# ===============================================================
def solve_state_once(
    method_override=None,
    line_search_override=None,
    max_iters_override=None,
    rtol_override=None,
    atol_override=None,
    accept_function_norm_override=None,
    accept_nonconverged_override=None,
    accept_residual_growth_override=None,
):
    def constrained_state_residual_norm(state_function):
        residual_vector = assemble(state_residual_form)
        fallback_residual_values = None
        current_state_values = None
        for bc in bc_state:
            try:
                # For nonzero Dirichlet data, the constrained residual must be
                # measured as x - g, not overwritten with the prescribed value.
                bc.apply(residual_vector, state_function.vector())
            except TypeError:
                if fallback_residual_values is None:
                    fallback_residual_values = residual_vector.get_local()
                    current_state_values = state_function.vector().get_local()
                for dof, value in bc.get_boundary_values().items():
                    fallback_residual_values[dof] = current_state_values[dof] - float(value)
        if fallback_residual_values is not None:
            residual_vector.set_local(fallback_residual_values)
            residual_vector.apply("insert")
        return float(residual_vector.norm("l2"))

    jac_state = derivative(state_residual_form, w_state)
    problem_state = NonlinearVariationalProblem(state_residual_form, w_state, bc_state, jac_state)
    solver_state = NonlinearVariationalSolver(problem_state)
    solver_state.parameters["nonlinear_solver"] = "snes"
    solver_state.parameters["snes_solver"]["linear_solver"] = str(
        get_state_option("STATE_LINEAR_SOLVER", "FULL_STATE_LINEAR_SOLVER", LINEAR_SOLVER_NAME)
    )
    method = method_override or str(get_state_option("STATE_SOLVE_METHOD", "FULL_STATE_SNES_METHOD", "newtonls"))
    solver_state.parameters["snes_solver"]["method"] = method
    if method == "newtonls":
        solver_state.parameters["snes_solver"]["line_search"] = (
            str(get_state_option("STATE_LINE_SEARCH", "FULL_STATE_SNES_LINE_SEARCH", "bt"))
            if line_search_override is None
            else line_search_override
        )
    current_rtol = float(
        get_state_option("STATE_RTOL", "FULL_STATE_SNES_RTOL", globals().get("FORWARD_SNES_RTOL", 1.0e-6))
        if rtol_override is None
        else rtol_override
    )
    current_atol = float(
        get_state_option("STATE_ATOL", "FULL_STATE_SNES_ATOL", globals().get("FORWARD_SNES_ATOL", 1.0e-8))
        if atol_override is None
        else atol_override
    )
    solver_state.parameters["snes_solver"]["relative_tolerance"] = current_rtol
    solver_state.parameters["snes_solver"]["absolute_tolerance"] = current_atol
    solver_state.parameters["snes_solver"]["maximum_iterations"] = int(
        get_state_option("STATE_MAX_ITERS", "FULL_STATE_SNES_MAX_ITERS", globals().get("SNES_MAX_ITERS", 80))
        if max_iters_override is None
        else max_iters_override
    )
    initial_residual_norm = constrained_state_residual_norm(w_state)
    accepted_function_norm = (
        max(current_atol, current_rtol * max(initial_residual_norm, 1.0e-16))
        if accept_function_norm_override is None
        else float(accept_function_norm_override)
    )
    accepted_function_norm_factor = max(
        1.0,
        float(get_state_option("STATE_ACCEPTED_RESIDUAL_FACTOR", "FULL_STATE_ACCEPTED_RESIDUAL_FACTOR", 2.0)),
    )
    accepted_function_norm_limit = accepted_function_norm * accepted_function_norm_factor
    if accept_residual_growth_override is not None:
        accepted_function_norm_limit = max(
            accepted_function_norm_limit,
            float(accept_residual_growth_override) * initial_residual_norm,
        )
    accept_nonconverged = bool(
        get_state_option(
            "STATE_ACCEPT_NONCONVERGED_WITH_ACCEPT_NORM",
            "FULL_STATE_ACCEPT_NONCONVERGED_WITH_ACCEPT_NORM",
            True,
        )
        if accept_nonconverged_override is None
        else accept_nonconverged_override
    )
    error_on_nonconvergence = bool(
        get_state_option("STATE_ERROR_ON_NONCONVERGENCE", "FULL_STATE_ERROR_ON_NONCONVERGENCE", True)
    )
    if accept_function_norm_override is not None and accept_nonconverged:
        error_on_nonconvergence = False
    solver_state.parameters["snes_solver"]["error_on_nonconvergence"] = error_on_nonconvergence

    solve_result = solver_state.solve()
    solver_converged = True
    if isinstance(solve_result, tuple) and len(solve_result) >= 2:
        solver_converged = bool(solve_result[1])

    residual_norm = constrained_state_residual_norm(w_state)
    if not np.isfinite(residual_norm):
        raise RuntimeError("Accepted SNES iterate produced a non-finite state residual norm.")
    if residual_norm > accepted_function_norm_limit:
        raise RuntimeError(
            "Accepted SNES iterate has state residual norm {:.4e}, exceeding the allowed {:.4e} "
            "(base atol {:.4e}, factor {:.2f}).".format(
                residual_norm,
                accepted_function_norm_limit,
                accepted_function_norm,
                accepted_function_norm_factor,
            )
        )
    if not solver_converged:
        solver_log(
            "    [SNES] accepted nonconverged continuation iterate with residual {:.2e} <= {:.2e}".format(
                residual_norm,
                accepted_function_norm_limit,
            )
        )
    return residual_norm


def build_state_snes_recovery_attempts():
    base_attempt = {
        "label": "primary solve",
        "method": str(get_state_option("STATE_SOLVE_METHOD", "FULL_STATE_SNES_METHOD", "newtonls")),
        "line_search": str(get_state_option("STATE_LINE_SEARCH", "FULL_STATE_SNES_LINE_SEARCH", "bt")),
        "rtol": float(get_state_option("STATE_RTOL", "FULL_STATE_SNES_RTOL", globals().get("FORWARD_SNES_RTOL", 1.0e-6))),
        "atol": float(get_state_option("STATE_ATOL", "FULL_STATE_SNES_ATOL", globals().get("FORWARD_SNES_ATOL", 1.0e-8))),
        "max_iters": int(get_state_option("STATE_MAX_ITERS", "FULL_STATE_SNES_MAX_ITERS", globals().get("SNES_MAX_ITERS", 80))),
        "restart_with_stokes": False,
    }
    configured_attempts = get_state_option("STATE_RECOVERY_ATTEMPTS", "FULL_STATE_SNES_RECOVERY_ATTEMPTS", None)
    if configured_attempts is None:
        fallback_attempt = dict(base_attempt)
        fallback_attempt.update(
            {
                "label": "fallback retry",
                "method": str(get_state_option("STATE_FALLBACK_METHOD", "FULL_STATE_SNES_FALLBACK_METHOD", base_attempt["method"])),
                "line_search": str(
                    get_state_option("STATE_FALLBACK_LINE_SEARCH", "FULL_STATE_SNES_FALLBACK_LINE_SEARCH", base_attempt["line_search"])
                ),
                "max_iters": int(
                    get_state_option("STATE_FALLBACK_MAX_ITERS", "FULL_STATE_SNES_FALLBACK_MAX_ITERS", base_attempt["max_iters"])
                ),
                "restart_with_stokes": bool(get_state_option("STATE_RESTART_WITH_STOKES", "FULL_STATE_RESTART_WITH_STOKES", True)),
            }
        )
        return [base_attempt, fallback_attempt]

    attempts = [base_attempt]
    for attempt_idx, attempt_spec in enumerate(configured_attempts, start=1):
        if not isinstance(attempt_spec, dict):
            raise TypeError("STATE_RECOVERY_ATTEMPTS entries must be dictionaries.")
        attempt = dict(base_attempt)
        attempt.update(attempt_spec)
        attempt["label"] = str(attempt.get("label", "recovery attempt {}".format(attempt_idx)))
        attempt["method"] = str(attempt.get("method", base_attempt["method"]))
        attempt["line_search"] = str(attempt.get("line_search", base_attempt["line_search"]))
        attempt["rtol"] = float(attempt.get("rtol", base_attempt["rtol"]))
        attempt["atol"] = float(attempt.get("atol", base_attempt["atol"]))
        attempt["max_iters"] = int(attempt.get("max_iters", base_attempt["max_iters"]))
        attempt["restart_with_stokes"] = bool(attempt.get("restart_with_stokes", False))
        attempts.append(attempt)
    return attempts


def describe_state_snes_attempt(attempt):
    description = [
        "method={}".format(attempt["method"]),
        "max_it={}".format(attempt["max_iters"]),
        "rtol={:.1e}".format(attempt["rtol"]),
        "atol={:.1e}".format(attempt["atol"]),
    ]
    if attempt["method"] == "newtonls":
        description.append("line_search={}".format(attempt["line_search"]))
    return ", ".join(description)


def build_state_turbulence_coupling_schedule():
    configured_schedule = get_state_option("STATE_TURBULENCE_COUPLING_SCHEDULE", "FULL_STATE_TURBULENCE_COUPLING_SCHEDULE", None)
    if configured_schedule is None:
        return [{"weight": 1.0, "convection_weight": 1.0}]

    if isinstance(configured_schedule, np.ndarray):
        configured_schedule = configured_schedule.tolist()
    elif not isinstance(configured_schedule, (list, tuple)):
        configured_schedule = [configured_schedule]

    def unit_interval(value):
        return min(max(float(value), 0.0), 1.0)

    schedule = []
    for step_idx, raw_step in enumerate(configured_schedule, start=1):
        if isinstance(raw_step, dict):
            if "weight" in raw_step:
                raw_weight = raw_step["weight"]
            elif "gamma" in raw_step:
                raw_weight = raw_step["gamma"]
            else:
                raise ValueError(
                    "STATE_TURBULENCE_COUPLING_SCHEDULE step {} must define 'weight' or 'gamma'.".format(
                        step_idx
                    )
                )
            step = {"weight": unit_interval(raw_weight), "convection_weight": 1.0}
            if "convection_weight" in raw_step:
                step["convection_weight"] = unit_interval(raw_step["convection_weight"])
            elif "convective_weight" in raw_step:
                step["convection_weight"] = unit_interval(raw_step["convective_weight"])
            elif "inertia_weight" in raw_step:
                step["convection_weight"] = unit_interval(raw_step["inertia_weight"])
            for key in ("method", "line_search"):
                if key in raw_step:
                    step[key] = str(raw_step[key])
            for key in ("rtol", "atol"):
                if key in raw_step:
                    step[key] = float(raw_step[key])
            if "max_iters" in raw_step:
                step["max_iters"] = int(raw_step["max_iters"])
            if "accept_norm" in raw_step:
                step["accept_norm"] = float(raw_step["accept_norm"])
            if "accept_growth" in raw_step:
                step["accept_growth"] = max(1.0, float(raw_step["accept_growth"]))
            elif "accept_residual_growth" in raw_step:
                step["accept_growth"] = max(1.0, float(raw_step["accept_residual_growth"]))
            if "accept_nonconverged" in raw_step:
                step["accept_nonconverged"] = bool(raw_step["accept_nonconverged"])
        else:
            step = {"weight": unit_interval(raw_step), "convection_weight": 1.0}

        if (
            not schedule
            or abs(step["weight"] - schedule[-1]["weight"]) > 1.0e-12
            or abs(step["convection_weight"] - schedule[-1]["convection_weight"]) > 1.0e-12
        ):
            schedule.append(step)

    if not schedule:
        schedule = [{"weight": 1.0, "convection_weight": 1.0}]
    if (
        abs(schedule[-1]["weight"] - 1.0) > 1.0e-12
        or abs(schedule[-1]["convection_weight"] - 1.0) > 1.0e-12
    ):
        schedule.append({"weight": 1.0, "convection_weight": 1.0})
    return schedule


def solve_state_attempt(attempt, coupling_schedule):
    total_coupling_steps = len(coupling_schedule)
    for coupling_idx, coupling_step in enumerate(coupling_schedule, start=1):
        coupling_weight = coupling_step["weight"]
        convection_weight = coupling_step["convection_weight"]
        state_turbulence_coupling_weight.assign(float(coupling_weight))
        state_convection_coupling_weight.assign(float(convection_weight))
        step_attempt = dict(attempt)
        for key in ("method", "line_search", "rtol", "atol", "max_iters"):
            if key in coupling_step:
                step_attempt[key] = coupling_step[key]
        if total_coupling_steps > 1:
            step_details = []
            if "max_iters" in coupling_step:
                step_details.append("max_it={}".format(step_attempt["max_iters"]))
            if "atol" in coupling_step:
                step_details.append("atol={:.1e}".format(step_attempt["atol"]))
            if "accept_norm" in coupling_step:
                step_details.append("accept={:.1e}".format(coupling_step["accept_norm"]))
            if "accept_growth" in coupling_step:
                step_details.append("growth<={:.2f}".format(coupling_step["accept_growth"]))
            if "rtol" in coupling_step:
                step_details.append("rtol={:.1e}".format(step_attempt["rtol"]))
            root_print(
                "    State continuation step {}/{}: convection = {:.2f}, turbulence = {:.2f}{}".format(
                    coupling_idx,
                    total_coupling_steps,
                    convection_weight,
                    coupling_weight,
                    "" if not step_details else " ({})".format(", ".join(step_details)),
                )
            )
        solve_state_once(
            method_override=step_attempt["method"],
            line_search_override=step_attempt["line_search"],
            max_iters_override=step_attempt["max_iters"],
            rtol_override=step_attempt["rtol"],
            atol_override=step_attempt["atol"],
            accept_function_norm_override=coupling_step.get("accept_norm"),
            accept_nonconverged_override=coupling_step.get("accept_nonconverged"),
            accept_residual_growth_override=coupling_step.get("accept_growth"),
        )


def solve_state_with_recovery():
    # Try the configured SNES recovery attempts before giving up on the current
    # MMA iterate. This keeps the outer optimization loop from failing on the
    # first nonlinear solve breakdown.
    recovery_attempts = build_state_snes_recovery_attempts()
    coupling_schedule = build_state_turbulence_coupling_schedule()
    num_retries = max(0, len(recovery_attempts) - 1)
    try:
        for attempt_idx, attempt in enumerate(recovery_attempts):
            if attempt_idx > 0:
                retry_idx = attempt_idx
                if attempt["restart_with_stokes"]:
                    root_print(
                        "  State solve diverged; rebuilding Stokes-Brinkman warm start before retry {}/{} "
                        "({}: {}).".format(
                            retry_idx,
                            num_retries,
                            attempt["label"],
                            describe_state_snes_attempt(attempt),
                        )
                    )
                    initialize_state_guess_with_stokes(reset_g=False)
                else:
                    root_print(
                        "  State solve diverged; retrying from the current iterate with retry {}/{} "
                        "({}: {}).".format(
                            retry_idx,
                            num_retries,
                            attempt["label"],
                            describe_state_snes_attempt(attempt),
                        )
                    )
            try:
                solve_state_attempt(attempt, coupling_schedule)
                state_turbulence_coupling_weight.assign(1.0)
                state_convection_coupling_weight.assign(1.0)
                return
            except RuntimeError:
                if attempt_idx == len(recovery_attempts) - 1:
                    raise
    finally:
        state_turbulence_coupling_weight.assign(1.0)
        state_convection_coupling_weight.assign(1.0)


# ===============================================================
# Adjoint solve
# ===============================================================
def solve_adjoint(adjoint_residual_form=objective_adjoint_form):
    # Linear adjoint of the current monolithic state system.
    solver_log("    [Adjoint] linear system")
    w_adj.vector().zero()
    A_adj = assemble(derivative(adjoint_residual_form, w_adj))
    b_adj = assemble(adjoint_residual_form)
    b_adj *= -1.0
    for bc in bc_state_adj:
        bc.apply(A_adj, b_adj)
    solve(A_adj, w_adj.vector(), b_adj, LINEAR_SOLVER_NAME)


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# Keep SemiFrozen and Full outputs separated by default so the two formulations
# can be compared without clobbering each other's result folders.
results_root_name = globals().get("RESULTS_ROOT_NAME")
if results_root_name is None:
    results_root_name_key = "RESULTS_ROOT_NAME_FULL" if FULL_STATE_INCLUDE_G else "RESULTS_ROOT_NAME_SEMIFROZEN"
    results_root_name = globals().get(results_root_name_key)
if results_root_name is None:
    if FULL_STATE_INCLUDE_G:
        semifrozen_root_name = globals().get("RESULTS_ROOT_NAME_SEMIFROZEN")
        if semifrozen_root_name is None:
            results_root_name = "Results_Full/Results_TurbulentTO_Full"
        else:
            semifrozen_root_basename = os.path.basename(str(semifrozen_root_name).rstrip("/"))
            if semifrozen_root_basename.endswith("_SemiFrozen"):
                full_basename = "{}_Full".format(semifrozen_root_basename[:-11])
            else:
                full_basename = "{}_Full".format(semifrozen_root_basename)
            results_root_name = os.path.join("Results_Full", full_basename)
    else:
        results_root_name = globals().get(
            "RESULTS_ROOT_NAME",
            "Results_SemiFrozen/Results_TurbulentTO_SemiFrozen",
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
initialize_optimization_log(
    log_path,
    pressure_drop_columns=(
        "Dissipation",
        "dP_static_nondesign",
        "dP_static_design",
    ),
)

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


# ===============================================================
# Continuation and MMA optimization loop
# ===============================================================
optimization_start_time = time.perf_counter()
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
        iteration_start_time = time.perf_counter()
        root_print(
            "--- Stage {}/{} | iter {:03d} (global {:03d}) ---".format(
                stage_idx + 1, len(Q_PENAL_SCHEDULE), inner_count, iter_count,
            )
        )

        # --- Filtering and external wall-distance update ---
        solver_log("  [Filter] design density")
        rho_f = pde_filter(rho, rho_f)
        rho_proj_plot.vector()[:] = project(rho_effective, DensitySpace).vector()[:]
        if not FULL_STATE_INCLUDE_G:
            solver_log("  [Wall distance] update")
            update_wall_distance_field()

        rho_out << rho
        rhop_out << rho_proj_plot

        if iter_count == 0:
            solver_log("  [Warm start] Stokes-Brinkman plus SA initialization")
            initialize_state_guess_with_stokes(reset_g=True)

        # --- State solve ---
        root_print("  [State solve]")
        solve_state_with_recovery()

        # --- Adjoint solve ---
        root_print("  [Adjoint solve]")
        solve_adjoint(objective_adjoint_form)

        u_out << w_state.sub(STATE_VEL_IDX)
        p_out << w_state.sub(STATE_P_IDX)
        nu_tilde_out << w_state.sub(STATE_TURB_IDX)

        f0val = assemble(ObjFunctional)
        dissipation_now = assemble(DissipationFunctional)
        pressure_drop_nondesign_now = pressure_drop_between_boundaries(
            w_state.sub(STATE_P_IDX), ds, MARK["inlet"], MARK["outlet"]
        )
        pressure_drop_design_now = pressure_drop_between_internal_facets(
            w_state.sub(STATE_P_IDX),
            dS_design_pressure,
            design_pressure_drop_mark["inlet"],
            design_pressure_drop_mark["outlet"],
        )
        obj_conv = abs((f0val - previous_objective) / max(abs(f0val), 1.0e-12))

        if obj_conv < OBJECTIVE_CONVERGENCE_TOL:
            convergence_history += 1
            if convergence_history >= OBJECTIVE_STREAK_TO_STOP:
                objective_converged = True
        else:
            convergence_history = 0
        previous_objective = f0val

        # --- Sensitivities and constraints ---
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

        # --- MMA update ---
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
            pressure_drop_nondesign_now,
            obj_conv,
            vol_fraction_now,
            vol_residual_now,
            pressure_drop_values=(
                dissipation_now,
                pressure_drop_nondesign_now,
                pressure_drop_design_now,
            ),
        )

        constraint_status_text = ""
        if mass_flow_status:
            constraint_status_text = " " + " ".join(mass_flow_status)

        iteration_elapsed = time.perf_counter() - iteration_start_time
        optimization_elapsed = time.perf_counter() - optimization_start_time
        root_print(
            "q={:.3f} beta={:.2f} move={:.3f} iter={:03d} iter_time={:.1f}s elapsed={:.1f}s J={:.4e} Dissipation={:.4e} dP_static_nondesign={:.4e} Pa dP_static_design={:.4e} Pa conv={:.3e} vol={:.4f} streak={}/{}{}".format(
                q_val, float(BETA_PROJ.values()[0]), move_limit_now,
                inner_count, iteration_elapsed, optimization_elapsed, f0val,
                dissipation_now, pressure_drop_nondesign_now, pressure_drop_design_now,
                obj_conv, vol_fraction_now,
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

optimization_elapsed = time.perf_counter() - optimization_start_time
if iter_count > 0:
    root_print("Average time per MMA iteration: {:.1f}s ({} iterations, total {:.1f}s).".format(
        optimization_elapsed / float(iter_count), iter_count, optimization_elapsed,
    ))
else:
    root_print("Average time per MMA iteration: n/a (0 iterations, total {:.1f}s).".format(
        optimization_elapsed,
    ))
root_print("Optimization finished. Results written to {}".format(results_root))
