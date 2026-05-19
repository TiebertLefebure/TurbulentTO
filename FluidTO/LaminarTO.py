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
from Utilities_SharedTO import (
    append_df0dx_log_entry,
    append_optimization_log_entry,
    as_list,
    boundary_average_functional,
    build_design_pressure_drop_markers,
    build_pressure_pin_expression_from_config,
    compute_filter_base_length_from_config,
    create_design_mesh_from_config,
    ensure_clean_dir,
    ensure_dir,
    initialize_df0dx_log,
    initialize_optimization_log,
    checkpoint_scalar,
    load_optimization_checkpoint,
    load_config_module_from_cli,
    optimization_checkpoint_path,
    pressure_drop_between_boundaries,
    pressure_drop_between_design_facets,
    resume_optimization_requested,
    resume_vtk_series_path,
    reset_vtk_series,
    save_optimization_checkpoint,
)

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
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
    return (
        tanh(BETA_PROJ * Constant(eta_proj))
        + tanh(BETA_PROJ * (rho_design - Constant(eta_proj)))
    ) / (
        tanh(BETA_PROJ * Constant(eta_proj))
        + tanh(BETA_PROJ * (Constant(1.0) - Constant(eta_proj)))
    )


def alpha(brinkman_density):
    return alpha_solid + (alpha_fluid - alpha_solid) * brinkman_density * (1 + q_penal) / (brinkman_density + q_penal)


def build_state_form(state_u, state_p, adj_u, adj_p, rho_eff, custom_dx):
    """UFL state form used in both forward and adjoint derivations."""
    return (
        rho_fluid * inner(dot(state_u, nabla_grad(state_u)), adj_u) * custom_dx
        + mu_fluid * inner(grad(state_u), grad(adj_u)) * custom_dx
        + inner(grad(state_p), adj_u) * custom_dx
        + inner(div(state_u), adj_p) * custom_dx
        + alpha(rho_eff) * inner(state_u, adj_u) * custom_dx
    )


# ------------------------------------------------------------
# Mesh, function spaces, and boundaries
# ------------------------------------------------------------
mesh = create_design_mesh_from_config(globals())

U_h = VectorElement("CG", mesh.ufl_cell(), 2)
P_h = FiniteElement("CG", mesh.ufl_cell(), 1)
A_h = FiniteElement("DG", mesh.ufl_cell(), 0)

FlowSpace = FunctionSpace(mesh, U_h * P_h)
FlowSpaceAdj = FunctionSpace(mesh, U_h * P_h)
DensitySpace = FunctionSpace(mesh, A_h)

w_fwd = Function(FlowSpace)
(u, p) = split(w_fwd)
w_adj = Function(FlowSpaceAdj)
(v, q) = split(w_adj)

rho = Function(DensitySpace)
rho_f = Function(DensitySpace)
rho_proj_plot = Function(DensitySpace)
unfiltered_gradient = Function(DensitySpace)
filtered_gradient = Function(DensitySpace)
unfiltered_s_vol = Function(DensitySpace)
filtered_s_vol = Function(DensitySpace)
df0dx_centered_plot = Function(DensitySpace)


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


def _as_scalar_dirichlet_value(value):
    if np.isscalar(value):
        return Constant(float(value))
    return value


def _normalize_velocity_component_bc_specs(raw_specs, marker_lookup):
    """Expand optional component-wise velocity BC specs from the config."""
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
ObjectiveRegion = build_region_function_from_config("build_objective_region", 1.0)
VolumeRegion = build_region_function_from_config("build_volume_region", 1.0)

volume_region_values = VolumeRegion.vector().get_local()
active_design_mask = (
    (volume_region_values > 0.5)
    & ((density_upper_values - density_lower_values) > 1.0e-12)
)
ActiveDV = np.flatnonzero(active_design_mask).astype(np.int64)
PassiveDV = np.flatnonzero(~active_design_mask).astype(np.int64)
if ActiveDV.size == 0:
    raise ValueError("Active design space is empty. Check the design-region cell tags.")
root_print(
    "Active design space: {} active DG0 cells, {} passive/non-design cells.".format(
        int(ActiveDV.size), int(PassiveDV.size),
    )
)

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
ds_design_pressure = Measure(
    "ds",
    domain=mesh,
    subdomain_data=design_pressure_drop_facets,
    metadata=measure_metadata,
)

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
outlet_pressure_value = Constant(float(globals().get("OUTLET_PRESSURE_VALUE", 0.0)))

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

root_print("Outlet BC type: {}".format(outlet_bc_type))
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

bcu_walls = [DirichletBC(FlowSpace.sub(0), u_noslip, boundaries, m) for m in wall_markers]
bcu_inlet = [DirichletBC(FlowSpace.sub(0), prof, boundaries, m) for prof, m in zip(inlet_profiles, inlet_markers)]
bcu_outlet = (
    [DirichletBC(FlowSpace.sub(0), prof, boundaries, m) for prof, m in zip(outlet_profiles, outlet_markers)]
    if use_outlet_velocity_bc else []
)
bcu_pressure_outlet_components = [
    DirichletBC(
        FlowSpace.sub(0).sub(spec["component"]),
        spec["value"],
        boundaries,
        spec["marker"],
    )
    for spec in pressure_outlet_component_bcs
]
bcp_outlet = (
    [DirichletBC(FlowSpace.sub(1), outlet_pressure_value, boundaries, m) for m in outlet_markers]
    if use_outlet_pressure_bc else []
)
bc_NS = bcu_walls + bcu_inlet + bcu_outlet + bcu_pressure_outlet_components + bcp_outlet
if use_pressure_pin:
    bcp_pin = DirichletBC(
        FlowSpace.sub(1), Constant(0.0),
        build_pressure_pin_expression_from_config(globals()), "pointwise",
    )
    bc_NS.append(bcp_pin)

adjoint_velocity_bc_builder = globals().get("build_adjoint_velocity_bcs")
if callable(adjoint_velocity_bc_builder):
    bc_NS_adj = list(adjoint_velocity_bc_builder(
        FlowSpaceAdj, boundaries, wall_markers, inlet_markers, outlet_markers,
        u_noslip, inlet_profiles, outlet_profiles,
    ))
else:
    bcu_walls_adj = [DirichletBC(FlowSpaceAdj.sub(0), u_noslip, boundaries, m) for m in wall_markers]
    bcu_inlet_adj = [DirichletBC(FlowSpaceAdj.sub(0), u_noslip, boundaries, m) for m in inlet_markers]
    bc_NS_adj = bcu_walls_adj + bcu_inlet_adj
    if use_outlet_velocity_bc:
        bc_NS_adj += [DirichletBC(FlowSpaceAdj.sub(0), u_noslip, boundaries, m) for m in outlet_markers]
    bc_NS_adj += [
        DirichletBC(
            FlowSpaceAdj.sub(0).sub(spec["component"]),
            Constant(0.0),
            boundaries,
            spec["marker"],
        )
        for spec in pressure_outlet_component_bcs
    ]
if use_outlet_pressure_bc:
    bc_NS_adj += [DirichletBC(FlowSpaceAdj.sub(1), Constant(0.0), boundaries, m) for m in outlet_markers]
if use_pressure_pin:
    bcp_pin_adj = DirichletBC(
        FlowSpaceAdj.sub(1), Constant(0.0),
        build_pressure_pin_expression_from_config(globals()), "pointwise",
    )
    bc_NS_adj.append(bcp_pin_adj)


# ------------------------------------------------------------
# Design filter (Helmholtz PDE filter, DG0)
# ------------------------------------------------------------
r_filter = (
    compute_filter_base_length_from_config(globals())
    * float(globals().get("FILTER_RADIUS_IN_CELLS", 3.0))
)
r = r_filter / (2.0 * 3.0**0.5)

u_filter = TrialFunction(DensitySpace)
v_filter = TestFunction(DensitySpace)
filter_in = Function(DensitySpace)
filter_work = Function(DensitySpace)
active_design_indicator = Function(DensitySpace)
filter_denominator = Function(DensitySpace)
n = FacetNormal(mesh)
h = CellDiameter(mesh)
h_avg = (h("+") + h("-")) / 2.0


active_indicator_values = np.zeros_like(density_lower_values)
active_indicator_values[ActiveDV] = 1.0
active_design_indicator.vector().set_local(active_indicator_values)
active_design_indicator.vector().apply("insert")
filter_denominator_values = None
filter_denominator_floor = max(
    float(globals().get("FILTER_DENOMINATOR_FLOOR", 1.0e-12)),
    1.0e-300,
)
filter_cell_mass_values = np.maximum(assemble(v_filter * dx).get_local(), 1.0e-300)


def pde_filter_raw(input_field, output_field):
    alpha_dg = 4.0
    helmholtz = (
        r**2 * (alpha_dg / h_avg * dot(jump(v_filter, n), jump(u_filter, n))) * dS
        + u_filter * v_filter * dx
        - filter_in * v_filter * dx
    )
    assign(filter_in, input_field)
    solve(lhs(helmholtz) == rhs(helmholtz), output_field)
    return output_field


def initialize_design_filter_normalization():
    """Precompute H(mask) for active-design-only filtering."""
    global filter_denominator_values
    pde_filter_raw(active_design_indicator, filter_denominator)
    filter_denominator_values = np.maximum(
        filter_denominator.vector().get_local(),
        filter_denominator_floor,
    )
    min_active_denom = float(np.min(filter_denominator_values[ActiveDV]))
    if min_active_denom <= 10.0 * filter_denominator_floor:
        raise RuntimeError(
            "Active-design filter normalization has near-zero denominator. "
            "Check design-region tags and FILTER_DENOMINATOR_FLOOR."
        )
    root_print(
        "Design filter: active-cell mask normalization enabled "
        "(min active denominator {:.3e}).".format(min_active_denom)
    )


def pde_filter_design_density(input_field, output_field):
    """Filter active design variables and restore configured passive cells."""
    if filter_denominator_values is None:
        initialize_design_filter_normalization()
    input_values = input_field.vector().get_local()
    work_values = np.zeros_like(input_values)
    work_values[ActiveDV] = input_values[ActiveDV]
    filter_work.vector().set_local(work_values)
    filter_work.vector().apply("insert")
    pde_filter_raw(filter_work, output_field)
    output_values = output_field.vector().get_local() / filter_denominator_values
    output_values = np.clip(output_values, 0.0, 1.0)
    output_values[PassiveDV] = np.clip(
        input_values[PassiveDV],
        density_lower_values[PassiveDV],
        density_upper_values[PassiveDV],
    )
    output_field.vector().set_local(output_values)
    output_field.vector().apply("insert")
    return output_field


def pde_filter_design_gradient(input_field, output_field):
    """Apply the transpose of the active mask-normalized density filter."""
    if filter_denominator_values is None:
        initialize_design_filter_normalization()
    input_values = input_field.vector().get_local()
    # Forward filtering solves A rho_f = M rho before pointwise mask normalization.
    # The coefficient-space transpose is M A^{-T} D^{-1}; the DG Helmholtz
    # operator is symmetric, so A^{-T}=A^{-1}.
    work_values = np.zeros_like(input_values)
    work_values[ActiveDV] = (
        input_values[ActiveDV]
        / filter_denominator_values[ActiveDV]
        / filter_cell_mass_values[ActiveDV]
    )
    filter_work.vector().set_local(work_values)
    filter_work.vector().apply("insert")
    pde_filter_raw(filter_work, output_field)
    output_values = output_field.vector().get_local() * filter_cell_mass_values
    output_values[PassiveDV] = 0.0
    output_field.vector().set_local(output_values)
    output_field.vector().apply("insert")
    return output_field


# ------------------------------------------------------------
# Optimization forms
# ------------------------------------------------------------
rho_effective = projection(rho_f, ETA_I)

dissipation_density_builder = globals().get("build_dissipation_density")
if callable(dissipation_density_builder):
    dissipation_density = dissipation_density_builder(u, mu_fluid)
else:
    deformation = nabla_grad(u) + nabla_grad(u).T
    dissipation_density = 0.5 * mu_fluid * inner(deformation, deformation)

objective_type = str(globals().get("OBJECTIVE_TYPE", "dissipation")).strip().lower()
if objective_type in ("dissipation", "power_dissipation", "volume_dissipation"):
    ObjFunctional = ObjectiveRegion * (
        dissipation_density + alpha(rho_effective) * inner(u, u)
    ) * dx
    objective_log_column = "J_dissipation_W_per_m"
    objective_console_label = "J_dissipation"
    objective_console_unit = " W/m"
    root_print(
        "Objective type: J_dissipation = viscous dissipation plus Brinkman drag "
        "(2D unit-depth power, W/m)."
    )
elif objective_type in ("average_inlet_pressure", "inlet_pressure", "mean_inlet_pressure"):
    ObjFunctional, physical_inlet_area = boundary_average_functional(
        p,
        ds,
        inlet_markers,
    )
    objective_log_column = "J_pressure_Pa"
    objective_console_label = "J_pressure"
    objective_console_unit = " Pa"
    root_print(
        "Objective type: J_pressure = physical inlet-boundary average pressure "
        "over Gamma_in area {:.6e} (Pa).".format(physical_inlet_area)
    )
else:
    raise ValueError(
        "OBJECTIVE_TYPE must be 'dissipation' or 'average_inlet_pressure', got '{}'.".format(
            objective_type
        )
    )
# The nondesign diagnostic mirrors dP_nondesign: use the full simulated domain.
ViscousDissipationNondesignFunctional = dissipation_density * dx
ViscousDissipationFunctional = ObjectiveRegion * dissipation_density * dx

state_form = build_state_form(u, p, v, q, rho_effective, dx)
lagrangian_form = ObjFunctional + state_form

forward_form = derivative(state_form, w_adj, TestFunction(FlowSpace))
adjoint_form = derivative(lagrangian_form, w_fwd, TestFunction(FlowSpaceAdj))

ddx = derivative(lagrangian_form, rho_f)
vol_constraint = VolumeRegion * rho_effective * dx - VolumeRegion * VOL_FRAC * dx
sensitivities_vol_constraint = derivative(vol_constraint, rho_f)


# ------------------------------------------------------------
# Output setup
# ------------------------------------------------------------
results_root = os.path.join(THIS_DIR, globals().get("RESULTS_ROOT_NAME", "LaminarTO_Results"))
rho_dir = os.path.join(results_root, "rho")
rho_p_dir = os.path.join(results_root, "rho_projected")
u_dir = os.path.join(results_root, "u")
p_dir = os.path.join(results_root, "p")
design_dir = os.path.join(results_root, "design")
df0dx_centered_dir = os.path.join(results_root, "df0dx_centered")
save_df0dx_vector = bool(globals().get("SAVE_DF0DX_VECTOR", False))
log_df0dx_stats = bool(globals().get("LOG_DF0DX_STATS", save_df0dx_vector))
save_df0dx_centered_field = bool(globals().get("SAVE_DF0DX_CENTERED_FIELD", log_df0dx_stats))

checkpoint_path = optimization_checkpoint_path(results_root)
resume_requested = resume_optimization_requested(globals())
resume_checkpoint = load_optimization_checkpoint(checkpoint_path, COMM) if resume_requested else None
resume_from_checkpoint = resume_checkpoint is not None
resume_vtk_iteration = (
    checkpoint_scalar(resume_checkpoint, "iter_count", scalar_type=int)
    if resume_from_checkpoint
    else None
)

if resume_from_checkpoint:
    root_print("Resuming optimization from checkpoint {}".format(checkpoint_path))
    output_dirs = [results_root, rho_dir, rho_p_dir, u_dir, p_dir, design_dir]
    if save_df0dx_centered_field:
        output_dirs.append(df0dx_centered_dir)
    for output_dir in output_dirs:
        ensure_dir(output_dir, COMM)
else:
    if resume_requested:
        raise FileNotFoundError(
            "RESUME_OPTIMIZATION was requested, but no checkpoint was found at {}.".format(
                checkpoint_path
            )
        )
    ensure_clean_dir(results_root)
    ensure_clean_dir(rho_dir)
    ensure_clean_dir(rho_p_dir)
    ensure_clean_dir(u_dir)
    ensure_clean_dir(p_dir)
    ensure_clean_dir(design_dir)
    if save_df0dx_centered_field:
        ensure_clean_dir(df0dx_centered_dir)

rho_out = File(reset_vtk_series(resume_vtk_series_path(os.path.join(rho_dir, "plot_rho.pvd"), resume_vtk_iteration), COMM))
rhop_out = File(reset_vtk_series(resume_vtk_series_path(os.path.join(rho_p_dir, "plot_rho_projected.pvd"), resume_vtk_iteration), COMM))
u_out = File(reset_vtk_series(resume_vtk_series_path(os.path.join(u_dir, "plot_u.pvd"), resume_vtk_iteration), COMM))
p_out = File(reset_vtk_series(resume_vtk_series_path(os.path.join(p_dir, "plot_p.pvd"), resume_vtk_iteration), COMM))
if save_df0dx_centered_field:
    df0dx_centered_out = File(
        reset_vtk_series(
            resume_vtk_series_path(os.path.join(df0dx_centered_dir, "plot_df0dx_centered.pvd"), resume_vtk_iteration),
            COMM,
        )
    )

log_path = os.path.join(results_root, "OptimizationLog.txt")
df0dx_log_path = os.path.join(results_root, "Df0dxLog.txt")
optimization_log_quantity_columns = (
    "ViscousDissipation_nondesign_W_per_m",
    "ViscousDissipation_design_W_per_m",
    "dP_nondesign_Pa",
    "dP_design_Pa",
)
if not resume_from_checkpoint:
    initialize_optimization_log(
        log_path,
        pressure_drop_columns=optimization_log_quantity_columns,
        objective_column=objective_log_column,
    )
    if log_df0dx_stats:
        initialize_df0dx_log(df0dx_log_path)


# ------------------------------------------------------------
# MMA initialization
# ------------------------------------------------------------
initial_density = float(globals().get("INITIAL_DENSITY_VALUE", VOL_FRAC))
assign(rho, interpolate(Constant(initial_density), DensitySpace))
rho.vector().set_local(np.clip(rho.vector().get_local(), density_lower_values, density_upper_values))
rho.vector().apply("insert")
initialize_design_filter_normalization()

iter_count = 0
previous_objective = 0.0
objective_scale_reference = None
initial_objective_reference = None

num_mma = int(ActiveDV.size)
active_density_lower_values = density_lower_values[ActiveDV]
active_density_upper_values = density_upper_values[ActiveDV]
xval = np.zeros((num_mma, 1))
xval[:, 0] = rho.vector().get_local()[ActiveDV]
xold1 = np.zeros((num_mma, 1))
xold2 = np.zeros((num_mma, 1))
low = np.zeros((num_mma, 1))
upp = np.zeros((num_mma, 1))

mmma = 1
a0 = 1.0
a = np.zeros((mmma, 1))
c = 1.0e4 * np.ones((mmma, 1))
d = np.ones((mmma, 1))

xmin = active_density_lower_values.reshape((num_mma, 1)).copy()
xmax = active_density_upper_values.reshape((num_mma, 1)).copy()

df0dx = np.zeros((num_mma, 1))
fval = np.zeros((mmma, 1))
dfdx = np.zeros((mmma, num_mma))

volume = assemble(VolumeRegion * dx)
if volume <= 0.0:
    raise ValueError("The volume-constrained design region has zero measure.")


def _assign_scalar_active_density(active_density_value):
    """Set a uniform active density while preserving configured passive cells."""
    density_values = np.clip(rho.vector().get_local(), density_lower_values, density_upper_values)
    density_values[ActiveDV] = np.clip(
        float(active_density_value),
        active_density_lower_values,
        active_density_upper_values,
    )
    rho.vector().set_local(density_values)
    rho.vector().apply("insert")
    return rho


def _filtered_volume_fraction_for_active_density(active_density_value):
    _assign_scalar_active_density(active_density_value)
    pde_filter_design_density(rho, rho_f)
    return float(assemble(VolumeRegion * rho_effective * dx) / volume)


def initialize_active_density_to_filtered_volume_target():
    """Choose the initial active value so rho_effective starts at VOL_FRAC."""
    if not bool(globals().get("INITIAL_DENSITY_MATCH_FILTERED_VOLUME", False)):
        pde_filter_design_density(rho, rho_f)
        return

    target = float(VOL_FRAC)
    lower_value = float(np.min(active_density_lower_values))
    upper_value = float(np.max(active_density_upper_values))
    lower_fraction = _filtered_volume_fraction_for_active_density(lower_value)
    upper_fraction = _filtered_volume_fraction_for_active_density(upper_value)
    if lower_fraction > upper_fraction:
        lower_value, upper_value = upper_value, lower_value
        lower_fraction, upper_fraction = upper_fraction, lower_fraction

    if target <= lower_fraction:
        chosen_density = lower_value
    elif target >= upper_fraction:
        chosen_density = upper_value
    else:
        lo = lower_value
        hi = upper_value
        for _ in range(36):
            mid = 0.5 * (lo + hi)
            mid_fraction = _filtered_volume_fraction_for_active_density(mid)
            if mid_fraction < target:
                lo = mid
            else:
                hi = mid
        chosen_density = 0.5 * (lo + hi)

    final_fraction = _filtered_volume_fraction_for_active_density(chosen_density)
    root_print(
        "Initial active density {:.6f} gives filtered volume {:.6f} "
        "(target {:.6f}).".format(chosen_density, final_fraction, target)
    )


resume_stage_idx = 0
resume_inner_count = 0
resume_convergence_history = 0
if resume_from_checkpoint:
    rho_values = np.asarray(resume_checkpoint["rho"], dtype=float).reshape(rho.vector().get_local().shape)
    rho.vector().set_local(np.clip(rho_values, density_lower_values, density_upper_values))
    rho.vector().apply("insert")
    xval = np.asarray(resume_checkpoint["xval"], dtype=float).reshape(xval.shape)
    xold1 = np.asarray(resume_checkpoint["xold1"], dtype=float).reshape(xold1.shape)
    xold2 = np.asarray(resume_checkpoint["xold2"], dtype=float).reshape(xold2.shape)
    low = np.asarray(resume_checkpoint["low"], dtype=float).reshape(low.shape)
    upp = np.asarray(resume_checkpoint["upp"], dtype=float).reshape(upp.shape)
    iter_count = checkpoint_scalar(resume_checkpoint, "iter_count", scalar_type=int)
    previous_objective = checkpoint_scalar(resume_checkpoint, "previous_objective")
    objective_scale_candidate = checkpoint_scalar(
        resume_checkpoint,
        "objective_scale_reference",
        default=np.nan,
    )
    initial_objective_candidate = checkpoint_scalar(
        resume_checkpoint,
        "initial_objective_reference",
        default=np.nan,
    )
    objective_scale_reference = objective_scale_candidate if np.isfinite(objective_scale_candidate) else None
    initial_objective_reference = initial_objective_candidate if np.isfinite(initial_objective_candidate) else None
    resume_stage_idx = checkpoint_scalar(resume_checkpoint, "stage_idx", scalar_type=int)
    resume_inner_count = checkpoint_scalar(resume_checkpoint, "inner_count", scalar_type=int)
    resume_convergence_history = checkpoint_scalar(
        resume_checkpoint,
        "convergence_history",
        default=0,
        scalar_type=int,
    )
    root_print(
        "Checkpoint state: next global iteration {}, stage {}, inner iteration {}.".format(
            iter_count,
            resume_stage_idx + 1,
            resume_inner_count,
        )
    )
else:
    initialize_active_density_to_filtered_volume_target()
    xval[:, 0] = rho.vector().get_local()[ActiveDV]

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
if len(MAX_INNER_ITERATIONS_SCHEDULE) != len(Q_PENAL_SCHEDULE):
    raise ValueError("MAX_INNER_ITERATIONS_SCHEDULE must match Q_PENAL_SCHEDULE length.")


# ------------------------------------------------------------
# Stokes warm-start: one linear solve before the first SNES call
# Drops the convective term so Newton has a physically reasonable
# initial velocity/pressure field to start from.
# ------------------------------------------------------------
_w_tr = TrialFunction(FlowSpace)
_w_te = TestFunction(FlowSpace)
_u_tr, _p_tr = split(_w_tr)
_v_te, _q_te = split(_w_te)
_stokes_a = (
    mu_fluid * inner(grad(_u_tr), grad(_v_te)) * dx
    + inner(grad(_p_tr), _v_te) * dx
    + inner(div(_u_tr), _q_te) * dx
    + alpha(rho_effective) * inner(_u_tr, _v_te) * dx
)
_stokes_L = inner(Constant((0.0, 0.0)), _v_te) * dx + Constant(0.0) * _q_te * dx


def initialize_forward_guess_with_stokes():
    solve(
        _stokes_a == _stokes_L,
        w_fwd,
        bc_NS,
        solver_parameters={"linear_solver": SNES_LINEAR_SOLVER},
    )


def solve_forward_once(method_override=None):
    jac_fwd = derivative(forward_form, w_fwd)
    problem_fwd = NonlinearVariationalProblem(forward_form, w_fwd, bc_NS, jac_fwd)
    solver_fwd = NonlinearVariationalSolver(problem_fwd)
    solver_fwd.parameters["nonlinear_solver"] = "snes"
    solver_fwd.parameters["snes_solver"]["linear_solver"] = SNES_LINEAR_SOLVER
    method = method_override or globals().get("FORWARD_SNES_METHOD", "newtonls")
    solver_fwd.parameters["snes_solver"]["method"] = method
    if method == "newtonls":
        solver_fwd.parameters["snes_solver"]["line_search"] = globals().get(
            "FORWARD_SNES_LINE_SEARCH", "bt"
        )
    solver_fwd.parameters["snes_solver"]["relative_tolerance"] = FORWARD_SNES_RTOL
    solver_fwd.parameters["snes_solver"]["absolute_tolerance"] = FORWARD_SNES_ATOL
    solver_fwd.parameters["snes_solver"]["maximum_iterations"] = int(
        globals().get("FORWARD_SNES_MAX_ITERS", globals().get("SNES_MAX_ITERS", SNES_MAX_ITERS))
    )
    solver_fwd.parameters["snes_solver"]["error_on_nonconvergence"] = True
    solver_fwd.solve()


root_print("[Stokes warm-start]")
rho_f = pde_filter_design_density(rho, rho_f)
initialize_forward_guess_with_stokes()


# ------------------------------------------------------------
# Optimization loop
# ------------------------------------------------------------
optimization_start_time = time.perf_counter()
for stage_idx, q_val in enumerate(Q_PENAL_SCHEDULE):
    if stage_idx < resume_stage_idx:
        continue

    beta_val = float(BETA_PROJ_SCHEDULE[stage_idx])
    BETA_PROJ.assign(beta_val)
    move_limit_now = MOVE_LIMIT_SCHEDULE[stage_idx]
    max_iters_now = MAX_INNER_ITERATIONS_SCHEDULE[stage_idx]
    q_penal.assign(q_val)
    inner_count = resume_inner_count if stage_idx == resume_stage_idx else 0
    convergence_history = resume_convergence_history if stage_idx == resume_stage_idx else 0
    objective_converged = False
    root_print("Starting continuation stage {}/{}: q = {:.3f}, beta = {:.2f}, move = {:.4f}".format(
        stage_idx + 1, len(Q_PENAL_SCHEDULE), q_val, beta_val, move_limit_now,
    ))

    while inner_count < max_iters_now and not objective_converged:
        iteration_start_time = time.perf_counter()

        root_print("--- Stage {}/{} | iter {:03d} (global {:03d}) ---".format(
            stage_idx + 1, len(Q_PENAL_SCHEDULE), inner_count, iter_count,
        ))

        # Filter current design
        rho_f = pde_filter_design_density(rho, rho_f)
        rho_proj_plot.vector()[:] = project(rho_effective, DensitySpace).vector()[:]
        rho_out << rho
        rhop_out << rho_proj_plot

        # Forward solve
        root_print("  [Forward solve]")
        try:
            solve_forward_once()
        except RuntimeError:
            root_print("  Forward SNES diverged; rebuilding Stokes warm-start and retrying.")
            initialize_forward_guess_with_stokes()
            fallback_method = globals().get("FORWARD_SNES_FALLBACK_METHOD", "newtontr")
            solve_forward_once(method_override=fallback_method)

        # Adjoint solve
        root_print("  [Adjoint solve]")
        jac_adj = derivative(adjoint_form, w_adj)
        problem_adj = NonlinearVariationalProblem(adjoint_form, w_adj, bc_NS_adj, jac_adj)
        solver_adj = NonlinearVariationalSolver(problem_adj)
        solver_adj.parameters["nonlinear_solver"] = "snes"
        solver_adj.parameters["snes_solver"]["linear_solver"] = SNES_LINEAR_SOLVER
        solver_adj.parameters["snes_solver"]["method"] = "newtonls"
        solver_adj.parameters["snes_solver"]["line_search"] = "bt"
        solver_adj.parameters["snes_solver"]["relative_tolerance"] = ADJOINT_SNES_RTOL
        solver_adj.parameters["snes_solver"]["absolute_tolerance"] = ADJOINT_SNES_ATOL
        solver_adj.parameters["snes_solver"]["maximum_iterations"] = SNES_MAX_ITERS
        solver_adj.parameters["snes_solver"]["error_on_nonconvergence"] = False
        solver_adj.solve()

        u_out << w_fwd.sub(0)
        p_out << w_fwd.sub(1)

        f0val = assemble(ObjFunctional)
        if objective_scale_reference is None:
            initial_objective_reference = float(f0val)
            if abs(initial_objective_reference) <= float(globals().get("OBJECTIVE_SCALE_FLOOR", 1.0e-30)):
                raise ValueError("Initial objective is too close to zero for MMA objective scaling.")
            objective_scale_reference = max(
                abs(float(f0val)),
                float(globals().get("OBJECTIVE_SCALE_FLOOR", 1.0e-30)),
            )
            root_print("MMA objective scale: initial objective {:.6e}.".format(objective_scale_reference))
        # MMA sees scaled values; logs keep the physical objective.
        f0val_mma = float(f0val) / objective_scale_reference
        viscous_dissipation_nondesign_now = assemble(ViscousDissipationNondesignFunctional)
        viscous_dissipation_design_now = assemble(ViscousDissipationFunctional)
        pressure_drop_nondesign_now = pressure_drop_between_boundaries(
            w_fwd.sub(1), ds, MARK["inlet"], MARK["outlet"]
        )
        pressure_drop_design_now = pressure_drop_between_design_facets(
            w_fwd.sub(1),
            dS_design_pressure,
            ds_design_pressure,
            design_pressure_drop_mark["inlet"],
            design_pressure_drop_mark["outlet"],
        )
        obj_conv = abs((f0val - previous_objective) / max(abs(f0val), 1e-12))

        if obj_conv < OBJECTIVE_CONVERGENCE_TOL:
            convergence_history += 1
            if convergence_history >= OBJECTIVE_STREAK_TO_STOP:
                objective_converged = True
        else:
            convergence_history = 0
        previous_objective = f0val

        # Objective gradient
        unfiltered_gradient.vector()[:] = assemble(ddx)[:]
        filtered_gradient = pde_filter_design_gradient(unfiltered_gradient, filtered_gradient)
        np.savetxt(os.path.join(design_dir, "rho_{:03}.txt".format(iter_count)), rho.vector()[:])

        # Constraint and constraint gradient
        fval[0, 0] = assemble(vol_constraint) / volume
        unfiltered_s_vol.vector()[:] = assemble(sensitivities_vol_constraint)[:]
        filtered_s_vol = pde_filter_design_gradient(unfiltered_s_vol, filtered_s_vol)
        vol_fraction_now = assemble(VolumeRegion * rho_effective * dx) / volume
        vol_residual_now = float(fval[0, 0])

        objective_gradient_active_unscaled = filtered_gradient.vector().get_local()[ActiveDV].copy()
        df0dx[:, 0] = objective_gradient_active_unscaled / objective_scale_reference
        dfdx[0, :] = filtered_s_vol.vector().get_local()[ActiveDV] / volume
        if save_df0dx_centered_field or save_df0dx_vector:
            df0dx_centered = df0dx[:, 0] - np.mean(df0dx[:, 0])
            if save_df0dx_centered_field:
                df0dx_centered_values = np.zeros_like(filtered_gradient.vector().get_local())
                df0dx_centered_values[ActiveDV] = df0dx_centered
                df0dx_centered_plot.vector().set_local(df0dx_centered_values)
                df0dx_centered_plot.vector().apply("insert")
                df0dx_centered_plot.rename("df0dx_centered", "df0dx_centered")
                df0dx_centered_out << df0dx_centered_plot
            if save_df0dx_vector:
                np.savetxt(os.path.join(design_dir, "df0dx_{:03}.txt".format(iter_count)), df0dx[:, 0])
                np.savetxt(
                    os.path.join(design_dir, "df0dx_unscaled_{:03}.txt".format(iter_count)),
                    objective_gradient_active_unscaled,
                )
                np.savetxt(
                    os.path.join(design_dir, "df0dx_centered_{:03}.txt".format(iter_count)),
                    df0dx_centered,
                )
        if log_df0dx_stats:
            append_df0dx_log_entry(
                df0dx_log_path,
                stage_idx + 1,
                q_val,
                beta_val,
                inner_count,
                iter_count,
                df0dx[:, 0],
            )

        # MMA update
        root_print("  [MMA update]")
        (xmma, _ymma, _zmma, _lam, _xsi, _eta, _mu_mma, _zet, _s, low, upp) = mmasub(
            mmma, num_mma, iter_count, xval, xmin, xmax, xold1, xold2,
            f0val_mma, df0dx, fval, dfdx, low, upp, a0, a, c, d, move_limit_now,
        )

        xold2 = xold1.copy()
        xold1 = xval.copy()
        xval = xmma.copy()
        active_rho_values = np.clip(
            xmma[:, 0].copy(),
            active_density_lower_values,
            active_density_upper_values,
        )
        xval[:, 0] = active_rho_values
        rho_values = np.clip(rho.vector().get_local(), density_lower_values, density_upper_values)
        rho_values[ActiveDV] = active_rho_values
        rho.vector().set_local(rho_values)
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
                viscous_dissipation_nondesign_now,
                viscous_dissipation_design_now,
                pressure_drop_nondesign_now,
                pressure_drop_design_now,
            ),
            pressure_drop_columns=optimization_log_quantity_columns,
            objective_column=objective_log_column,
        )

        checkpoint_next_iter = iter_count + 1
        checkpoint_next_inner = inner_count + 1
        checkpoint_next_stage = stage_idx
        checkpoint_next_convergence_history = convergence_history
        if objective_converged or checkpoint_next_inner >= max_iters_now:
            checkpoint_next_stage = stage_idx + 1
            checkpoint_next_inner = 0
            checkpoint_next_convergence_history = 0
        save_optimization_checkpoint(
            checkpoint_path,
            COMM,
            iter_count=np.array(checkpoint_next_iter, dtype=np.int64),
            stage_idx=np.array(checkpoint_next_stage, dtype=np.int64),
            inner_count=np.array(checkpoint_next_inner, dtype=np.int64),
            convergence_history=np.array(checkpoint_next_convergence_history, dtype=np.int64),
            previous_objective=np.array(float(previous_objective)),
            objective_scale_reference=np.array(float(objective_scale_reference)),
            initial_objective_reference=np.array(float(initial_objective_reference)),
            rho=rho.vector().get_local(),
            xval=xval,
            xold1=xold1,
            xold2=xold2,
            low=low,
            upp=upp,
        )

        iteration_elapsed = time.perf_counter() - iteration_start_time
        optimization_elapsed = time.perf_counter() - optimization_start_time
        root_print("q={:.3f} beta={:.2f} move={:.3f} iter={:03d} iter_time={:.1f}s elapsed={:.1f}s {}={:.4e}{} ViscousDissipation_nondesign={:.4e} W/m ViscousDissipation_design={:.4e} W/m dP_nondesign={:.4e} Pa dP_design={:.4e} Pa conv={:.3e} vol={:.4f} streak={}/{}".format(
            q_val, float(BETA_PROJ.values()[0]), move_limit_now,
            inner_count, iteration_elapsed, optimization_elapsed,
            objective_console_label, f0val, objective_console_unit,
            viscous_dissipation_nondesign_now, viscous_dissipation_design_now,
            pressure_drop_nondesign_now, pressure_drop_design_now,
            obj_conv, vol_fraction_now,
            convergence_history, OBJECTIVE_STREAK_TO_STOP,
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
