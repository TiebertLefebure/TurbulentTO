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
from TurbulenceModel_SpalartAllmaras_TO_Frozen import (
    SpalartAllmarasSteadyState,
    sa_turbulent_viscosity,
)
from Utilities_SharedTO import (
    append_optimization_log_entry,
    as_list,
    boundary_average_functional,
    build_design_pressure_drop_markers,
    build_sa_inlet_nu_tilde_targets,
    build_pressure_pin_expression_from_config,
    compute_filter_base_length_from_config,
    config_truthy,
    create_design_mesh_from_config,
    checkpoint_scalar,
    ensure_clean_dir,
    ensure_dir,
    initialize_optimization_log,
    load_optimization_checkpoint,
    load_config_module_from_cli,
    optimization_checkpoint_path,
    pressure_drop_between_boundaries,
    pressure_drop_between_design_facets,
    resume_optimization_requested,
    resume_vtk_series_path,
    ResilientVTKFile,
    save_optimization_checkpoint,
)
from Utilities_DilgenPostprocess import (
    build_cell_area_normalized_field,
    build_objective_normalized_field,
    build_velocity_magnitude_field,
    copy_scalar_field,
    write_combined_dilgen_table2_if_available,
    write_dilgen_paper_data,
)
from Utilities_TurbulentTO_Frozen import (
    build_penalized_wall_distance_solver,
    build_penalized_poisson_wall_distance_solver,
    calculate_poisson_wall_distance_field,
    nu_tilde_from_viscosity_ratio,
    positive_part,
)

# ====================================================================================
# Turbulent topology optimization with a frozen-turbulence adjoint.
# The flow is advanced with IPCS pressure correction or monolithic SNES, the SA
# model is updated in an outer Picard loop, and the adjoint is solved only for
# the flow variables (u, p).
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


DF0DX_LOG_COLUMNS = [
    ("Stage", 5),
    ("Q", 6),
    ("Beta", 5),
    ("InnerIter", 9),
    ("GlobalIter", 10),
    ("Count", 7),
    ("FiniteCount", 11),
    ("Min", 18),
    ("Max", 18),
    ("Range", 18),
    ("Mean", 18),
    ("Std", 18),
    ("Rms", 18),
    ("Linf", 18),
    ("AbsMean", 18),
    ("RelStdAbsMean", 18),
    ("PosFrac", 18),
    ("NegFrac", 18),
    ("Timestamp", 24),
]

SA_CLIPPING_LOG_COLUMNS = [
    ("Stage", 5),
    ("Q", 6),
    ("Beta", 5),
    ("InnerIter", 9),
    ("GlobalIter", 10),
    ("Picard", 6),
    ("RawMin", 18),
    ("RawMax", 18),
    ("PreclipMin", 18),
    ("PreclipMax", 18),
    ("ClippedMin", 18),
    ("ClippedMax", 18),
    ("Floor", 18),
    ("Ceiling", 18),
    ("FloorClipCount", 14),
    ("CeilingClipCount", 16),
    ("FloorBoundCount", 15),
    ("CeilingBoundCount", 17),
    ("DofCount", 8),
    ("FloorClipFrac", 18),
    ("CeilingClipFrac", 18),
    ("FloorBoundFrac", 18),
    ("CeilingBoundFrac", 18),
    ("Timestamp", 24),
]


def fixed_width_log_row(values, columns):
    widths = [width for _name, width in columns]
    cells = [
        str(value).ljust(width)
        for value, width in zip(values, widths)
    ]
    return " ".join(cells).rstrip() + "\n"


def initialize_df0dx_log(log_path):
    if IS_ROOT:
        with open(log_path, "w") as txtout:
            txtout.write(fixed_width_log_row([name for name, _width in DF0DX_LOG_COLUMNS], DF0DX_LOG_COLUMNS))
    MPI.barrier(COMM)


def initialize_sa_clipping_log(log_path):
    if IS_ROOT:
        with open(log_path, "w") as txtout:
            txtout.write(
                fixed_width_log_row([name for name, _width in SA_CLIPPING_LOG_COLUMNS], SA_CLIPPING_LOG_COLUMNS)
            )
    MPI.barrier(COMM)


def append_sa_clipping_log_entry(log_path, stats):
    if not IS_ROOT:
        return
    row_values = [
        "{:d}".format(int(stats["stage"])),
        "{:.3f}".format(float(stats["q"])),
        "{:.2f}".format(float(stats["beta"])),
        "{:d}".format(int(stats["inner_iter"])),
        "{:d}".format(int(stats["global_iter"])),
        "{:d}".format(int(stats["picard"])),
        "{:.10e}".format(float(stats["raw_min"])),
        "{:.10e}".format(float(stats["raw_max"])),
        "{:.10e}".format(float(stats["preclip_min"])),
        "{:.10e}".format(float(stats["preclip_max"])),
        "{:.10e}".format(float(stats["clipped_min"])),
        "{:.10e}".format(float(stats["clipped_max"])),
        "{:.10e}".format(float(stats["floor"])),
        "{:.10e}".format(float(stats["ceiling"])),
        "{:d}".format(int(stats["floor_clip_count"])),
        "{:d}".format(int(stats["ceiling_clip_count"])),
        "{:d}".format(int(stats["floor_bound_count"])),
        "{:d}".format(int(stats["ceiling_bound_count"])),
        "{:d}".format(int(stats["dof_count"])),
        "{:.10e}".format(float(stats["floor_clip_frac"])),
        "{:.10e}".format(float(stats["ceiling_clip_frac"])),
        "{:.10e}".format(float(stats["floor_bound_frac"])),
        "{:.10e}".format(float(stats["ceiling_bound_frac"])),
        time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()),
    ]
    with open(log_path, "a") as txtout:
        txtout.write(fixed_width_log_row(row_values, SA_CLIPPING_LOG_COLUMNS))


def append_df0dx_log_entry(log_path, stage_idx, q_value, beta_value, inner_iter, global_iter, values):
    values = np.asarray(values, dtype=float).ravel()
    finite_values = values[np.isfinite(values)]
    count = int(values.size)
    finite_count = int(finite_values.size)
    if finite_count:
        min_value = float(np.min(finite_values))
        max_value = float(np.max(finite_values))
        mean_value = float(np.mean(finite_values))
        std_value = float(np.std(finite_values))
        rms_value = float(np.sqrt(np.mean(finite_values**2)))
        linf_value = float(np.max(np.abs(finite_values)))
        abs_mean_value = float(np.mean(np.abs(finite_values)))
        rel_std_abs_mean = std_value / max(abs_mean_value, 1.0e-300)
        pos_frac = float(np.count_nonzero(finite_values > 0.0)) / finite_count
        neg_frac = float(np.count_nonzero(finite_values < 0.0)) / finite_count
    else:
        min_value = max_value = mean_value = std_value = np.nan
        rms_value = linf_value = abs_mean_value = rel_std_abs_mean = np.nan
        pos_frac = neg_frac = np.nan

    if IS_ROOT:
        row_values = [
            "{:d}".format(int(stage_idx)),
            "{:.3f}".format(float(q_value)),
            "{:.2f}".format(float(beta_value)),
            "{:d}".format(int(inner_iter)),
            "{:d}".format(int(global_iter)),
            "{:d}".format(count),
            "{:d}".format(finite_count),
            "{:.10e}".format(min_value),
            "{:.10e}".format(max_value),
            "{:.10e}".format(max_value - min_value),
            "{:.10e}".format(mean_value),
            "{:.10e}".format(std_value),
            "{:.10e}".format(rms_value),
            "{:.10e}".format(linf_value),
            "{:.10e}".format(abs_mean_value),
            "{:.10e}".format(rel_std_abs_mean),
            "{:.10e}".format(pos_frac),
            "{:.10e}".format(neg_frac),
            time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()),
        ]
        with open(log_path, "a") as txtout:
            txtout.write(fixed_width_log_row(row_values, DF0DX_LOG_COLUMNS))


CONFIG_MODULE_NAME, CONFIG = load_config_module_from_cli()
CONFIG_OPTION_NAMES = {
    _name for _name in vars(CONFIG)
    if not _name.startswith("_")
}
for _name, _value in vars(CONFIG).items():
    if not _name.startswith("_"):
        globals()[_name] = _value


def normalize_forward_flow_solver_name(value, setting_name, allow_ipcs_snes_polish=False):
    token = str(value).strip().lower().replace("-", "_").replace("+", "_").replace(" ", "_")
    if token in {"newton", "newtonls", "snes"}:
        return "snes"
    if token == "ipcs":
        return "ipcs"
    if allow_ipcs_snes_polish and token in {
        "ipcs_snes",
        "ipcs_snes_polish",
        "ipcs_then_snes",
        "ipcs_with_snes_polish",
    }:
        return "ipcs_snes_polish"
    allowed = "'ipcs' or 'snes'"
    if allow_ipcs_snes_polish:
        allowed = "'ipcs', 'ipcs_snes_polish', or 'snes'"
    raise ValueError("{} must be {}.".format(setting_name, allowed))


if "FORWARD_IPCS_FINAL_SNES_POLISH" in CONFIG_OPTION_NAMES:
    raise ValueError(
        "FORWARD_IPCS_FINAL_SNES_POLISH has been replaced by "
        "FORWARD_FLOW_SOLVER = 'ipcs_snes_polish'."
    )
if "FORWARD_IPCS_SNES_POLISH_ALL" in CONFIG_OPTION_NAMES:
    raise ValueError(
        "FORWARD_IPCS_SNES_POLISH_ALL has been removed. Use "
        "FORWARD_FLOW_SOLVER = 'ipcs_snes_polish' for the final flow solve, "
        "or FORWARD_PICARD_FLOW_SOLVER = 'snes' for Picard solves."
    )


SHOW_SOLVE_LABELS = bool(globals().get("SHOW_SOLVE_LABELS", True))
SHOW_DOLFIN_SOLVER_LOGS = bool(globals().get("SHOW_DOLFIN_SOLVER_LOGS", False))
FORWARD_IPCS_LOG_EVERY = max(1, int(globals().get("FORWARD_IPCS_LOG_EVERY", 25)))
LINEAR_SOLVER_NAME = str(globals().get("LINEAR_SOLVER", "mumps"))
FORWARD_FLOW_SOLVER = normalize_forward_flow_solver_name(
    globals().get("FORWARD_FLOW_SOLVER", "snes"),
    "FORWARD_FLOW_SOLVER",
    allow_ipcs_snes_polish=True,
)
_default_picard_flow_solver = (
    "ipcs" if FORWARD_FLOW_SOLVER in {"ipcs", "ipcs_snes_polish"} else FORWARD_FLOW_SOLVER
)
FORWARD_PICARD_FLOW_SOLVER = normalize_forward_flow_solver_name(
    globals().get("FORWARD_PICARD_FLOW_SOLVER", _default_picard_flow_solver),
    "FORWARD_PICARD_FLOW_SOLVER",
    allow_ipcs_snes_polish=False,
)
_fd_flow_solver_raw = str(globals().get("FINITE_DIFFERENCE_CHECK_FLOW_SOLVER", "same")).strip().lower()
if _fd_flow_solver_raw in {"same", "configured", "forward"}:
    FINITE_DIFFERENCE_CHECK_FLOW_SOLVER = FORWARD_FLOW_SOLVER
else:
    FINITE_DIFFERENCE_CHECK_FLOW_SOLVER = normalize_forward_flow_solver_name(
        _fd_flow_solver_raw,
        "FINITE_DIFFERENCE_CHECK_FLOW_SOLVER",
        allow_ipcs_snes_polish=True,
    )
_fd_picard_flow_solver_raw = str(
    globals().get("FINITE_DIFFERENCE_CHECK_PICARD_FLOW_SOLVER", "same")
).strip().lower()
if _fd_picard_flow_solver_raw in {"same", "configured", "picard"}:
    FINITE_DIFFERENCE_CHECK_PICARD_FLOW_SOLVER = FORWARD_PICARD_FLOW_SOLVER
else:
    FINITE_DIFFERENCE_CHECK_PICARD_FLOW_SOLVER = normalize_forward_flow_solver_name(
        _fd_picard_flow_solver_raw,
        "FINITE_DIFFERENCE_CHECK_PICARD_FLOW_SOLVER",
        allow_ipcs_snes_polish=False,
    )

if not SHOW_DOLFIN_SOLVER_LOGS:
    try:
        set_log_level(LogLevel.WARNING)
    except NameError:
        set_log_level(30)


def final_flow_uses_ipcs():
    return FORWARD_FLOW_SOLVER in {"ipcs", "ipcs_snes_polish"}


def format_forward_flow_solver_name(solver_name):
    if solver_name == "ipcs_snes_polish":
        return "IPCS + SNES polish"
    return solver_name.upper()


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
forward_convection_coupling_weight = Constant(float(globals().get("FORWARD_SNES_CONVECTION_WEIGHT", 1.0)))


def projection(rho_design, eta_proj):
    """Projection used to sharpen the filtered design."""
    if not bool(globals().get("USE_HEAVISIDE_PROJECTION", True)):
        return rho_design
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

def strain_tensor(velocity):
    """Symmetric velocity-gradient operator used by residual and objective."""
    return nabla_grad(velocity) + nabla_grad(velocity).T


def viscous_stress_form(mu_value, trial_velocity, test_velocity):
    return Constant(0.5) * mu_value * inner(
        strain_tensor(trial_velocity),
        strain_tensor(test_velocity),
    )


def sa_positive_viscosity(state_nu_tilde):
    """Return the SA turbulent viscosity with negative values clipped away."""
    nu_lam = mu_fluid / rho_fluid
    return sa_turbulent_viscosity(
        state_nu_tilde,
        nu_lam,
        smooth_abs_eps=float(globals().get("SA_SMOOTH_ABS_EPS", 1.0e-12)),
        nu_tilde_floor=float(globals().get("SA_NU_TILDE_FLOOR", 1.0e-12)),
    )


def effective_dynamic_viscosity(frozen_nu_tilde):
    """Return the effective viscosity used in the flow solve."""
    return mu_fluid + rho_fluid * sa_positive_viscosity(frozen_nu_tilde)


def build_state_form(state_u, state_p, adj_u, adj_p, rho_eff, custom_dx, frozen_nu_tilde):
    """Build the flow weak form using the frozen turbulent viscosity."""
    mu_effective = effective_dynamic_viscosity(frozen_nu_tilde)
    return (
        forward_convection_coupling_weight * rho_fluid * inner(dot(state_u, nabla_grad(state_u)), adj_u) * custom_dx
        + viscous_stress_form(mu_effective, state_u, adj_u) * custom_dx
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
df0dx_plot = Function(DensitySpace)
df0dx_centered_plot = Function(DensitySpace)
j_d_per_cell_plot = Function(DensitySpace)
viscous_dissipation_per_cell_plot = Function(DensitySpace)
unfiltered_s_vol = Function(DensitySpace)
filtered_s_vol = Function(DensitySpace)
unfiltered_constraint_gradient = Function(DensitySpace)
filtered_constraint_gradient = Function(DensitySpace)
cell_integral_test = TestFunction(DensitySpace)

rho.rename("rho", "rho")
rho_f.rename("rho_filtered", "rho_filtered")
rho_proj_plot.rename("rho_projected", "rho_projected")
j_d_per_cell_plot.rename("J_D", "J_D")
viscous_dissipation_per_cell_plot.rename("viscous_dissipation", "viscous_dissipation")


def assemble_cell_integral_field(integrand, target, name):
    """Store one DG0 value per cell equal to the integral over that cell."""
    target.vector().set_local(assemble(integrand * cell_integral_test * dx).get_local())
    target.vector().apply("insert")
    target.rename(name, name)
    return target


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
            "Density bounds: {} fixed density cells ({} fluid, {} solid).".format(
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
    """Convert scalar component BC values to Constant when appropriate."""
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
AreaOfInterest = VolumeRegion

area_of_interest_values = AreaOfInterest.vector().get_local()
active_design_mask = (
    (area_of_interest_values > 0.5)
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
root_print("Frozen final flow solver: {}".format(format_forward_flow_solver_name(FORWARD_FLOW_SOLVER)))
root_print("Frozen Picard flow solver: {}".format(FORWARD_PICARD_FLOW_SOLVER.upper()))

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
# Velocity outlet mode: prescribe u on the outlet marker.
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
# Pressure outlet mode: prescribe p on the outlet marker instead of an outlet velocity profile.
bcp_outlet = (
    [DirichletBC(FlowSpace.sub(1), outlet_pressure_value, boundaries, m) for m in outlet_markers]
    if use_outlet_pressure_bc else []
)
bc_NS = bcu_walls + bcu_inlet + bcu_outlet + bcu_pressure_outlet_components + bcp_outlet
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

nu_tilde_outlet_bc_values = []
custom_turbulence_outlet_builder = globals().get("build_turbulence_outlet_profile_sets")
if callable(custom_turbulence_outlet_builder):
    nu_tilde_outlet_bc_values = as_list(custom_turbulence_outlet_builder())
    if len(nu_tilde_outlet_bc_values) != len(outlet_markers):
        raise ValueError(
            "Expected {} custom SA outlet profiles, got {}.".format(
                len(outlet_markers), len(nu_tilde_outlet_bc_values),
            )
        )
    root_print("SA outlet nu_tilde: custom profiles  (nu_lam = {:.3e})".format(_nu_lam))
elif "SA_NU_TILDE_OUTLETS" in globals() or "SA_NU_TILDE_OUTLET" in globals():
    outlet_raw = globals().get("SA_NU_TILDE_OUTLETS", globals().get("SA_NU_TILDE_OUTLET"))
    outlet_values = [float(value) for value in as_list(outlet_raw)]
    if len(outlet_values) == 1 and len(outlet_markers) > 1:
        outlet_values = outlet_values * len(outlet_markers)
    if len(outlet_values) != len(outlet_markers):
        raise ValueError(
            "Expected {} SA outlet nu_tilde values, got {}.".format(
                len(outlet_markers), len(outlet_values),
            )
        )
    nu_tilde_outlet_bc_values = [Constant(value) for value in outlet_values]
    root_print(
        "SA outlet nu_tilde: {}  (nu_lam = {:.3e})".format(
            ", ".join("{:.4e}".format(value) for value in outlet_values),
            _nu_lam,
        )
    )
bcn_turbulence = (
    [DirichletBC(TurbulenceSpace, bc_val, boundaries, m) for bc_val, m in zip(nu_tilde_inlet_bc_values, inlet_markers)]
    + [DirichletBC(TurbulenceSpace, bc_val, boundaries, m) for bc_val, m in zip(nu_tilde_outlet_bc_values, outlet_markers)]
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
filter_work = Function(DensitySpace)
active_design_indicator = Function(DensitySpace)
filter_denominator = Function(DensitySpace)
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
    """Precompute H(mask) for the active-design normalized Helmholtz filter."""
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
    """Filter only active design variables and normalize by the filtered mask."""
    if filter_denominator_values is None:
        initialize_design_filter_normalization()
    input_values = input_field.vector().get_local()
    work_values = np.zeros_like(input_values)
    work_values[ActiveDV] = input_values[ActiveDV]
    filter_work.vector().set_local(work_values)
    filter_work.vector().apply("insert")
    pde_filter_raw(filter_work, output_field)
    # rho_f = H(M rho) / H(M); passive cells are restored after filtering.
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


def enforce_density_bounds_inplace(density_field):
    values = density_field.vector().get_local()
    values = np.clip(values, density_lower_values, density_upper_values)
    density_field.vector().set_local(values)
    density_field.vector().apply("insert")
    return density_field


def enforce_scalar_bounds_inplace(scalar_field, floor_value=None, ceiling_value=None):
    values = scalar_field.vector().get_local()
    if floor_value is not None:
        values = np.maximum(values, float(floor_value))
    if ceiling_value is not None:
        values = np.minimum(values, float(ceiling_value))
    scalar_field.vector().set_local(values)
    scalar_field.vector().apply("insert")
    return scalar_field

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
sa_wall_sigma = float(globals().get("SA_WALL_SIGMA", 0.01))
sa_wall_g0 = float(globals().get("SA_WALL_G0", 20.0))
sa_wall_penalty_alpha_base = float(globals().get("SA_WALL_PENALTY_ALPHA", 1.0e3))
sa_wall_penalty_alpha = Constant(sa_wall_penalty_alpha_base)
sa_wall_penalty_power = float(globals().get("SA_WALL_PENALTY_N", 3.0))
sa_wall_penalty_interpolation = str(
    globals().get("SA_WALL_PENALTY_INTERPOLATION", "power")
).strip().lower()
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
sa_wall_distance_mode = str(globals().get("SA_WALL_DISTANCE_MODE", "reciprocal_penalized")).strip().lower()
if sa_wall_density_source == "passive":
    wall_penalty_fluid_indicator = density_upper_bound
else:
    wall_penalty_fluid_indicator = rho_effective
if sa_wall_distance_mode in {"poisson", "poisson_like", "tucker"}:
    if custom_initial_wall_distance is None:
        wall_distance = calculate_poisson_wall_distance_field(
            TurbulenceSpace, boundaries, wall_markers, dx
        )
    elif isinstance(custom_initial_wall_distance, Function):
        wall_distance = Function(TurbulenceSpace)
        wall_distance.assign(custom_initial_wall_distance)
    else:
        wall_distance = project(custom_initial_wall_distance, TurbulenceSpace)

    def update_wall_distance_field():
        return wall_distance

    root_print("SA wall-distance mode: Tucker Poisson-like static field")
elif sa_wall_distance_mode in {"poisson_penalized", "penalized_poisson", "dilgen_poisson"}:
    if sa_wall_penalty_interpolation in {"brinkman", "dilgen", "chi"}:
        wall_penalty_reaction = (
            sa_wall_penalty_alpha / max(float(ALPHA_SOLID), 1.0e-300)
        ) * alpha(wall_penalty_fluid_indicator)
    elif sa_wall_penalty_interpolation == "power":
        wall_penalty_reaction = sa_wall_penalty_alpha * (
            positive_part(Constant(1.0) - wall_penalty_fluid_indicator) ** sa_wall_penalty_power
        )
    else:
        raise ValueError(
            "SA_WALL_PENALTY_INTERPOLATION must be 'power' or 'brinkman', got {!r}.".format(
                sa_wall_penalty_interpolation
            )
        )

    (
        wall_distance,
        update_wall_distance_field,
    ) = build_penalized_poisson_wall_distance_solver(
        TurbulenceSpace,
        boundaries,
        wall_markers,
        dx,
        wall_penalty_reaction,
        initial_wall_distance=custom_initial_wall_distance,
        floor_value=float(globals().get("SA_WALL_DISTANCE_SOLVE_FLOOR", 0.0)),
    )
    root_print("SA wall-distance mode: Dilgen penalized Poisson-like field")
else:
    if sa_wall_distance_mode not in {"reciprocal_penalized", "reciprocal", "penalized_reciprocal", "yoon"}:
        raise ValueError(
            "SA_WALL_DISTANCE_MODE must be 'reciprocal_penalized', 'poisson', or 'poisson_penalized', got {!r}.".format(
                sa_wall_distance_mode
            )
        )

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
    root_print("SA wall-distance mode: reciprocal_penalized wall-distance field")
# Initial SA field.
sa_nu_tilde_init = float(globals().get("SA_NU_TILDE_INITIAL", _sa_nu_tilde_initial_default))
sa_nu_tilde_floor = float(globals().get("SA_NU_TILDE_FLOOR", 1.0e-12))
sa_nu_tilde_ceiling = globals().get("SA_NU_TILDE_CEILING", None)
if "SA_EDDY_VISCOSITY_RATIO_CEILING" in globals():
    sa_nu_tilde_ceiling = (
        float(globals()["SA_EDDY_VISCOSITY_RATIO_CEILING"])
        * float(MU_FLUID_VALUE)
        / float(RHO_FLUID_VALUE)
    )
if sa_nu_tilde_ceiling is not None:
    sa_nu_tilde_ceiling = float(sa_nu_tilde_ceiling)
    if sa_nu_tilde_ceiling <= sa_nu_tilde_floor:
        raise ValueError("SA_NU_TILDE_CEILING must be larger than SA_NU_TILDE_FLOOR.")
    root_print("SA nu_tilde ceiling: {:.4e}".format(sa_nu_tilde_ceiling))

nu_laminar = Constant(MU_FLUID_VALUE / RHO_FLUID_VALUE)
sa_nu_tilde_penalty_alpha_base = float(globals().get("SA_NU_TILDE_PENALTY_ALPHA", 1.0e3))
sa_nu_tilde_penalty_alpha = Constant(sa_nu_tilde_penalty_alpha_base)
sa_nu_tilde_penalty_interpolation = str(
    globals().get("SA_NU_TILDE_PENALTY_INTERPOLATION", "power")
).strip().lower()
if sa_nu_tilde_penalty_interpolation in {"brinkman", "dilgen", "chi"}:
    sa_nu_tilde_penalty_reaction = (
        sa_nu_tilde_penalty_alpha / max(float(ALPHA_SOLID), 1.0e-300)
    ) * alpha(rho_effective)
elif sa_nu_tilde_penalty_interpolation == "power":
    sa_nu_tilde_penalty_reaction = sa_nu_tilde_penalty_alpha * (
        positive_part(Constant(1.0) - rho_effective) ** float(globals().get("SA_NU_TILDE_PENALTY_N", 3.0))
    )
else:
    raise ValueError(
        "SA_NU_TILDE_PENALTY_INTERPOLATION must be 'power' or 'brinkman', got {!r}.".format(
            sa_nu_tilde_penalty_interpolation
        )
    )
sa_pseudo_time_stabilization = bool(globals().get("SA_PSEUDO_TIME_STABILIZATION", False))
sa_pseudo_dt = float(globals().get("SA_PSEUDO_DT", 1.0))
sa_pseudo_time_steps = max(1, int(globals().get("SA_PSEUDO_TIME_STEPS", 1)))
if sa_pseudo_time_stabilization:
    root_print(
        "SA pseudo-time stabilization: dt = {:.3e}, substeps = {}".format(
            sa_pseudo_dt,
            sa_pseudo_time_steps,
        )
    )

sa_model = SpalartAllmarasSteadyState(
    TurbulenceSpace, bcn_turbulence, sa_nu_tilde_init, nu_laminar,
    dx, wall_distance,
    nu_tilde_penalty_reaction=sa_nu_tilde_penalty_reaction,
    wall_distance_floor=float(globals().get("SA_WALL_DISTANCE_FLOOR", 0.0)),
    smooth_abs_eps=float(globals().get("SA_SMOOTH_ABS_EPS", 1.0e-12)),
    nu_tilde_floor=sa_nu_tilde_floor,
    supg_stabilization=bool(globals().get("SA_SUPG_STABILIZATION", True)),
    supg_tau_scale=float(globals().get("SA_SUPG_TAU_SCALE", 1.0)),
    pseudo_time_stabilization=sa_pseudo_time_stabilization,
    pseudo_dt=sa_pseudo_dt,
)

save_sa_clipping_diagnostics = bool(globals().get("SAVE_SA_CLIPPING_DIAGNOSTICS", False))
sa_clipping_diagnostic_tolerance = max(
    float(globals().get("SA_CLIPPING_DIAGNOSTIC_TOL", 1.0e-14)),
    0.0,
)
nu_tilde_raw_diagnostic = Function(TurbulenceSpace)
nu_tilde_raw_diagnostic.rename("nu_tilde_raw_sa_solve", "nu_tilde_raw_sa_solve")
nu_tilde_preclip_diagnostic = Function(TurbulenceSpace)
nu_tilde_preclip_diagnostic.rename("nu_tilde_relaxed_preclip", "nu_tilde_relaxed_preclip")
nu_tilde_floor_clip_mask = Function(TurbulenceSpace)
nu_tilde_floor_clip_mask.rename("nu_tilde_floor_clip_mask", "nu_tilde_floor_clip_mask")
nu_tilde_ceiling_clip_mask = Function(TurbulenceSpace)
nu_tilde_ceiling_clip_mask.rename("nu_tilde_ceiling_clip_mask", "nu_tilde_ceiling_clip_mask")
nu_tilde_floor_bound_mask = Function(TurbulenceSpace)
nu_tilde_floor_bound_mask.rename("nu_tilde_floor_bound_mask", "nu_tilde_floor_bound_mask")
nu_tilde_ceiling_bound_mask = Function(TurbulenceSpace)
nu_tilde_ceiling_bound_mask.rename("nu_tilde_ceiling_bound_mask", "nu_tilde_ceiling_bound_mask")


def _field_minmax(values):
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return np.nan, np.nan
    return float(np.min(finite_values)), float(np.max(finite_values))


def _set_diagnostic_values(function, values):
    function.vector().set_local(np.asarray(values, dtype=float))
    function.vector().apply("insert")


def update_sa_clipping_diagnostics(stage_idx, q_value, beta_value, inner_iter, global_iter, picard_idx):
    """Store the last SA Picard raw/preclip fields and masks for VTK/log output."""
    if not save_sa_clipping_diagnostics:
        return None

    raw_values = sa_model.nu_tilde1.vector().get_local().copy()
    preclip_values = sa_model.nu_tilde0.vector().get_local().copy()
    clipped_values = nu_tilde_frozen.vector().get_local().copy()
    ceiling_value = np.inf if sa_nu_tilde_ceiling is None else float(sa_nu_tilde_ceiling)

    floor_clip_values = (preclip_values < float(sa_nu_tilde_floor)).astype(float)
    ceiling_clip_values = (preclip_values > ceiling_value).astype(float)
    floor_bound_values = (
        clipped_values <= float(sa_nu_tilde_floor) + sa_clipping_diagnostic_tolerance
    ).astype(float)
    if sa_nu_tilde_ceiling is None:
        ceiling_bound_values = np.zeros_like(clipped_values, dtype=float)
    else:
        ceiling_bound_values = (
            clipped_values >= float(sa_nu_tilde_ceiling) - sa_clipping_diagnostic_tolerance
        ).astype(float)

    _set_diagnostic_values(nu_tilde_raw_diagnostic, raw_values)
    _set_diagnostic_values(nu_tilde_preclip_diagnostic, preclip_values)
    _set_diagnostic_values(nu_tilde_floor_clip_mask, floor_clip_values)
    _set_diagnostic_values(nu_tilde_ceiling_clip_mask, ceiling_clip_values)
    _set_diagnostic_values(nu_tilde_floor_bound_mask, floor_bound_values)
    _set_diagnostic_values(nu_tilde_ceiling_bound_mask, ceiling_bound_values)

    raw_min, raw_max = _field_minmax(raw_values)
    preclip_min, preclip_max = _field_minmax(preclip_values)
    clipped_min, clipped_max = _field_minmax(clipped_values)
    dof_count = int(preclip_values.size)
    denominator = max(dof_count, 1)
    floor_clip_count = int(np.count_nonzero(floor_clip_values))
    ceiling_clip_count = int(np.count_nonzero(ceiling_clip_values))
    floor_bound_count = int(np.count_nonzero(floor_bound_values))
    ceiling_bound_count = int(np.count_nonzero(ceiling_bound_values))

    return {
        "stage": int(stage_idx) + 1,
        "q": float(q_value),
        "beta": float(beta_value),
        "inner_iter": int(inner_iter),
        "global_iter": int(global_iter),
        "picard": int(picard_idx) + 1,
        "raw_min": raw_min,
        "raw_max": raw_max,
        "preclip_min": preclip_min,
        "preclip_max": preclip_max,
        "clipped_min": clipped_min,
        "clipped_max": clipped_max,
        "floor": float(sa_nu_tilde_floor),
        "ceiling": float(sa_nu_tilde_ceiling) if sa_nu_tilde_ceiling is not None else np.nan,
        "floor_clip_count": floor_clip_count,
        "ceiling_clip_count": ceiling_clip_count,
        "floor_bound_count": floor_bound_count,
        "ceiling_bound_count": ceiling_bound_count,
        "dof_count": dof_count,
        "floor_clip_frac": float(floor_clip_count) / denominator,
        "ceiling_clip_frac": float(ceiling_clip_count) / denominator,
        "floor_bound_frac": float(floor_bound_count) / denominator,
        "ceiling_bound_frac": float(ceiling_bound_count) / denominator,
    }


# ---------------------------------------------------------------
# Objective, state, adjoint, and sensitivity forms.
# The turbulent viscosity is treated as a frozen coefficient here.
# ---------------------------------------------------------------
dissipation_density_builder = globals().get("build_dissipation_density")
mu_effective = effective_dynamic_viscosity(nu_tilde_frozen)
if callable(dissipation_density_builder):
    dissipation_density = dissipation_density_builder(u, mu_effective)
else:
    deformation = strain_tensor(u)
    dissipation_density = 0.5 * mu_effective * inner(deformation, deformation)

# Objective:
#   - dissipation: Dilgen-style volume power loss with Brinkman drag.
#   - average_inlet_pressure: Alexandersen-style physical inlet boundary pressure.
J_D_integrand = ObjectiveRegion * (
    dissipation_density
    + alpha(rho_effective) * inner(u, u)
)
viscous_dissipation_integrand = ObjectiveRegion * dissipation_density
objective_type = str(globals().get("OBJECTIVE_TYPE", "dissipation")).strip().lower()
if objective_type in ("dissipation", "power_dissipation", "volume_dissipation"):
    ObjFunctional = J_D_integrand * dx
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
ViscousDissipationFunctional = viscous_dissipation_integrand * dx

state_form = build_state_form(u, p, v, q, rho_effective, dx, nu_tilde_frozen)
flow_test_u, flow_test_p = TestFunctions(FlowSpace)
forward_flow_residual_form = build_state_form(
    u, p, flow_test_u, flow_test_p, rho_effective, dx, nu_tilde_frozen,
)
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
    viscous_stress_form(mu_effective, u_lin, v_lin)
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
            if use_outlet_velocity_bc else [])
         + [DirichletBC(VelocitySpace.sub(spec["component"]), spec["value"], boundaries, spec["marker"])
            for spec in pressure_outlet_component_bcs])
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
           + viscous_stress_form(mu_effective, u_pc_tr, z_pc)
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
def initialize_forward_guess_with_stokes(reset_turbulence=True):
    """Build a Stokes-Brinkman flow guess with the current frozen viscosity."""
    if reset_turbulence:
        wall_dist_scale = max(float(globals().get("SA_INIT_WALL_DIST_SCALE", 0.05 * float(globals().get("L", 1.0)))), 1.0e-12)
        nu_guess = project(
            Constant(sa_nu_tilde_init) * wall_distance / (wall_distance + Constant(wall_dist_scale)),
            TurbulenceSpace,
        )
        nu_tilde_frozen.assign(nu_guess)
        enforce_scalar_bounds_inplace(nu_tilde_frozen, sa_nu_tilde_floor, sa_nu_tilde_ceiling)
        sa_model.nu_tilde0.assign(nu_tilde_frozen)
        sa_model.nu_tilde1.assign(nu_tilde_frozen)

    A = assemble(a_stokes)
    b = assemble(l_stokes)
    for bc in bc_NS:
        bc.apply(A, b)
    solve(A, w_fwd.vector(), b, LINEAR_SOLVER_NAME)


def constrained_forward_residual_norm():
    """Measure the frozen-flow residual with Dirichlet rows treated consistently."""
    residual_vector = assemble(forward_flow_residual_form)
    fallback_residual_values = None
    current_state_values = None
    for bc in bc_NS:
        try:
            bc.apply(residual_vector, w_fwd.vector())
        except TypeError:
            if fallback_residual_values is None:
                fallback_residual_values = residual_vector.get_local()
                current_state_values = w_fwd.vector().get_local()
            for dof, value in bc.get_boundary_values().items():
                fallback_residual_values[dof] = current_state_values[dof] - float(value)
    if fallback_residual_values is not None:
        residual_vector.set_local(fallback_residual_values)
        residual_vector.apply("insert")
    return float(residual_vector.norm("l2"))


def _unit_interval(value):
    return min(max(float(value), 0.0), 1.0)


def _coerce_forward_snes_convection_schedule(raw_schedule):
    if raw_schedule is None:
        return [{"convection_weight": 1.0}]
    if isinstance(raw_schedule, np.ndarray):
        raw_schedule = raw_schedule.tolist()
    elif not isinstance(raw_schedule, (list, tuple)):
        raw_schedule = [raw_schedule]

    schedule = []
    for step_idx, raw_step in enumerate(raw_schedule, start=1):
        if isinstance(raw_step, dict):
            if "convection_weight" in raw_step:
                raw_weight = raw_step["convection_weight"]
            elif "convective_weight" in raw_step:
                raw_weight = raw_step["convective_weight"]
            elif "inertia_weight" in raw_step:
                raw_weight = raw_step["inertia_weight"]
            elif "weight" in raw_step:
                raw_weight = raw_step["weight"]
            else:
                raise ValueError(
                    "FORWARD_SNES_CONVECTION_SCHEDULE step {} must define "
                    "'convection_weight' or 'weight'.".format(step_idx)
                )
            step = {"convection_weight": _unit_interval(raw_weight)}
            for key in ("method", "line_search"):
                if key in raw_step:
                    step[key] = str(raw_step[key])
            for key in ("rtol", "atol", "accept_norm"):
                if key in raw_step:
                    step[key] = float(raw_step[key])
            if "accept_growth" in raw_step:
                step["accept_growth"] = max(1.0, float(raw_step["accept_growth"]))
            elif "accept_residual_growth" in raw_step:
                step["accept_growth"] = max(1.0, float(raw_step["accept_residual_growth"]))
            if "max_iters" in raw_step:
                step["max_iters"] = int(raw_step["max_iters"])
            if "accept_nonconverged" in raw_step:
                step["accept_nonconverged"] = bool(raw_step["accept_nonconverged"])
        else:
            step = {"convection_weight": _unit_interval(raw_step)}
        schedule.append(step)

    if not schedule:
        schedule = [{"convection_weight": 1.0}]
    if abs(schedule[-1]["convection_weight"] - 1.0) > 1.0e-12:
        schedule.append({"convection_weight": 1.0})
    return schedule


def build_forward_snes_convection_schedule():
    configured_schedule = globals().get("FORWARD_SNES_CONVECTION_SCHEDULE", None)
    startup_schedule = globals().get("FORWARD_SNES_STARTUP_CONVECTION_SCHEDULE", None)
    if int(globals().get("iter_count", 0)) == 0 and startup_schedule is not None:
        configured_schedule = startup_schedule
    return _coerce_forward_snes_convection_schedule(configured_schedule)


def build_forward_snes_recovery_attempts():
    base_attempt = {
        "label": "primary solve",
        "method": str(globals().get("FORWARD_SNES_METHOD", "newtonls")),
        "line_search": str(globals().get("FORWARD_SNES_LINE_SEARCH", "bt")),
        "rtol": float(globals().get("FORWARD_SNES_RTOL", 1.0e-6)),
        "atol": float(globals().get("FORWARD_SNES_ATOL", 1.0e-8)),
        "max_iters": int(globals().get("FORWARD_SNES_MAX_ITERS", 80)),
        "restart_with_stokes": False,
        "accept_norm": None,
        "accept_growth": None,
        "accept_nonconverged": None,
    }
    configured_attempts = globals().get("FORWARD_SNES_RECOVERY_ATTEMPTS", None)
    if configured_attempts is None:
        return [base_attempt]

    if isinstance(configured_attempts, np.ndarray):
        configured_attempts = configured_attempts.tolist()
    elif isinstance(configured_attempts, dict):
        configured_attempts = [configured_attempts]

    attempts = [base_attempt]
    for attempt_idx, attempt_spec in enumerate(configured_attempts, start=1):
        if not isinstance(attempt_spec, dict):
            raise TypeError("FORWARD_SNES_RECOVERY_ATTEMPTS entries must be dictionaries.")
        attempt = dict(base_attempt)
        attempt.update(attempt_spec)
        attempt["label"] = str(attempt.get("label", "recovery attempt {}".format(attempt_idx)))
        attempt["method"] = str(attempt.get("method", base_attempt["method"]))
        attempt["line_search"] = str(attempt.get("line_search", base_attempt["line_search"]))
        attempt["rtol"] = float(attempt.get("rtol", base_attempt["rtol"]))
        attempt["atol"] = float(attempt.get("atol", base_attempt["atol"]))
        attempt["max_iters"] = int(attempt.get("max_iters", base_attempt["max_iters"]))
        attempt["restart_with_stokes"] = bool(attempt.get("restart_with_stokes", False))
        if attempt.get("accept_norm") is not None:
            attempt["accept_norm"] = float(attempt["accept_norm"])
        if attempt.get("accept_growth") is not None:
            attempt["accept_growth"] = max(1.0, float(attempt["accept_growth"]))
        elif attempt.get("accept_residual_growth") is not None:
            attempt["accept_growth"] = max(1.0, float(attempt["accept_residual_growth"]))
        if attempt.get("accept_nonconverged") is not None:
            attempt["accept_nonconverged"] = bool(attempt["accept_nonconverged"])
        attempts.append(attempt)
    return attempts


def describe_forward_snes_attempt(attempt):
    description = [
        "method={}".format(attempt["method"]),
        "max_it={}".format(attempt["max_iters"]),
        "rtol={:.1e}".format(attempt["rtol"]),
        "atol={:.1e}".format(attempt["atol"]),
    ]
    if attempt["method"] == "newtonls":
        description.append("line_search={}".format(attempt["line_search"]))
    return ", ".join(description)


def solve_forward_snes_once(
    method_override=None,
    line_search_override=None,
    max_iters_override=None,
    rtol_override=None,
    atol_override=None,
    accept_function_norm_override=None,
    accept_nonconverged_override=None,
    accept_residual_growth_override=None,
):
    """Solve the frozen-viscosity Navier-Stokes system once."""
    solver_log("      [SNES] monolithic frozen-viscosity flow")
    jac_forward = derivative(forward_flow_residual_form, w_fwd)
    problem_forward = NonlinearVariationalProblem(
        forward_flow_residual_form, w_fwd, bc_NS, jac_forward,
    )
    solver_forward = NonlinearVariationalSolver(problem_forward)
    solver_forward.parameters["nonlinear_solver"] = "snes"
    solver_forward.parameters["snes_solver"]["linear_solver"] = str(
        globals().get("FORWARD_SNES_LINEAR_SOLVER", LINEAR_SOLVER_NAME)
    )
    method = method_override or str(globals().get("FORWARD_SNES_METHOD", "newtonls"))
    solver_forward.parameters["snes_solver"]["method"] = method
    if method == "newtonls":
        solver_forward.parameters["snes_solver"]["line_search"] = str(
            globals().get("FORWARD_SNES_LINE_SEARCH", "bt")
            if line_search_override is None else line_search_override
        )

    rtol = float(globals().get("FORWARD_SNES_RTOL", 1.0e-6) if rtol_override is None else rtol_override)
    atol = float(globals().get("FORWARD_SNES_ATOL", 1.0e-8) if atol_override is None else atol_override)
    max_iters = int(globals().get("FORWARD_SNES_MAX_ITERS", 80) if max_iters_override is None else max_iters_override)
    solver_forward.parameters["snes_solver"]["relative_tolerance"] = rtol
    solver_forward.parameters["snes_solver"]["absolute_tolerance"] = atol
    solver_forward.parameters["snes_solver"]["maximum_iterations"] = max_iters
    accept_nonconverged = bool(
        globals().get("FORWARD_SNES_ACCEPT_NONCONVERGED_WITH_ACCEPT_NORM", False)
        if accept_nonconverged_override is None else accept_nonconverged_override
    )
    error_on_nonconvergence = bool(globals().get("FORWARD_SNES_ERROR_ON_NONCONVERGENCE", True))
    if accept_nonconverged and (
        accept_function_norm_override is not None or accept_residual_growth_override is not None
    ):
        error_on_nonconvergence = False
    solver_forward.parameters["snes_solver"]["error_on_nonconvergence"] = error_on_nonconvergence

    initial_norm = constrained_forward_residual_norm()
    accepted_norm = (
        max(atol, rtol * max(initial_norm, 1.0e-16))
        if accept_function_norm_override is None
        else float(accept_function_norm_override)
    )
    accepted_factor = max(1.0, float(globals().get("FORWARD_SNES_ACCEPTED_RESIDUAL_FACTOR", 2.0)))
    accepted_limit = accepted_factor * accepted_norm
    if accept_residual_growth_override is not None:
        accepted_limit = max(accepted_limit, float(accept_residual_growth_override) * initial_norm)
    max_absolute_residual = globals().get("FORWARD_SNES_MAX_ACCEPTED_ABSOLUTE_RESIDUAL", None)
    if (
        accept_nonconverged
        and bool(globals().get("FORWARD_SNES_ACCEPT_INITIAL_IF_WITHIN_ACCEPT_NORM", True))
        and initial_norm <= accepted_limit
        and (
            max_absolute_residual is None
            or initial_norm <= float(max_absolute_residual)
        )
    ):
        solver_log(
            "      [SNES] initial iterate already satisfies accept limit "
            "{:.2e} <= {:.2e}; skipping Newton update".format(
                initial_norm,
                accepted_limit,
            )
        )
        return 1.0, float(initial_norm)

    if (
        accept_nonconverged
        and bool(globals().get("FORWARD_SNES_STOP_AT_ACCEPT_NORM", False))
        and (accept_function_norm_override is not None or accept_residual_growth_override is not None)
    ):
        solve_stop_factor = max(0.0, float(globals().get("FORWARD_SNES_ACCEPT_NORM_SOLVE_FACTOR", 1.0)))
        solve_atol = solve_stop_factor * accepted_limit
        if max_absolute_residual is not None:
            solve_atol = min(solve_atol, float(max_absolute_residual))
        if solve_atol > atol:
            solver_forward.parameters["snes_solver"]["absolute_tolerance"] = solve_atol
            solver_log(
                "      [SNES] solve atol {:.2e} from accept limit {:.2e}".format(
                    solve_atol, accepted_limit,
                )
            )

    previous_state_values = w_fwd.vector().get_local()
    try:
        solve_result = solver_forward.solve()
    except RuntimeError:
        w_fwd.vector().set_local(previous_state_values)
        w_fwd.vector().apply("insert")
        raise
    solver_converged = True
    if isinstance(solve_result, tuple) and len(solve_result) >= 2:
        solver_converged = bool(solve_result[1])
    residual_norm = constrained_forward_residual_norm()
    if not np.isfinite(residual_norm):
        w_fwd.vector().set_local(previous_state_values)
        w_fwd.vector().apply("insert")
        raise RuntimeError("Accepted SNES forward iterate produced a non-finite residual norm.")
    if max_absolute_residual is not None and residual_norm > float(max_absolute_residual):
        w_fwd.vector().set_local(previous_state_values)
        w_fwd.vector().apply("insert")
        raise RuntimeError(
            "Accepted SNES forward iterate has residual norm {:.4e}, exceeding the "
            "absolute acceptance limit {:.4e}.".format(
                residual_norm,
                float(max_absolute_residual),
            )
        )
    if residual_norm > accepted_limit:
        w_fwd.vector().set_local(previous_state_values)
        w_fwd.vector().apply("insert")
        raise RuntimeError(
            "Accepted SNES forward iterate has residual norm {:.4e}, exceeding {:.4e} "
            "(base tolerance {:.4e}, factor {:.2f}).".format(
                residual_norm,
                accepted_limit,
                accepted_norm,
                accepted_factor,
            )
        )
    if not solver_converged:
        solver_log(
            "      [SNES] accepted nonconverged continuation iterate with residual {:.2e} <= {:.2e}".format(
                residual_norm, accepted_limit,
            )
        )
    relative_residual = residual_norm / max(initial_norm, 1.0e-16)
    solver_log(
        "      [SNES] residual {:.2e} (relative {:.2e})".format(
            residual_norm, relative_residual,
        )
    )
    return float(relative_residual), float(residual_norm)


def solve_forward_snes_attempt(attempt, convection_schedule):
    result = (np.inf, np.inf)
    pending_steps = [dict(step) for step in convection_schedule]
    adaptive_convection = bool(globals().get("FORWARD_SNES_ADAPTIVE_CONVECTION", False))
    min_convection_step = max(0.0, float(globals().get("FORWARD_SNES_MIN_CONVECTION_STEP", 0.01)))
    max_insertions = max(0, int(globals().get("FORWARD_SNES_MAX_ADAPTIVE_CONVECTION_STEPS", 16)))
    accepted_convection_weight = None
    inserted_steps = 0
    step_idx = 0
    while step_idx < len(pending_steps):
        step = pending_steps[step_idx]
        convection_weight = float(step["convection_weight"])
        forward_convection_coupling_weight.assign(convection_weight)
        step_attempt = dict(attempt)
        for key in ("method", "line_search", "rtol", "atol", "max_iters", "accept_norm", "accept_growth", "accept_nonconverged"):
            if key in step:
                step_attempt[key] = step[key]
        if len(pending_steps) > 1:
            details = []
            if "max_iters" in step and step_attempt.get("max_iters") is not None:
                details.append("max_it={}".format(step_attempt["max_iters"]))
            if "atol" in step and step_attempt.get("atol") is not None:
                details.append("atol={:.1e}".format(step_attempt["atol"]))
            if "accept_norm" in step:
                if step_attempt.get("accept_norm") is None:
                    details.append("accept=off")
                else:
                    details.append("accept={:.1e}".format(step_attempt["accept_norm"]))
            if "accept_growth" in step and step_attempt.get("accept_growth") is not None:
                details.append("growth<={:.2f}".format(step_attempt["accept_growth"]))
            if "rtol" in step and step_attempt.get("rtol") is not None:
                details.append("rtol={:.1e}".format(step_attempt["rtol"]))
            solver_log(
                "      [SNES continuation] step {}/{}: convection = {:.2f}{}".format(
                    step_idx + 1,
                    len(pending_steps),
                    convection_weight,
                    "" if not details else " ({})".format(", ".join(details)),
                )
            )
        try:
            result = solve_forward_snes_once(
                method_override=step_attempt["method"],
                line_search_override=step_attempt["line_search"],
                max_iters_override=step_attempt["max_iters"],
                rtol_override=step_attempt["rtol"],
                atol_override=step_attempt["atol"],
                accept_function_norm_override=step_attempt.get("accept_norm"),
                accept_nonconverged_override=step_attempt.get("accept_nonconverged"),
                accept_residual_growth_override=step_attempt.get("accept_growth"),
            )
        except RuntimeError:
            if (
                adaptive_convection
                and accepted_convection_weight is not None
                and inserted_steps < max_insertions
                and abs(convection_weight - accepted_convection_weight) > min_convection_step
            ):
                midpoint_weight = 0.5 * (accepted_convection_weight + convection_weight)
                midpoint_step = dict(step)
                midpoint_step["convection_weight"] = midpoint_weight
                pending_steps.insert(step_idx, midpoint_step)
                inserted_steps += 1
                solver_log(
                    "      [SNES continuation] splitting failed convection step {:.2f}->{:.2f}; "
                    "retrying midpoint {:.2f}".format(
                        accepted_convection_weight, convection_weight, midpoint_weight,
                    )
                )
                continue
            raise
        accepted_convection_weight = convection_weight
        step_idx += 1
    return result


def solve_forward_snes(solve_label=None):
    """Solve the frozen-viscosity Navier-Stokes system with configured recovery."""
    global snes_ipcs_warm_start_done
    recovery_attempts = build_forward_snes_recovery_attempts()
    convection_schedule = build_forward_snes_convection_schedule()
    label_text = "" if solve_label is None else str(solve_label).lower()
    # The adjoint needs a true monolithic residual, not a loose continuation accept.
    if "final" in label_text and bool(globals().get("FORWARD_SNES_STRICT_FINAL_SOLVE", True)):
        for attempt in recovery_attempts:
            attempt["accept_norm"] = None
            attempt["accept_growth"] = None
            attempt["accept_nonconverged"] = False
        for step in convection_schedule:
            if abs(float(step["convection_weight"]) - 1.0) < 1.0e-12:
                step["accept_norm"] = None
                step["accept_growth"] = None
                step["accept_nonconverged"] = False
    entry_state_values = w_fwd.vector().get_local()
    warm_start_with_ipcs = bool(globals().get("FORWARD_SNES_WARM_START_WITH_IPCS", False))
    warm_start_mode = str(globals().get("FORWARD_SNES_IPCS_WARM_START_MODE", "initial")).strip().lower()
    if warm_start_with_ipcs:
        # IPCS is only a pseudo-transient initializer; SNES still owns acceptance.
        run_ipcs_warm_start = False
        if warm_start_mode == "always":
            run_ipcs_warm_start = True
        elif warm_start_mode == "final":
            run_ipcs_warm_start = "final" in label_text
        else:
            run_ipcs_warm_start = not snes_ipcs_warm_start_done
        if run_ipcs_warm_start:
            solver_log("      [SNES warm start] IPCS pseudo-transient initialization")
            try:
                solve_forward_ipcs(
                    "{}_ipcs_warm_start".format(solve_label if solve_label is not None else "snes")
                )
            except RuntimeError as exc:
                root_print(
                    "Warning: IPCS warm start did not meet its acceptance criteria; "
                    "continuing to SNES from the best IPCS iterate. {}".format(exc)
                )
            snes_ipcs_warm_start_done = True
            entry_state_values = w_fwd.vector().get_local()
    num_retries = max(0, len(recovery_attempts) - 1)
    try:
        for attempt_idx, attempt in enumerate(recovery_attempts):
            if attempt_idx > 0:
                retry_idx = attempt_idx
                if attempt["restart_with_stokes"]:
                    root_print(
                        "  Forward SNES diverged; rebuilding Stokes-Brinkman flow guess before retry {}/{} "
                        "({}: {}).".format(
                            retry_idx,
                            num_retries,
                            attempt["label"],
                            describe_forward_snes_attempt(attempt),
                        )
                    )
                    initialize_forward_guess_with_stokes(reset_turbulence=False)
                else:
                    root_print(
                        "  Forward SNES diverged; retrying from the current iterate with retry {}/{} "
                        "({}: {}).".format(
                            retry_idx,
                            num_retries,
                            attempt["label"],
                            describe_forward_snes_attempt(attempt),
                        )
                    )
            try:
                return solve_forward_snes_attempt(attempt, convection_schedule)
            except RuntimeError:
                if attempt_idx == len(recovery_attempts) - 1:
                    raise
    except RuntimeError:
        w_fwd.vector().set_local(entry_state_values)
        w_fwd.vector().apply("insert")
        raise
    finally:
        forward_convection_coupling_weight.assign(1.0)


def solve_forward(solve_label=None):
    """Dispatch the final forward solve."""
    if FORWARD_FLOW_SOLVER == "snes":
        return solve_forward_snes(solve_label)
    if FORWARD_FLOW_SOLVER == "ipcs_snes_polish":
        ipcs_result = solve_forward_ipcs(solve_label, allow_best_without_acceptance=True)
        solver_log("      [IPCS] polishing best IPCS iterate with monolithic SNES")
        solve_forward_snes("{}_snes_polish".format(solve_label))
        return ipcs_result
    return solve_forward_ipcs(solve_label)


def solve_forward_picard(solve_label=None):
    """Use a configurable cheaper solve for frozen-SA Picard updates."""
    if FORWARD_PICARD_FLOW_SOLVER == "snes":
        return solve_forward_snes(solve_label)
    if FORWARD_PICARD_FLOW_SOLVER == "ipcs":
        return solve_forward_ipcs(solve_label)
    raise ValueError("FORWARD_PICARD_FLOW_SOLVER must be either 'ipcs' or 'snes'.")


def solve_forward_finite_difference(solve_label=None):
    """Use a configurable forward solve for finite-difference validation."""
    if FINITE_DIFFERENCE_CHECK_FLOW_SOLVER == "snes":
        return solve_forward_snes(solve_label)
    if FINITE_DIFFERENCE_CHECK_FLOW_SOLVER == "ipcs_snes_polish":
        ipcs_result = solve_forward_ipcs(solve_label, allow_best_without_acceptance=True)
        solver_log("      [IPCS] polishing best FD IPCS iterate with monolithic SNES")
        solve_forward_snes("{}_snes_polish".format(solve_label))
        return ipcs_result
    return solve_forward_ipcs(solve_label)


def solve_forward_finite_difference_picard(solve_label=None):
    """Use a configurable Picard solve for finite-difference updated-SA checks."""
    if FINITE_DIFFERENCE_CHECK_PICARD_FLOW_SOLVER == "snes":
        return solve_forward_snes(solve_label)
    return solve_forward_ipcs(solve_label)


def finite_difference_check_iterations():
    raw_iterations = globals().get("FINITE_DIFFERENCE_CHECK_ITERATIONS", (0,))
    if isinstance(raw_iterations, np.ndarray):
        raw_iterations = raw_iterations.tolist()
    elif not isinstance(raw_iterations, (list, tuple, set)):
        raw_iterations = [raw_iterations]
    return {int(value) for value in raw_iterations}


def finite_difference_sample_active_positions(global_iter, sample_count):
    raw_positions = globals().get("FINITE_DIFFERENCE_CHECK_ACTIVE_POSITIONS", None)
    raw_dofs = globals().get("FINITE_DIFFERENCE_CHECK_DOF_INDICES", None)
    active_position_by_dof = {int(dof): idx for idx, dof in enumerate(ActiveDV.tolist())}

    if raw_dofs is not None:
        if isinstance(raw_dofs, np.ndarray):
            raw_dofs = raw_dofs.tolist()
        elif not isinstance(raw_dofs, (list, tuple, set)):
            raw_dofs = [raw_dofs]
        positions = []
        for dof in raw_dofs:
            dof = int(dof)
            if dof not in active_position_by_dof:
                raise ValueError(
                    "FINITE_DIFFERENCE_CHECK_DOF_INDICES contains inactive/unknown dof {}.".format(dof)
                )
            positions.append(active_position_by_dof[dof])
        return np.asarray(positions, dtype=np.int64)

    if raw_positions is not None:
        if isinstance(raw_positions, np.ndarray):
            raw_positions = raw_positions.tolist()
        elif not isinstance(raw_positions, (list, tuple, set)):
            raw_positions = [raw_positions]
        positions = np.asarray([int(value) for value in raw_positions], dtype=np.int64)
        if np.any(positions < 0) or np.any(positions >= int(ActiveDV.size)):
            raise ValueError("FINITE_DIFFERENCE_CHECK_ACTIVE_POSITIONS contains an out-of-range index.")
        return positions

    rng = np.random.RandomState(int(globals().get("FINITE_DIFFERENCE_CHECK_SEED", 13)) + int(global_iter))
    return rng.choice(int(ActiveDV.size), size=sample_count, replace=False)


def density_dof_coordinates():
    try:
        coords = DensitySpace.tabulate_dof_coordinates()
        return coords.reshape((DensitySpace.dim(), -1))
    except RuntimeError:
        return None


def initialize_sensitivity_check_log(log_path):
    if not bool(globals().get("RUN_FINITE_DIFFERENCE_CHECKS", False)):
        return
    if not IS_ROOT:
        return
    columns = [
        "Stage",
        "InnerIter",
        "GlobalIter",
        "Mode",
        "CV",
        "ActivePosition",
        "DensityDof",
        "X",
        "Y",
        "Step",
        "BaseObjective",
        "AdjointDerivative",
        "FiniteDifferenceDerivative",
        "AbsoluteError",
        "RelativeError",
        "PlusObjective",
        "MinusObjective",
    ]
    with open(log_path, "w") as handle:
        handle.write(format_sensitivity_check_log_row(columns) + "\n")


def append_sensitivity_check_log_entry(log_path, values):
    if not bool(globals().get("RUN_FINITE_DIFFERENCE_CHECKS", False)):
        return
    if not IS_ROOT:
        return
    with open(log_path, "a") as handle:
        handle.write(format_sensitivity_check_log_row(values) + "\n")


def sensitivity_verification_mode_label(default_name="frozen"):
    return str(globals().get("SENSITIVITY_VERIFICATION_MODE_NAME", default_name))


def sensitivity_verification_derivative_column(default_name="AdjointDerivative"):
    return str(globals().get("SENSITIVITY_VERIFICATION_DERIVATIVE_COLUMN", default_name))


def sanitize_sensitivity_label(raw_label):
    text = "solve" if raw_label is None else str(raw_label)
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in text)
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_") or "solve"


SENSITIVITY_CHECK_LOG_COLUMN_WIDTHS = (
    7,
    9,
    10,
    18,
    5,
    14,
    10,
    23,
    23,
    23,
    23,
    23,
    26,
    23,
    23,
    23,
    23,
)


SENSITIVITY_VERIFICATION_COLUMN_WIDTHS = (
    5,
    14,
    10,
    23,
    23,
    23,
    23,
    26,
    23,
    23,
    23,
)


def format_fixed_width_row(values, column_widths, left_aligned_columns):
    cells = []
    for column_idx, (value, width) in enumerate(zip(values, column_widths)):
        text = str(value)
        if column_idx in left_aligned_columns:
            cells.append(text.ljust(width))
        else:
            cells.append(text.rjust(width))
    return "  ".join(cells)


def format_sensitivity_check_log_row(values):
    return format_fixed_width_row(values, SENSITIVITY_CHECK_LOG_COLUMN_WIDTHS, {3, 4})


def format_sensitivity_verification_table_row(values):
    return format_fixed_width_row(values, SENSITIVITY_VERIFICATION_COLUMN_WIDTHS, {0})


def initialize_sensitivity_verification_table(log_path):
    if not bool(globals().get("RUN_FINITE_DIFFERENCE_CHECKS", False)):
        return
    if not IS_ROOT:
        return
    derivative_column = sensitivity_verification_derivative_column("FrozenTurbulence")
    columns = [
        "CV",
        "ActivePosition",
        "DensityDof",
        "X",
        "Y",
        "Step",
        "BaseObjective",
        "FiniteDifferenceDerivative",
        derivative_column,
        "AbsoluteError",
        "RelativeError",
    ]
    with open(log_path, "w") as handle:
        handle.write(format_sensitivity_verification_table_row(columns) + "\n")


def append_sensitivity_verification_table_entry(log_path, values):
    if not bool(globals().get("RUN_FINITE_DIFFERENCE_CHECKS", False)):
        return
    if not IS_ROOT:
        return
    with open(log_path, "a") as handle:
        handle.write(format_sensitivity_verification_table_row(values) + "\n")


def taylor_check_steps():
    raw_steps = globals().get("TAYLOR_CHECK_STEPS", (1.0e-3, 3.0e-4, 1.0e-4, 3.0e-5))
    if isinstance(raw_steps, np.ndarray):
        raw_steps = raw_steps.tolist()
    elif not isinstance(raw_steps, (list, tuple, set)):
        raw_steps = [raw_steps]
    return [float(value) for value in raw_steps if float(value) > 0.0]


def taylor_direction_values(entry_count, global_iter):
    raw_values = globals().get("TAYLOR_CHECK_DIRECTION_VALUES", None)
    if raw_values is not None:
        if isinstance(raw_values, np.ndarray):
            values = raw_values.astype(float).copy()
        else:
            if not isinstance(raw_values, (list, tuple)):
                raw_values = [raw_values]
            values = np.asarray([float(value) for value in raw_values], dtype=float)
        if values.size != int(entry_count):
            raise ValueError(
                "TAYLOR_CHECK_DIRECTION_VALUES must contain {} entries, got {}.".format(
                    int(entry_count),
                    int(values.size),
                )
            )
    else:
        rng = np.random.RandomState(int(globals().get("TAYLOR_CHECK_SEED", 29)) + int(global_iter))
        values = rng.choice(np.asarray([-1.0, 1.0]), size=int(entry_count)).astype(float)

    if not np.any(np.abs(values) > 0.0):
        raise ValueError("Taylor check direction has zero norm.")
    if bool(globals().get("TAYLOR_CHECK_NORMALIZE_DIRECTION", False)):
        norm = float(np.linalg.norm(values))
        values = values / norm
    return values


def initialize_taylor_check_log(log_path):
    if not bool(globals().get("RUN_TAYLOR_SENSITIVITY_CHECKS", False)):
        return
    if not IS_ROOT:
        return
    columns = [
        "Stage",
        "InnerIter",
        "GlobalIter",
        "Mode",
        "Direction",
        "Epsilon",
        "FDDenominator",
        "BaseObjective",
        "PlusObjective",
        "MinusObjective",
        "AdjointDirectionalDerivative",
        "CentralFiniteDifferenceDerivative",
        "DirectionalAbsoluteError",
        "DirectionalRelativeError",
        "TaylorRemainderPlus",
        "TaylorRemainderMinus",
        "TaylorRemainderSymmetric",
        "TaylorOrderPlus",
        "TaylorOrderMinus",
        "TaylorOrderSymmetric",
        "ActiveDofCount",
    ]
    with open(log_path, "w") as handle:
        handle.write("\t".join(columns) + "\n")


def append_taylor_check_log_entry(log_path, values):
    if not bool(globals().get("RUN_TAYLOR_SENSITIVITY_CHECKS", False)):
        return
    if not IS_ROOT:
        return
    with open(log_path, "a") as handle:
        handle.write("\t".join(str(value) for value in values) + "\n")


def _format_taylor_float(value):
    if np.isfinite(value):
        return "{:.16e}".format(float(value))
    return "inf"


def write_taylor_direction_table(
    path,
    mode_name,
    direction_name,
    sampled_active_positions,
    direction_values,
    objective_gradient_active,
    coords,
    max_symmetric_eps_values,
):
    if not IS_ROOT:
        return
    with open(path, "w") as handle:
        handle.write(
            "Mode\tDirection\tEntry\tActivePosition\tDensityDof\tX\tY\t"
            "DirectionValue\tAdjointDerivative\tAdjointContribution\tMaxSymmetricEpsilon\n"
        )
        for entry_idx, (active_position, direction_value, max_eps_value) in enumerate(
            zip(sampled_active_positions, direction_values, max_symmetric_eps_values),
            start=1,
        ):
            density_dof = int(ActiveDV[int(active_position)])
            if coords is not None and density_dof < coords.shape[0]:
                xy = coords[density_dof]
                x_coord = "{:.16e}".format(float(xy[0]))
                y_coord = "{:.16e}".format(float(xy[1])) if xy.size > 1 else "nan"
            else:
                x_coord = "nan"
                y_coord = "nan"
            adjoint_derivative = float(objective_gradient_active[int(active_position)])
            handle.write(
                "{}\t{}\t{:d}\t{:d}\t{:d}\t{}\t{}\t{:.16e}\t{:.16e}\t{:.16e}\t{}\n".format(
                    mode_name,
                    direction_name,
                    entry_idx,
                    int(active_position),
                    density_dof,
                    x_coord,
                    y_coord,
                    float(direction_value),
                    adjoint_derivative,
                    adjoint_derivative * float(direction_value),
                    _format_taylor_float(max_eps_value),
                )
            )


def estimate_taylor_order(remainder, previous_remainder, epsilon, previous_epsilon):
    if previous_remainder is None or previous_epsilon is None:
        return np.nan
    if remainder <= 0.0 or previous_remainder <= 0.0:
        return np.nan
    if epsilon <= 0.0 or previous_epsilon <= 0.0 or epsilon == previous_epsilon:
        return np.nan
    return np.log(float(remainder) / float(previous_remainder)) / np.log(
        float(epsilon) / float(previous_epsilon)
    )


def maybe_write_combined_dilgen_table2():
    if not bool(globals().get("RUN_FINITE_DIFFERENCE_CHECKS", False)):
        return
    if not IS_ROOT:
        return
    frozen_root = globals().get("DILGEN_FROZEN_RESULTS_ROOT_NAME")
    semifrozen_root = globals().get("DILGEN_SEMIFROZEN_RESULTS_ROOT_NAME")
    if not frozen_root or not semifrozen_root:
        return
    frozen_table = os.path.join(THIS_DIR, str(frozen_root), "SensitivityVerificationTable.tsv")
    semifrozen_table = os.path.join(THIS_DIR, str(semifrozen_root), "SensitivityVerificationTable.tsv")
    combined_path = os.path.join(results_root, "DilgenTable2_Combined.tsv")
    write_combined_dilgen_table2_if_available(combined_path, frozen_table, semifrozen_table)


def run_finite_difference_checks(stage_idx, inner_iter, global_iter, base_objective, objective_gradient_active):
    """Optional Dilgen-style coordinate checks for the frozen-adjoint gradient."""
    if not bool(globals().get("RUN_FINITE_DIFFERENCE_CHECKS", False)):
        return
    if int(global_iter) not in finite_difference_check_iterations():
        return

    sample_count = min(
        int(globals().get("FINITE_DIFFERENCE_CHECK_SAMPLES", 4)),
        int(ActiveDV.size),
    )
    if sample_count <= 0:
        return

    fd_step = float(globals().get("FINITE_DIFFERENCE_CHECK_STEP", 1.0e-3))
    sampled_active_positions = finite_difference_sample_active_positions(global_iter, sample_count)
    check_updated_turbulence = bool(globals().get("FINITE_DIFFERENCE_CHECK_UPDATED_TURBULENCE", False))
    clip_to_bounds = bool(globals().get("FINITE_DIFFERENCE_CHECK_CLIP_TO_BOUNDS", True))
    coords = density_dof_coordinates()

    base_rho_values = rho.vector().get_local().copy()
    base_rho_f_values = rho_f.vector().get_local().copy()
    base_w_values = w_fwd.vector().get_local().copy()
    base_adj_values = w_adj.vector().get_local().copy()
    base_nu_values = nu_tilde_frozen.vector().get_local().copy()
    base_sa0_values = sa_model.nu_tilde0.vector().get_local().copy()
    base_sa1_values = sa_model.nu_tilde1.vector().get_local().copy()

    def restore_base_state(refresh_wall_distance):
        rho.vector().set_local(base_rho_values)
        rho.vector().apply("insert")
        rho_f.vector().set_local(base_rho_f_values)
        rho_f.vector().apply("insert")
        w_fwd.vector().set_local(base_w_values)
        w_fwd.vector().apply("insert")
        w_adj.vector().set_local(base_adj_values)
        w_adj.vector().apply("insert")
        nu_tilde_frozen.vector().set_local(base_nu_values)
        nu_tilde_frozen.vector().apply("insert")
        sa_model.nu_tilde0.vector().set_local(base_sa0_values)
        sa_model.nu_tilde0.vector().apply("insert")
        sa_model.nu_tilde1.vector().set_local(base_sa1_values)
        sa_model.nu_tilde1.vector().apply("insert")
        if refresh_wall_distance:
            update_wall_distance_field()

    def evaluate_perturbed_objective(perturbations, update_turbulence, label):
        perturbed_values = base_rho_values.copy()
        for active_position, delta in perturbations:
            density_dof = int(ActiveDV[int(active_position)])
            perturbed_values[density_dof] += float(delta)
        if clip_to_bounds:
            perturbed_values = np.clip(perturbed_values, density_lower_values, density_upper_values)
        rho.vector().set_local(perturbed_values)
        rho.vector().apply("insert")
        pde_filter_design_density(rho, rho_f)

        if update_turbulence:
            update_wall_distance_field()
            fd_picard_steps = max(1, int(globals().get("FINITE_DIFFERENCE_CHECK_PICARD_STEPS", 1)))
            for fd_picard_idx in range(fd_picard_steps):
                solve_forward_finite_difference_picard("{}_picard{:02d}".format(label, fd_picard_idx + 1))
                velocity_for_sa = w_fwd.sub(0, deepcopy=True)
                sa_model.construct_forms(velocity_for_sa)
                sa_model.solve_turbulence_model()
                sa_model.update_variables(
                    relaxation=float(globals().get("TURBULENCE_RELAXATION", 1.0))
                )
                nu_tilde_frozen.assign(sa_model.nu_tilde0)
                enforce_scalar_bounds_inplace(nu_tilde_frozen, sa_nu_tilde_floor, sa_nu_tilde_ceiling)
                sa_model.nu_tilde0.assign(nu_tilde_frozen)
                sa_model.nu_tilde1.assign(nu_tilde_frozen)
        solve_forward_finite_difference(label)
        return float(assemble(ObjFunctional))

    def coordinate_step_for_position(active_position):
        density_dof = int(ActiveDV[int(active_position)])
        if clip_to_bounds:
            max_step = min(
                base_rho_values[density_dof] - density_lower_values[density_dof],
                density_upper_values[density_dof] - base_rho_values[density_dof],
            )
            return min(fd_step, 0.5 * max_step)
        return fd_step

    def coordinate_xy(density_dof):
        if coords is not None and int(density_dof) < coords.shape[0]:
            xy = coords[int(density_dof)]
            x_coord = "{:.16e}".format(float(xy[0]))
            y_coord = "{:.16e}".format(float(xy[1])) if xy.size > 1 else "nan"
            return x_coord, y_coord
        return "nan", "nan"

    def symmetric_taylor_epsilon_limits(active_positions, direction_values):
        if not clip_to_bounds:
            return np.full(len(active_positions), np.inf, dtype=float), np.inf
        limits = []
        for active_position, direction_value in zip(active_positions, direction_values):
            abs_direction = abs(float(direction_value))
            if abs_direction <= 0.0:
                limits.append(np.inf)
                continue
            density_dof = int(ActiveDV[int(active_position)])
            lower_margin = float(base_rho_values[density_dof] - density_lower_values[density_dof])
            upper_margin = float(density_upper_values[density_dof] - base_rho_values[density_dof])
            limits.append(max(0.0, min(lower_margin, upper_margin) / abs_direction))
        limit_values = np.asarray(limits, dtype=float)
        finite_limits = limit_values[np.isfinite(limit_values)]
        if finite_limits.size == 0:
            return limit_values, np.inf
        return limit_values, float(np.min(finite_limits))

    def run_taylor_check_for_mode(mode_name, update_turbulence):
        if not bool(globals().get("RUN_TAYLOR_SENSITIVITY_CHECKS", False)):
            return
        steps = taylor_check_steps()
        if not steps:
            return

        direction_values = taylor_direction_values(len(sampled_active_positions), global_iter)
        max_eps_values, max_symmetric_eps = symmetric_taylor_epsilon_limits(
            sampled_active_positions,
            direction_values,
        )
        direction_name = "{}_stage{:02d}_iter{:03d}_global{:03d}".format(
            sanitize_sensitivity_label(mode_name),
            int(stage_idx),
            int(inner_iter),
            int(global_iter),
        )
        direction_path = os.path.join(results_root, "TaylorDirection_{}.tsv".format(direction_name))
        write_taylor_direction_table(
            direction_path,
            mode_name,
            direction_name,
            sampled_active_positions,
            direction_values,
            objective_gradient_active,
            coords,
            max_eps_values,
        )

        adjoint_directional_derivative = float(
            np.dot(objective_gradient_active[sampled_active_positions], direction_values)
        )
        root_print(
            "  [Taylor check] mode {} using {} active entries, adjoint g.p={:.6e}".format(
                mode_name,
                len(sampled_active_positions),
                adjoint_directional_derivative,
            )
        )

        previous_eps = None
        previous_remainder_plus = None
        previous_remainder_minus = None
        previous_remainder_symmetric = None
        for eps_value in steps:
            eps_value = float(eps_value)
            if clip_to_bounds and eps_value >= max_symmetric_eps * (1.0 - 1.0e-12):
                root_print(
                    "    Taylor eps {:.3e} skipped: symmetric bound limit is {:.3e}.".format(
                        eps_value,
                        max_symmetric_eps,
                    )
                )
                continue

            perturb_plus = [
                (active_position, eps_value * float(direction_value))
                for active_position, direction_value in zip(sampled_active_positions, direction_values)
            ]
            perturb_minus = [
                (active_position, -eps_value * float(direction_value))
                for active_position, direction_value in zip(sampled_active_positions, direction_values)
            ]

            restore_base_state(refresh_wall_distance=update_turbulence)
            plus_objective = evaluate_perturbed_objective(
                perturb_plus,
                update_turbulence,
                "taylor_{}_eps{:.0e}_plus".format(mode_name, eps_value),
            )
            restore_base_state(refresh_wall_distance=update_turbulence)
            minus_objective = evaluate_perturbed_objective(
                perturb_minus,
                update_turbulence,
                "taylor_{}_eps{:.0e}_minus".format(mode_name, eps_value),
            )

            fd_denominator = 2.0 * eps_value
            central_fd = (plus_objective - minus_objective) / fd_denominator
            directional_abs_error = abs(central_fd - adjoint_directional_derivative)
            directional_rel_error = directional_abs_error / max(
                abs(central_fd),
                abs(adjoint_directional_derivative),
                1.0e-30,
            )
            remainder_plus = abs(
                plus_objective - float(base_objective)
                - eps_value * adjoint_directional_derivative
            )
            remainder_minus = abs(
                minus_objective - float(base_objective)
                + eps_value * adjoint_directional_derivative
            )
            remainder_symmetric = abs(plus_objective + minus_objective - 2.0 * float(base_objective))
            order_plus = estimate_taylor_order(
                remainder_plus,
                previous_remainder_plus,
                eps_value,
                previous_eps,
            )
            order_minus = estimate_taylor_order(
                remainder_minus,
                previous_remainder_minus,
                eps_value,
                previous_eps,
            )
            order_symmetric = estimate_taylor_order(
                remainder_symmetric,
                previous_remainder_symmetric,
                eps_value,
                previous_eps,
            )

            root_print(
                "    eps={:.3e} adj={:.6e} fd={:.6e} abs_err={:.3e} rel_err={:.3e}".format(
                    eps_value,
                    adjoint_directional_derivative,
                    central_fd,
                    directional_abs_error,
                    directional_rel_error,
                )
            )
            append_taylor_check_log_entry(
                taylor_check_log_path,
                [
                    int(stage_idx),
                    int(inner_iter),
                    int(global_iter),
                    mode_name,
                    direction_name,
                    "{:.16e}".format(eps_value),
                    "{:.16e}".format(fd_denominator),
                    "{:.16e}".format(float(base_objective)),
                    "{:.16e}".format(float(plus_objective)),
                    "{:.16e}".format(float(minus_objective)),
                    "{:.16e}".format(float(adjoint_directional_derivative)),
                    "{:.16e}".format(float(central_fd)),
                    "{:.16e}".format(float(directional_abs_error)),
                    "{:.16e}".format(float(directional_rel_error)),
                    "{:.16e}".format(float(remainder_plus)),
                    "{:.16e}".format(float(remainder_minus)),
                    "{:.16e}".format(float(remainder_symmetric)),
                    "{:.6e}".format(float(order_plus)) if np.isfinite(order_plus) else "nan",
                    "{:.6e}".format(float(order_minus)) if np.isfinite(order_minus) else "nan",
                    "{:.6e}".format(float(order_symmetric)) if np.isfinite(order_symmetric) else "nan",
                    int(len(sampled_active_positions)),
                ],
            )

            previous_eps = eps_value
            previous_remainder_plus = remainder_plus
            previous_remainder_minus = remainder_minus
            previous_remainder_symmetric = remainder_symmetric

    root_print(
        "  [FD check] stage {} iter {} global {} with {} coordinate sample(s).".format(
            stage_idx, inner_iter, global_iter, sample_count,
        )
    )
    try:
        base_mode_name = sensitivity_verification_mode_label("frozen")
        for mode_name, update_turbulence in (
            (base_mode_name, False),
            ("updated-SA-wall", True),
        ):
            if update_turbulence and not check_updated_turbulence:
                continue
            root_print("  [FD check] mode: {}".format(mode_name))
            for cv_idx, active_position in enumerate(sampled_active_positions, start=1):
                density_dof = int(ActiveDV[int(active_position)])
                step = coordinate_step_for_position(active_position)
                if step <= 1.0e-12:
                    root_print(
                        "    dof {} skipped: insufficient bound margin for FD step.".format(
                            density_dof,
                        )
                    )
                    continue
                restore_base_state(refresh_wall_distance=update_turbulence)
                plus_objective = evaluate_perturbed_objective(
                    [(active_position, step)],
                    update_turbulence,
                    "fd_final_{}_dof{}_plus".format(mode_name, density_dof),
                )
                restore_base_state(refresh_wall_distance=update_turbulence)
                minus_objective = evaluate_perturbed_objective(
                    [(active_position, -step)],
                    update_turbulence,
                    "fd_final_{}_dof{}_minus".format(mode_name, density_dof),
                )
                fd_derivative = (plus_objective - minus_objective) / (2.0 * step)
                adjoint_derivative = float(objective_gradient_active[active_position])
                absolute_error = abs(fd_derivative - adjoint_derivative)
                rel_error = absolute_error / max(
                    abs(fd_derivative),
                    abs(adjoint_derivative),
                    1.0e-30,
                )
                root_print(
                    "    CV{} dof {} adj={:.6e} fd={:.6e} abs_err={:.3e} rel_err={:.3e} J0={:.6e}".format(
                        cv_idx,
                        density_dof,
                        adjoint_derivative,
                        fd_derivative,
                        absolute_error,
                        rel_error,
                        float(base_objective),
                    )
                )
                x_coord, y_coord = coordinate_xy(density_dof)
                append_sensitivity_check_log_entry(
                    sensitivity_check_log_path,
                    [
                        int(stage_idx),
                        int(inner_iter),
                        int(global_iter),
                        mode_name,
                        "CV{}".format(cv_idx),
                        int(active_position),
                        int(density_dof),
                        x_coord,
                        y_coord,
                        "{:.16e}".format(float(step)),
                        "{:.16e}".format(float(base_objective)),
                        "{:.16e}".format(float(adjoint_derivative)),
                        "{:.16e}".format(float(fd_derivative)),
                        "{:.16e}".format(float(absolute_error)),
                        "{:.16e}".format(float(rel_error)),
                        "{:.16e}".format(float(plus_objective)),
                        "{:.16e}".format(float(minus_objective)),
                    ],
                )
                append_sensitivity_verification_table_entry(
                    sensitivity_verification_table_path,
                    [
                        "CV{}".format(cv_idx),
                        int(active_position),
                        int(density_dof),
                        x_coord,
                        y_coord,
                        "{:.16e}".format(float(step)),
                        "{:.16e}".format(float(base_objective)),
                        "{:.16e}".format(float(fd_derivative)),
                        "{:.16e}".format(float(adjoint_derivative)),
                        "{:.16e}".format(float(absolute_error)),
                        "{:.16e}".format(float(rel_error)),
                    ],
                )
            run_taylor_check_for_mode(mode_name, update_turbulence)
        maybe_write_combined_dilgen_table2()
    finally:
        restore_base_state(refresh_wall_distance=check_updated_turbulence)


def ipcs_accept_best_score_for_label(solve_label, default_score):
    """Return the near-steady IPCS acceptance score for this solve role."""
    label_text = "" if solve_label is None else str(solve_label).lower()
    if "final" in label_text:
        return float(globals().get("FORWARD_IPCS_FINAL_ACCEPT_BEST_SCORE", default_score))
    if "picard" in label_text:
        return float(globals().get("FORWARD_IPCS_PICARD_ACCEPT_BEST_SCORE", default_score))
    return float(default_score)


def ipcs_tolerances_for_label(solve_label, default_rtol_u, default_rtol_p):
    """Return role-specific IPCS steady-state tolerances when configured."""
    label_text = "" if solve_label is None else str(solve_label).lower()
    rtol_u = default_rtol_u
    rtol_p = default_rtol_p
    if "final" in label_text:
        rtol_u = globals().get("FORWARD_IPCS_FINAL_VELOCITY_RTOL", rtol_u)
        rtol_p = globals().get("FORWARD_IPCS_FINAL_PRESSURE_RTOL", rtol_p)
    elif "picard" in label_text:
        rtol_u = globals().get("FORWARD_IPCS_PICARD_VELOCITY_RTOL", rtol_u)
        rtol_p = globals().get("FORWARD_IPCS_PICARD_PRESSURE_RTOL", rtol_p)
    return float(rtol_u), float(rtol_p)


def summarize_linear_solver_failure(exc):
    """Extract the useful PETSc/DOLFIN reason from a failed linear solve."""
    lines = [line.strip(" *") for line in str(exc).splitlines() if line.strip(" *")]
    for line in lines:
        if line.startswith("Reason:"):
            return line
    return lines[-1] if lines else exc.__class__.__name__


# ===============================================================
# IPCS forward solve
# ===============================================================
def solve_forward_ipcs(solve_label=None, allow_best_without_acceptance=False):
    """Run IPCS to a steady state and store the result in w_fwd."""
    global ipcs_solve_counter
    global save_ipcs_residual_plots
    max_it  = int(globals().get("FORWARD_IPCS_MAX_ITERS", 200))
    rtol_u  = float(globals().get("FORWARD_IPCS_VELOCITY_RTOL", 1.0e-3))
    rtol_p  = float(globals().get("FORWARD_IPCS_PRESSURE_RTOL", 2.0e-2))
    rtol_u, rtol_p = ipcs_tolerances_for_label(solve_label, rtol_u, rtol_p)
    omega_u_base = float(globals().get("FORWARD_IPCS_VEL_RELAXATION", 0.5))
    omega_p_base = float(globals().get("FORWARD_IPCS_P_RELAXATION", 0.2))
    min_omega_u = float(globals().get("FORWARD_IPCS_MIN_U_RELAXATION", 0.05))
    min_omega_p = float(globals().get("FORWARD_IPCS_MIN_P_RELAXATION", 0.02))
    base_dt = float(globals().get("FORWARD_IPCS_DT", 2.0e-4))
    max_restarts = max(0, int(globals().get("FORWARD_IPCS_MAX_RESTARTS", 0)))
    dt_reduction = float(globals().get("FORWARD_IPCS_DT_REDUCTION_FACTOR", 0.5))
    relax_reduction = float(globals().get("FORWARD_IPCS_RELAXATION_REDUCTION_FACTOR", 0.7))
    error_on_nonconvergence = bool(globals().get("FORWARD_IPCS_ERROR_ON_NONCONVERGENCE", True))
    default_accept_best_score = float(globals().get("FORWARD_IPCS_ACCEPT_BEST_SCORE", 1.0))
    accept_best_score = ipcs_accept_best_score_for_label(solve_label, default_accept_best_score)
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
        attempt_failure = None

        solver_log(
            "      [IPCS] attempt {}/{}: dt={:.2e}, omega_u={:.2f}, omega_p={:.2f}".format(
                attempt_idx + 1, max_restarts + 1, current_dt, current_omega_u, current_omega_p,
            )
        )

        for step_idx in range(1, max_it + 1):
            substep_name = "tentative velocity"
            try:
                solve(a_pc_u    == L_pc_u,    u_pc_star, bcu_pc, solver_parameters=ksp_u)
                substep_name = "pressure correction"
                solve(a_pc_p    == L_pc_p,    p_pc_new,  bcp_pc, solver_parameters=ksp_p)
                substep_name = "velocity correction"
                solve(a_pc_corr == L_pc_corr, u_pc_new,  bcu_pc, solver_parameters=ksp_u)
            except RuntimeError as exc:
                attempt_failure = "{} solve failed at step {:03d}: {}".format(
                    substep_name, step_idx, summarize_linear_solver_failure(exc)
                )
                solver_log(
                    "      [IPCS] attempt {}/{} {}".format(
                        attempt_idx + 1, max_restarts + 1, attempt_failure,
                    )
                )
                break

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

        summary = (
            "attempt {}: dt={:.2e}, best@{:03d} du={:.2e} (target {:.2e}), "
            "dp={:.2e} (target {:.2e}); final du={:.2e}, dp={:.2e}"
        ).format(
            attempt_idx + 1, current_dt, attempt_best_step_idx, attempt_best_du, rtol_u,
            attempt_best_dp, rtol_p, du_rel, dp_rel,
        )
        if attempt_failure is not None:
            summary = "{}; {}".format(summary, attempt_failure)
        attempt_summaries.append(summary)
        if converged:
            assign(w_fwd.sub(0), u_pc_old)
            assign(w_fwd.sub(1), p_pc_old)
            dt_pc.assign(base_dt)
            return float(du_rel), float(dp_rel)

        # Continue the adaptive attempt ladder from the best iterate seen so far.
        assign(u_start, attempt_best_u)
        assign(p_start, attempt_best_p)

        if attempt_failure is None:
            solver_log(
                "      [IPCS] attempt {}/{} reached the {}-step limit without meeting tolerances: "
                "du={:.2e} (target {:.2e}), dp={:.2e} (target {:.2e})".format(
                    attempt_idx + 1, max_restarts + 1, max_it, du_rel, rtol_u, dp_rel, rtol_p,
                )
            )
        else:
            if attempt_idx < max_restarts:
                solver_log(
                    "      [IPCS] attempt {}/{} will retry from the best stable iterate with reduced dt/relaxation".format(
                        attempt_idx + 1, max_restarts + 1,
                    )
                )
            else:
                solver_log(
                    "      [IPCS] attempt {}/{} ended after a linear solver failure; no retries remain".format(
                        attempt_idx + 1, max_restarts + 1,
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
    if best_score <= accept_best_score:
        root_print(
            "Warning: {} Accepting best iterate because normalized residual score "
            "{:.2f} <= acceptance limit {:.2f}.".format(
                message, best_score, accept_best_score,
            )
        )
        return float(best_du), float(best_dp)
    if error_on_nonconvergence and not allow_best_without_acceptance:
        raise RuntimeError(message)
    root_print("Warning: {}".format(message))
    return float(best_du), float(best_dp)


# ===============================================================
# Adjoint solve
# ===============================================================
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
results_root = os.path.join(THIS_DIR, globals().get("RESULTS_ROOT_NAME", "Results_Frozen/Results_Frozen"))
rho_dir = os.path.join(results_root, "rho")
rho_p_dir = os.path.join(results_root, "rho_projected")
u_dir = os.path.join(results_root, "u")
u_magnitude_dir = os.path.join(results_root, "u_magnitude")
p_dir = os.path.join(results_root, "p")
nu_tilde_dir = os.path.join(results_root, "nu_tilde")
j_d_dir = os.path.join(results_root, "J_D")
viscous_dissipation_dir = os.path.join(results_root, "viscous_dissipation")
nu_tilde_raw_dir = os.path.join(results_root, "nu_tilde_raw_sa_solve")
nu_tilde_preclip_dir = os.path.join(results_root, "nu_tilde_relaxed_preclip")
nu_tilde_clip_floor_dir = os.path.join(results_root, "nu_tilde_floor_clip_mask")
nu_tilde_clip_ceiling_dir = os.path.join(results_root, "nu_tilde_ceiling_clip_mask")
nu_tilde_bound_floor_dir = os.path.join(results_root, "nu_tilde_floor_bound_mask")
nu_tilde_bound_ceiling_dir = os.path.join(results_root, "nu_tilde_ceiling_bound_mask")
df0dx_dir = os.path.join(results_root, "df0dx")
df0dx_normalized_dir = os.path.join(results_root, "df0dx_normalized")
df0dx_centered_dir = os.path.join(results_root, "df0dx_centered")
design_dir = os.path.join(results_root, "design")
ipcs_residual_dir = os.path.join(results_root, "ipcs_residuals")
paper_data_dir = os.path.join(results_root, "paper_data")
save_dilgen_paper_data = bool(globals().get("SAVE_DILGEN_PAPER_DATA", False))
save_cellwise_dissipation_fields = config_truthy(
    globals().get("SAVE_CELLWISE_DISSIPATION_FIELDS", True)
)
save_ipcs_residual_plots = (
    bool(globals().get("SAVE_IPCS_RESIDUAL_PLOTS", False))
    and (
        final_flow_uses_ipcs()
        or FORWARD_PICARD_FLOW_SOLVER == "ipcs"
        or bool(globals().get("FORWARD_SNES_WARM_START_WITH_IPCS", False))
    )
)
save_ipcs_residual_svgs = bool(globals().get("SAVE_IPCS_RESIDUAL_SVGS", False))
ipcs_solve_counter = 0
snes_ipcs_warm_start_done = False

checkpoint_path = optimization_checkpoint_path(results_root)
resume_requested = resume_optimization_requested(globals())
resume_checkpoint = load_optimization_checkpoint(checkpoint_path, COMM) if resume_requested else None
resume_from_checkpoint = resume_checkpoint is not None
resume_vtk_iteration = (
    checkpoint_scalar(resume_checkpoint, "iter_count", scalar_type=int)
    if resume_from_checkpoint
    else None
)

output_dirs = [
    results_root,
    rho_dir,
    rho_p_dir,
    u_dir,
    p_dir,
    nu_tilde_dir,
    df0dx_centered_dir,
    design_dir,
]
if save_cellwise_dissipation_fields:
    output_dirs.extend([j_d_dir, viscous_dissipation_dir])
if save_dilgen_paper_data:
    output_dirs.extend([u_magnitude_dir, df0dx_dir, df0dx_normalized_dir, paper_data_dir])
if save_sa_clipping_diagnostics:
    output_dirs.extend([
        nu_tilde_raw_dir,
        nu_tilde_preclip_dir,
        nu_tilde_clip_floor_dir,
        nu_tilde_clip_ceiling_dir,
        nu_tilde_bound_floor_dir,
        nu_tilde_bound_ceiling_dir,
    ])
if save_ipcs_residual_plots:
    output_dirs.append(ipcs_residual_dir)

if resume_from_checkpoint:
    root_print("Resuming optimization from checkpoint {}".format(checkpoint_path))
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
    for output_dir in output_dirs[1:]:
        ensure_clean_dir(output_dir)


def output_series_path(output_path):
    return resume_vtk_series_path(output_path, resume_vtk_iteration)


rho_out = ResilientVTKFile(output_series_path(os.path.join(rho_dir, "plot_rho.pvd")), COMM)
rhop_out = ResilientVTKFile(output_series_path(os.path.join(rho_p_dir, "plot_rho_projected.pvd")), COMM)
u_out = ResilientVTKFile(output_series_path(os.path.join(u_dir, "plot_u.pvd")), COMM)
if save_dilgen_paper_data:
    u_magnitude_out = ResilientVTKFile(output_series_path(os.path.join(u_magnitude_dir, "plot_u_magnitude.pvd")), COMM)
p_out = ResilientVTKFile(output_series_path(os.path.join(p_dir, "plot_p.pvd")), COMM)
nu_tilde_out = ResilientVTKFile(output_series_path(os.path.join(nu_tilde_dir, "plot_nu_tilde.pvd")), COMM)
if save_cellwise_dissipation_fields:
    j_d_out = ResilientVTKFile(output_series_path(os.path.join(j_d_dir, "J_D.pvd")), COMM)
    viscous_dissipation_out = ResilientVTKFile(
        output_series_path(os.path.join(viscous_dissipation_dir, "viscous_dissipation.pvd")),
        COMM,
    )
if save_sa_clipping_diagnostics:
    nu_tilde_raw_out = ResilientVTKFile(
        output_series_path(os.path.join(nu_tilde_raw_dir, "plot_nu_tilde_raw_sa_solve.pvd")), COMM
    )
    nu_tilde_preclip_out = ResilientVTKFile(
        output_series_path(os.path.join(nu_tilde_preclip_dir, "plot_nu_tilde_relaxed_preclip.pvd")), COMM
    )
    nu_tilde_floor_clip_mask_out = ResilientVTKFile(
        output_series_path(os.path.join(nu_tilde_clip_floor_dir, "plot_nu_tilde_floor_clip_mask.pvd")), COMM
    )
    nu_tilde_ceiling_clip_mask_out = ResilientVTKFile(
        output_series_path(os.path.join(nu_tilde_clip_ceiling_dir, "plot_nu_tilde_ceiling_clip_mask.pvd")), COMM
    )
    nu_tilde_floor_bound_mask_out = ResilientVTKFile(
        output_series_path(os.path.join(nu_tilde_bound_floor_dir, "plot_nu_tilde_floor_bound_mask.pvd")), COMM
    )
    nu_tilde_ceiling_bound_mask_out = ResilientVTKFile(
        output_series_path(os.path.join(nu_tilde_bound_ceiling_dir, "plot_nu_tilde_ceiling_bound_mask.pvd")), COMM
    )
if save_dilgen_paper_data:
    df0dx_out = ResilientVTKFile(output_series_path(os.path.join(df0dx_dir, "plot_df0dx.pvd")), COMM)
    df0dx_normalized_out = ResilientVTKFile(
        output_series_path(os.path.join(df0dx_normalized_dir, "plot_df0dx_normalized.pvd")),
        COMM,
    )
df0dx_centered_out = ResilientVTKFile(output_series_path(os.path.join(df0dx_centered_dir, "plot_df0dx_centered.pvd")), COMM)

log_path = os.path.join(results_root, "OptimizationLog.txt")
df0dx_log_path = os.path.join(results_root, "Df0dxLog.txt")
sensitivity_check_log_path = os.path.join(results_root, "SensitivityCheckLog.tsv")
sensitivity_verification_table_path = os.path.join(results_root, "SensitivityVerificationTable.tsv")
taylor_check_log_path = os.path.join(results_root, "TaylorCheckLog.tsv")
sa_clipping_log_path = os.path.join(results_root, "NuTildeClippingLog.txt")
optimization_log_quantity_columns = (
    "ViscousDissipation_nondesign_W_per_m",
    "ViscousDissipation_design_W_per_m",
    "dP_nondesign_Pa",
    "dP_design_Pa",
)
optimization_log_extra_columns = (
    ("VolumeConstraint_Dilgen", "Objective_Normalized_Dilgen")
    if bool(globals().get("LOG_DILGEN_FIG8_COLUMNS", False))
    else ()
)
dilgen_volume_constraint_log_path = os.path.join(results_root, "VolumeConstraint_Dilgen.txt")
dilgen_objective_normalized_log_path = os.path.join(results_root, "Objective_Normalized_Dilgen.txt")


def initialize_dilgen_fig8_metric_logs():
    if not optimization_log_extra_columns:
        return
    if IS_ROOT:
        with open(dilgen_volume_constraint_log_path, "w") as handle:
            handle.write("VolumeConstraint_Dilgen\tGlobalIter\n")
        with open(dilgen_objective_normalized_log_path, "w") as handle:
            handle.write("Objective_Normalized_Dilgen\tGlobalIter\n")
    MPI.barrier(COMM)


if not resume_from_checkpoint:
    initialize_optimization_log(
        log_path,
        pressure_drop_columns=optimization_log_quantity_columns,
        objective_column=objective_log_column,
        extra_columns=optimization_log_extra_columns,
    )
    initialize_dilgen_fig8_metric_logs()
    initialize_df0dx_log(df0dx_log_path)
    initialize_sensitivity_check_log(sensitivity_check_log_path)
    initialize_sensitivity_verification_table(sensitivity_verification_table_path)
    initialize_taylor_check_log(taylor_check_log_path)
    if save_sa_clipping_diagnostics:
        initialize_sa_clipping_log(sa_clipping_log_path)

# ---------------------------------------------------------------
# MMA setup.
# ---------------------------------------------------------------
initial_density = float(globals().get("INITIAL_DENSITY_VALUE", VOL_FRAC))
assign(rho, interpolate(Constant(initial_density), DensitySpace))
enforce_density_bounds_inplace(rho)

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

mmma = 1 + len(mass_flow_constraints)
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


def dilgen_objective_normalization_scale():
    """Return the dimensional scale used for the Dilgen Fig. 8 comparison column."""
    if "DILGEN_OBJECTIVE_NORMALIZATION_SCALE" in globals():
        return float(globals()["DILGEN_OBJECTIVE_NORMALIZATION_SCALE"])

    reference_density = float(globals().get("DILGEN_OBJECTIVE_REFERENCE_DENSITY", RHO_FLUID_VALUE))
    reference_velocity = float(globals().get("DILGEN_OBJECTIVE_REFERENCE_VELOCITY", U_BULK_INLET))
    reference_length = float(
        globals().get(
            "DILGEN_OBJECTIVE_REFERENCE_LENGTH",
            globals().get("H", globals().get("INLET_HALF_HEIGHT", 1.0)),
        )
    )
    if reference_density <= 0.0 or reference_velocity <= 0.0 or reference_length <= 0.0:
        raise ValueError("Dilgen objective normalization needs positive rho, U, and length scales.")

    # Eq. (42) is logged as 2D unit-depth power. For Dilgen's plotted
    # nondimensional objective, scale by rho * U_b^3 * V_design / H.
    return reference_density * reference_velocity**3.0 * float(volume) / reference_length


def dilgen_optimization_log_extra_values(objective_value, volume_fraction):
    if not optimization_log_extra_columns:
        return ()

    volume_target = float(VOL_FRAC)
    if volume_target <= 0.0:
        raise ValueError("Dilgen volume-constraint logging needs VOL_FRAC > 0.")

    objective_scale = dilgen_objective_normalization_scale()
    if objective_scale <= 0.0:
        raise ValueError("Dilgen objective normalization scale must be positive.")

    volume_constraint = float(volume_fraction) / volume_target - 1.0
    objective_normalized = float(objective_value) / objective_scale
    return volume_constraint, objective_normalized


def append_dilgen_fig8_metric_logs(metric_values, global_iter):
    if not optimization_log_extra_columns:
        return

    volume_constraint, objective_normalized = metric_values
    if IS_ROOT:
        with open(dilgen_volume_constraint_log_path, "a") as handle:
            handle.write("{:.16e}\t{:d}\n".format(float(volume_constraint), int(global_iter)))
        with open(dilgen_objective_normalized_log_path, "a") as handle:
            handle.write("{:.16e}\t{:d}\n".format(float(objective_normalized), int(global_iter)))


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
    if not bool(globals().get("INITIAL_DENSITY_MATCH_FILTERED_VOLUME", True)):
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


initialize_design_filter_normalization()
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


def stage_schedule_value(schedule_name, fallback_value, stage_idx):
    raw_schedule = globals().get(schedule_name, None)
    if raw_schedule is None:
        return float(fallback_value)
    if isinstance(raw_schedule, np.ndarray):
        values = raw_schedule.tolist()
    elif isinstance(raw_schedule, (list, tuple)):
        values = list(raw_schedule)
    else:
        values = [raw_schedule]
    if not values:
        return float(fallback_value)
    return float(values[min(int(stage_idx), len(values) - 1)])


def update_stage_penalty_parameters(stage_idx):
    """Advance wall-distance and nu_tilde solid penalties with continuation."""
    wall_alpha_now = stage_schedule_value(
        "SA_WALL_PENALTY_ALPHA_SCHEDULE",
        sa_wall_penalty_alpha_base,
        stage_idx,
    )
    nu_tilde_alpha_now = stage_schedule_value(
        "SA_NU_TILDE_PENALTY_ALPHA_SCHEDULE",
        sa_nu_tilde_penalty_alpha_base,
        stage_idx,
    )
    sa_wall_penalty_alpha.assign(wall_alpha_now)
    sa_nu_tilde_penalty_alpha.assign(nu_tilde_alpha_now)
    return wall_alpha_now, nu_tilde_alpha_now


def final_stage_metric_target_status(stage_idx, viscous_dissipation_design_value, pressure_drop_design_value):
    """Return configured final-stage physical metric target status for logging."""
    if (
        not bool(globals().get("REPORT_FINAL_METRIC_TARGETS", False))
        or int(stage_idx) != len(Q_PENAL_SCHEDULE) - 1
    ):
        return True, ""

    checks = []
    pressure_drop_target = globals().get("FINAL_DESIGN_PRESSURE_DROP_TARGET_PA", None)
    if pressure_drop_target is not None:
        checks.append(("dP_design", float(pressure_drop_design_value), float(pressure_drop_target), "Pa"))

    viscous_dissipation_target = globals().get(
        "FINAL_DESIGN_VISCOUS_DISSIPATION_TARGET_W_PER_M",
        None,
    )
    if viscous_dissipation_target is not None:
        checks.append((
            "ViscousDissipation_design",
            float(viscous_dissipation_design_value),
            float(viscous_dissipation_target),
            "W/m",
        ))

    if not checks:
        return True, ""

    target_status = "targets " + " ".join(
        "{}={:.4e}/{:.4e} {}".format(label, value, target, unit)
        for label, value, target, unit in checks
    )
    return all(value <= target for _label, value, target, _unit in checks), target_status

# ===============================================================
# Continuation and MMA optimization loop.
# Each iteration: filter design -> update wall distance -> Picard flow/SA
# updates -> strict final flow -> adjoint -> filtered sensitivities -> MMA.
# ===============================================================
optimization_start_time = time.perf_counter()
for stage_idx, q_val in enumerate(Q_PENAL_SCHEDULE):
    if stage_idx < resume_stage_idx:
        continue

    beta_val = float(BETA_PROJ_SCHEDULE[stage_idx])
    BETA_PROJ.assign(beta_val)
    move_limit_now = MOVE_LIMIT_SCHEDULE[stage_idx]
    max_iters_now = MAX_INNER_ITERATIONS_SCHEDULE[stage_idx]
    q_penal.assign(q_val)
    wall_alpha_now, nu_tilde_alpha_now = update_stage_penalty_parameters(stage_idx)
    inner_count = resume_inner_count if stage_idx == resume_stage_idx else 0
    convergence_history = resume_convergence_history if stage_idx == resume_stage_idx else 0
    objective_converged = False
    last_metric_targets_met = True
    last_metric_target_status = ""
    root_print("Starting continuation stage {}/{}: q = {:.3f}, beta = {:.2f}, move = {:.4f}, wall_alpha = {:.3e}, nu_tilde_alpha = {:.3e}".format(
        stage_idx + 1, len(Q_PENAL_SCHEDULE), q_val, beta_val, move_limit_now,
        wall_alpha_now, nu_tilde_alpha_now,
    ))

    while inner_count < max_iters_now and not objective_converged:
        iteration_start_time = time.perf_counter()
        root_print("--- Stage {}/{} | iter {:03d} (global {:03d}) ---".format(
            stage_idx + 1, len(Q_PENAL_SCHEDULE), inner_count, iter_count,
        ))

        # --- Filtering and wall distance ---
        solver_log("  [Filter] design density")
        rho_f = pde_filter_design_density(rho, rho_f)
        rho_proj_plot.vector()[:] = project(rho_effective, DensitySpace).vector()[:]
        solver_log("  [Wall distance] update")
        update_wall_distance_field()

        rho_out << rho
        rhop_out << rho_proj_plot

        if iter_count == 0:
            solver_log("  [Warm start] Stokes-Brinkman and initial SA field")
            initialize_forward_guess_with_stokes()

        # --- Forward solve: configured flow solver + frozen SA Picard updates ---
        root_print("  [Forward solve]")
        picard_steps = max(1, int(globals().get("PICARD_STEPS", 1)))
        last_sa_clipping_stats = None
        for picard_idx in range(picard_steps):
            solver_log("    [Picard {}/{}] flow".format(picard_idx + 1, picard_steps))
            solve_forward_picard("stage{:02d}_iter{:03d}_picard{:02d}".format(
                stage_idx + 1, inner_count, picard_idx + 1,
            ))
            velocity_for_sa = w_fwd.sub(0, deepcopy=True)
            sa_model.construct_forms(velocity_for_sa)
            sa_substeps_now = sa_pseudo_time_steps if sa_pseudo_time_stabilization else 1
            for sa_substep_idx in range(sa_substeps_now):
                if sa_substeps_now > 1:
                    solver_log(
                        "    [Picard {}/{}] SA transport substep {}/{}".format(
                            picard_idx + 1,
                            picard_steps,
                            sa_substep_idx + 1,
                            sa_substeps_now,
                        )
                    )
                else:
                    solver_log("    [Picard {}/{}] SA transport".format(picard_idx + 1, picard_steps))
                sa_model.solve_turbulence_model()
                sa_model.update_variables(
                    relaxation=float(globals().get("TURBULENCE_RELAXATION", 1.0))
                )
                nu_tilde_frozen.assign(sa_model.nu_tilde0)
                enforce_scalar_bounds_inplace(nu_tilde_frozen, sa_nu_tilde_floor, sa_nu_tilde_ceiling)
                if picard_idx == picard_steps - 1 and sa_substep_idx == sa_substeps_now - 1:
                    last_sa_clipping_stats = update_sa_clipping_diagnostics(
                        stage_idx,
                        q_val,
                        beta_val,
                        inner_count,
                        iter_count,
                        picard_idx,
                    )
                sa_model.nu_tilde0.assign(nu_tilde_frozen)
                sa_model.nu_tilde1.assign(nu_tilde_frozen)

        final_flow_solver_label = format_forward_flow_solver_name(FORWARD_FLOW_SOLVER)
        solver_log("    [Final flow] {} with updated turbulent viscosity".format(final_flow_solver_label))
        final_flow_du_ipcs, final_flow_dp_ipcs = solve_forward("stage{:02d}_iter{:03d}_final".format(
            stage_idx + 1, inner_count,
        ))

        # --- Adjoint solve ---
        root_print("  [Adjoint solve]")
        solve_adjoint(objective_adjoint_form)

        u_out << w_fwd.sub(0)
        p_out << w_fwd.sub(1)
        nu_tilde_out << nu_tilde_frozen
        velocity_magnitude_plot = None
        if save_dilgen_paper_data:
            velocity_magnitude_plot = build_velocity_magnitude_field(w_fwd.sub(0, deepcopy=True), DensitySpace)
            u_magnitude_out << velocity_magnitude_plot
        if save_sa_clipping_diagnostics:
            nu_tilde_raw_out << nu_tilde_raw_diagnostic
            nu_tilde_preclip_out << nu_tilde_preclip_diagnostic
            nu_tilde_floor_clip_mask_out << nu_tilde_floor_clip_mask
            nu_tilde_ceiling_clip_mask_out << nu_tilde_ceiling_clip_mask
            nu_tilde_floor_bound_mask_out << nu_tilde_floor_bound_mask
            nu_tilde_ceiling_bound_mask_out << nu_tilde_ceiling_bound_mask
            if last_sa_clipping_stats is not None:
                append_sa_clipping_log_entry(sa_clipping_log_path, last_sa_clipping_stats)

        f0val = assemble(ObjFunctional)
        if objective_scale_reference is None:
            initial_objective_reference = float(f0val)
            if abs(initial_objective_reference) <= float(globals().get("OBJECTIVE_SCALE_FLOOR", 1.0e-30)):
                raise ValueError("Initial objective is too close to zero for Dilgen normalized line outputs.")
            objective_scale_reference = max(abs(float(f0val)), float(globals().get("OBJECTIVE_SCALE_FLOOR", 1.0e-30)))
            root_print("MMA objective scale: initial objective {:.6e}.".format(objective_scale_reference))
        # MMA sees scaled objective values; logs keep physical values.
        f0val_mma = float(f0val) / objective_scale_reference
        viscous_dissipation_nondesign_now = assemble(ViscousDissipationNondesignFunctional)
        viscous_dissipation_design_now = assemble(ViscousDissipationFunctional)
        if save_cellwise_dissipation_fields:
            j_d_out << assemble_cell_integral_field(
                J_D_integrand,
                j_d_per_cell_plot,
                "J_D",
            )
            viscous_dissipation_out << assemble_cell_integral_field(
                viscous_dissipation_integrand,
                viscous_dissipation_per_cell_plot,
                "viscous_dissipation",
            )
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
        metric_targets_met, metric_target_status = final_stage_metric_target_status(
            stage_idx,
            viscous_dissipation_design_now,
            pressure_drop_design_now,
        )
        last_metric_targets_met = metric_targets_met
        last_metric_target_status = metric_target_status

        if obj_conv < OBJECTIVE_CONVERGENCE_TOL:
            convergence_history += 1
            if convergence_history >= OBJECTIVE_STREAK_TO_STOP:
                objective_converged = True
        else:
            convergence_history = 0
        previous_objective = f0val

        # --- Sensitivities and constraints ---
        unfiltered_gradient.vector()[:] = assemble(objective_ddx)[:]
        filtered_gradient = pde_filter_design_gradient(unfiltered_gradient, filtered_gradient)
        np.savetxt(os.path.join(design_dir, "rho_{:03}.txt".format(iter_count)), rho.vector()[:])
        if save_dilgen_paper_data:
            df0dx_plot = copy_scalar_field(filtered_gradient, "df0dx")
            df0dx_out << df0dx_plot
            df0dx_normalized_plot = build_cell_area_normalized_field(
                filtered_gradient,
                "df0dx_normalized",
            )
            df0dx_normalized_out << df0dx_normalized_plot

        fval[0, 0] = assemble(vol_constraint) / volume
        unfiltered_s_vol.vector()[:] = assemble(sensitivities_vol_constraint)[:]
        filtered_s_vol = pde_filter_design_gradient(unfiltered_s_vol, filtered_s_vol)
        vol_fraction_now = assemble(VolumeRegion * rho_effective * dx) / volume
        vol_residual_now = float(fval[0, 0])

        # Store only active-design sensitivities for MMA.
        objective_gradient_active_unscaled = filtered_gradient.vector().get_local()[ActiveDV].copy()
        df0dx[:, 0] = objective_gradient_active_unscaled / objective_scale_reference
        dfdx[0, :] = filtered_s_vol.vector().get_local()[ActiveDV] / volume
        df0dx_centered = df0dx[:, 0] - np.mean(df0dx[:, 0])
        df0dx_centered_values = np.zeros_like(filtered_gradient.vector().get_local())
        df0dx_centered_values[ActiveDV] = df0dx_centered
        df0dx_centered_plot.vector().set_local(df0dx_centered_values)
        df0dx_centered_plot.vector().apply("insert")
        df0dx_centered_plot.rename("df0dx_centered", "df0dx_centered")
        df0dx_centered_out << df0dx_centered_plot
        if bool(globals().get("SAVE_DF0DX_VECTOR", True)):
            np.savetxt(os.path.join(design_dir, "df0dx_{:03}.txt".format(iter_count)), df0dx[:, 0])
            if save_dilgen_paper_data:
                np.savetxt(
                    os.path.join(design_dir, "df0dx_unscaled_{:03}.txt".format(iter_count)),
                    objective_gradient_active_unscaled,
                )
            np.savetxt(os.path.join(design_dir, "df0dx_centered_{:03}.txt".format(iter_count)), df0dx_centered)
        if save_dilgen_paper_data:
            df0dx_objective_normalized_plot = build_objective_normalized_field(
                filtered_gradient,
                initial_objective_reference,
                "df0dx_objective_normalized",
            )
            write_dilgen_paper_data(
                paper_data_dir,
                iter_count,
                velocity_magnitude_plot,
                df0dx_plot,
                DensitySpace,
                ActiveDV,
                globals(),
                COMM,
                include_g_state=False,
                df0dx_objective_normalized_field=df0dx_objective_normalized_plot,
            )
        append_df0dx_log_entry(
            df0dx_log_path,
            stage_idx + 1,
            q_val,
            beta_val,
            inner_count,
            iter_count,
            df0dx[:, 0],
        )
        finite_difference_check_due = (
            bool(globals().get("RUN_FINITE_DIFFERENCE_CHECKS", False))
            and int(iter_count) in finite_difference_check_iterations()
        )
        run_finite_difference_checks(
            stage_idx + 1,
            inner_count,
            iter_count,
            f0val,
            objective_gradient_active_unscaled,
        )
        if finite_difference_check_due and bool(globals().get("STOP_AFTER_FINITE_DIFFERENCE_CHECKS", False)):
            root_print("Stopping after finite-difference sensitivity verification as requested by config.")
            raise SystemExit(0)

        mass_flow_status = []
        mass_flow_status_markers = set()
        for constraint_idx, constraint_spec in enumerate(mass_flow_constraints, start=1):
            solve_adjoint(constraint_spec["adjoint_form"])
            fval[constraint_idx, 0] = (
                assemble(constraint_spec["functional"]) + constraint_spec["offset"]
            )
            unfiltered_constraint_gradient.vector()[:] = assemble(constraint_spec["gradient_form"])[:]
            filtered_constraint_gradient = pde_filter_design_gradient(
                unfiltered_constraint_gradient, filtered_constraint_gradient
            )
            dfdx[constraint_idx, :] = filtered_constraint_gradient.vector().get_local()[ActiveDV]

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
            f0val_mma, df0dx, fval, dfdx, low, upp, a0, a, c, d, move_limit_now,
        )

        xold2 = xold1.copy()
        xold1 = xval.copy()
        xval = xmma.copy()
        active_rho_values = np.clip(
            xmma[:, 0].copy(), active_density_lower_values, active_density_upper_values,
        )
        xval[:, 0] = active_rho_values
        rho_values = np.clip(rho.vector().get_local(), density_lower_values, density_upper_values)
        rho_values[ActiveDV] = active_rho_values
        rho.vector().set_local(rho_values)
        rho.vector().apply("insert")

        dilgen_fig8_metric_values = dilgen_optimization_log_extra_values(f0val, vol_fraction_now)
        append_dilgen_fig8_metric_logs(dilgen_fig8_metric_values, iter_count)

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
            extra_values=dilgen_fig8_metric_values,
            extra_columns=optimization_log_extra_columns,
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

        constraint_status_text = ""
        if mass_flow_status:
            constraint_status_text = " " + " ".join(mass_flow_status)
        if metric_target_status:
            constraint_status_text += " " + metric_target_status

        iteration_elapsed = time.perf_counter() - iteration_start_time
        optimization_elapsed = time.perf_counter() - optimization_start_time
        root_print("q={:.3f} beta={:.2f} move={:.3f} iter={:03d} iter_time={:.1f}s elapsed={:.1f}s {}={:.4e}{} ViscousDissipation_nondesign={:.4e} W/m ViscousDissipation_design={:.4e} W/m dP_nondesign={:.4e} Pa dP_design={:.4e} Pa conv={:.3e} vol={:.4f} streak={}/{}{}".format(
            q_val, float(BETA_PROJ.values()[0]), move_limit_now,
            inner_count, iteration_elapsed, optimization_elapsed,
            objective_console_label, f0val, objective_console_unit,
            viscous_dissipation_nondesign_now, viscous_dissipation_design_now,
            pressure_drop_nondesign_now, pressure_drop_design_now,
            obj_conv, vol_fraction_now,
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
        if last_metric_target_status and not last_metric_targets_met:
            root_print(
                "Warning: final metric targets were not met at the stage iteration cap: {}.".format(
                    last_metric_target_status,
                )
            )

root_print("Writing final post-update density output.")
rho_f = pde_filter_design_density(rho, rho_f)
rho_proj_plot.vector()[:] = project(rho_effective, DensitySpace).vector()[:]
rho_out << rho
rhop_out << rho_proj_plot
np.savetxt(os.path.join(design_dir, "rho_{:03}.txt".format(iter_count)), rho.vector()[:])

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
