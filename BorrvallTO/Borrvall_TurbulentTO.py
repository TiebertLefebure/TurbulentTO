from dolfin import *
import argparse
import importlib
import inspect
import numpy as np
import os
import shutil
import sys
from time import localtime, strftime
from ufl import tanh

from mma import mmasub

# --------------------------------------------------------------------------------
# Topology Optimization solver uses steady (!) Spalart-Allmaras turbulence model
# --------------------------------------------------------------------------------

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
TURB_MODELS_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", "TurbulenceModels"))
if TURB_MODELS_DIR not in sys.path:
    sys.path.insert(0, TURB_MODELS_DIR)
from TurbulenceModel_SpalartAllmaras_TO import (
    SpalartAllmarasSteadyState,
    sa_turbulent_viscosity,
)


def _normalize_module_name(module_name):
    normalized = module_name.strip()
    if normalized.endswith(".py"):
        normalized = normalized[:-3]
    normalized = normalized.replace(os.sep, ".")
    return normalized


def _load_config_module():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "--config",
        required=True,
        help="Python module name for solver configuration (required).",
    )
    args, _unknown = parser.parse_known_args()
    module_name = _normalize_module_name(args.config)
    if not module_name:
        raise ValueError("Empty config module name is not allowed.")
    return module_name, importlib.import_module(module_name)


def _float_scalar(value):
    if hasattr(value, "values"):
        values = value.values()
        if len(values) == 1:
            return float(values[0])
    return float(value)


def _as_list(value):
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _expand_to_match(values, target_size, label):
    if len(values) == target_size:
        return values
    if len(values) == 1 and target_size > 1:
        return values * target_size
    raise ValueError(
        "{} count ({}) must match marker count ({})".format(label, len(values), target_size)
    )


CONFIG_MODULE_NAME, CONFIG = _load_config_module()
for _name, _value in vars(CONFIG).items():
    if not _name.startswith("_"):
        globals()[_name] = _value
print("Using config module: {}".format(CONFIG_MODULE_NAME))

BETA_PROJ = Constant(_float_scalar(BETA_PROJ_VALUE))

# Brinkman penalization constants (same style as diffuser script)
mu_fluid = Constant(MU_FLUID_VALUE)
rho_fluid = Constant(RHO_FLUID_VALUE)
alpha_fluid = Constant(2.5 * MU_FLUID_VALUE / 100.0**2.0)
alpha_solid = Constant(2.5 * MU_FLUID_VALUE / 0.01**2.0)
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
    return alpha_solid + (alpha_fluid - alpha_solid) * brinkman_density * (1 + q_penal) / (brinkman_density + q_penal)


def create_design_mesh_from_config():
    mesh_builder = globals().get("create_design_mesh")
    if callable(mesh_builder):
        return mesh_builder()

    x_min = float(globals().get("DOMAIN_X_MIN", 0.0))
    y_min = float(globals().get("DOMAIN_Y_MIN", 0.0))

    x_max_default = globals().get("DOMAIN_X_MAX", globals().get("L"))
    y_max_default = globals().get("DOMAIN_Y_MAX", globals().get("L", x_max_default))
    if x_max_default is None or y_max_default is None:
        raise ValueError("Config must define DOMAIN_X_MAX/DOMAIN_Y_MAX or legacy L.")
    x_max = float(x_max_default)
    y_max = float(y_max_default)

    nx_default = globals().get("NX", globals().get("N"))
    ny_default = globals().get("NY", globals().get("N", nx_default))
    if nx_default is None or ny_default is None:
        raise ValueError("Config must define NX/NY or legacy N.")
    nx = int(nx_default)
    ny = int(ny_default)
    mesh_diagonal = globals().get("MESH_DIAGONAL", "crossed")

    return Mesh(
        RectangleMesh(
            MPI.comm_world,
            Point(x_min, y_min),
            Point(x_max, y_max),
            nx,
            ny,
            mesh_diagonal,
        )
    )


def compute_filter_base_length():
    custom_base = globals().get("FILTER_BASE_LENGTH")
    if custom_base is not None:
        return float(custom_base)

    if "L" in globals() and "N" in globals():
        return float(L) / float(N)

    x_min = float(globals().get("DOMAIN_X_MIN", 0.0))
    y_min = float(globals().get("DOMAIN_Y_MIN", 0.0))
    x_max = float(globals().get("DOMAIN_X_MAX", x_min + 1.0))
    y_max = float(globals().get("DOMAIN_Y_MAX", y_min + 1.0))
    nx = int(globals().get("NX", globals().get("N", 1)))
    ny = int(globals().get("NY", globals().get("N", 1)))
    return min((x_max - x_min) / float(max(nx, 1)), (y_max - y_min) / float(max(ny, 1)))


def compute_legacy_port_extents():
    if not callable(globals().get("compute_port_extents")):
        return None
    required = [
        "L",
        "INLET_TOP_OFFSET",
        "INLET_WIDTH",
        "OUTLET_RIGHT_OFFSET",
        "OUTLET_WIDTH",
    ]
    if any(name not in globals() for name in required):
        return None
    return compute_port_extents(
        L, INLET_TOP_OFFSET, INLET_WIDTH, OUTLET_RIGHT_OFFSET, OUTLET_WIDTH
    )


def build_marked_boundaries(custom_mesh):
    custom_marker = globals().get("mark_boundaries")
    if callable(custom_marker):
        return custom_marker(custom_mesh)

    extents = compute_legacy_port_extents()
    if extents is None:
        raise ValueError(
            "Config must define mark_boundaries(mesh) or legacy pipe-bend extent helpers."
        )
    inlet_y_min, inlet_y_max, outlet_x_min, outlet_x_max = extents
    return mark_pipe_bend_boundaries(
        custom_mesh,
        L,
        TOL,
        inlet_y_min,
        inlet_y_max,
        outlet_x_min,
        outlet_x_max,
    )


def build_velocity_profile_sets_from_config():
    custom_builder = globals().get("build_velocity_profile_sets")
    if callable(custom_builder):
        inlet_profiles, outlet_profiles = custom_builder()
        return _as_list(inlet_profiles), _as_list(outlet_profiles)

    profile_builder = globals().get("build_velocity_profiles")
    if not callable(profile_builder):
        raise ValueError(
            "Config must define build_velocity_profile_sets() or build_velocity_profiles(...)."
        )

    builder_signature = inspect.signature(profile_builder)
    required_positionals = [
        parameter
        for parameter in builder_signature.parameters.values()
        if parameter.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        and parameter.default is inspect.Parameter.empty
    ]

    if len(required_positionals) == 0:
        profile_data = profile_builder()
        if not isinstance(profile_data, (tuple, list)) or len(profile_data) != 2:
            raise ValueError(
                "build_velocity_profiles() must return (inlet_profiles, outlet_profiles)."
            )
        return _as_list(profile_data[0]), _as_list(profile_data[1])

    extents = compute_legacy_port_extents()
    if extents is None:
        raise ValueError(
            "Legacy velocity-profile builder requires compute_port_extents and pipe-bend parameters."
        )
    inlet_y_min, inlet_y_max, outlet_x_min, outlet_x_max = extents
    u_inlet, u_outlet = profile_builder(
        U_MAX_INLET,
        U_MAX_OUTLET,
        inlet_y_min,
        inlet_y_max,
        outlet_x_min,
        outlet_x_max,
        INLET_WIDTH,
        OUTLET_WIDTH,
    )
    return [u_inlet], [u_outlet]


def build_pressure_pin_expression():
    if "PRESSURE_PIN_POINT" in globals():
        pin_x, pin_y = PRESSURE_PIN_POINT
    else:
        pin_x = float(globals().get("DOMAIN_X_MIN", 0.0))
        pin_y = float(globals().get("DOMAIN_Y_MIN", 0.0))
    return "near(x[0], {:.16g}) && near(x[1], {:.16g})".format(float(pin_x), float(pin_y))


def resolve_sa_nu_tilde_inlet_targets(num_inlets):
    if "SA_NU_TILDE_INLETS" in globals():
        inlet_targets = _as_list(SA_NU_TILDE_INLETS)
    elif "SA_NU_TILDE_INLET" in globals():
        inlet_targets = [SA_NU_TILDE_INLET]
    else:
        raise ValueError("Config must define SA_NU_TILDE_INLETS or SA_NU_TILDE_INLET.")
    return _expand_to_match(inlet_targets, num_inlets, "SA inlet nu_tilde targets")


def apply_ramped_boundary_values(
    ramp,
    inlet_profiles,
    outlet_profiles,
    nu_tilde_inlet_constants,
    sa_nu_tilde_targets,
):
    custom_ramp = globals().get("apply_velocity_and_turbulence_ramp")
    if callable(custom_ramp):
        custom_ramp(
            ramp,
            inlet_profiles,
            outlet_profiles,
            nu_tilde_inlet_constants,
            sa_nu_tilde_targets,
        )
        return

    if "U_MAX_INLETS" in globals():
        inlet_targets = _as_list(U_MAX_INLETS)
    elif "U_MAX_INLET" in globals():
        inlet_targets = [U_MAX_INLET]
    else:
        raise ValueError("Config must define U_MAX_INLETS or U_MAX_INLET.")

    if "U_MAX_OUTLETS" in globals():
        outlet_targets = _as_list(U_MAX_OUTLETS)
    elif "U_MAX_OUTLET" in globals():
        outlet_targets = [U_MAX_OUTLET]
    else:
        raise ValueError("Config must define U_MAX_OUTLETS or U_MAX_OUTLET.")

    inlet_targets = _expand_to_match(inlet_targets, len(inlet_profiles), "inlet velocity targets")
    outlet_targets = _expand_to_match(outlet_targets, len(outlet_profiles), "outlet velocity targets")

    for profile, u_target in zip(inlet_profiles, inlet_targets):
        if not hasattr(profile, "u_max"):
            raise ValueError("Inlet profile is missing writable attribute 'u_max' for ramping.")
        profile.u_max = ramp * _float_scalar(u_target)
    for profile, u_target in zip(outlet_profiles, outlet_targets):
        if not hasattr(profile, "u_max"):
            raise ValueError("Outlet profile is missing writable attribute 'u_max' for ramping.")
        profile.u_max = ramp * _float_scalar(u_target)

    for constant_value, nu_target in zip(nu_tilde_inlet_constants, sa_nu_tilde_targets):
        constant_value.assign(ramp * _float_scalar(nu_target))


def sa_positive_viscosity(state_nu_tilde):
    nu_lam = mu_fluid / rho_fluid
    return sa_turbulent_viscosity(
        state_nu_tilde,
        nu_lam,
        smooth_abs_eps=SA_SMOOTH_ABS_EPS,
    )


def ensure_clean_dir(path, comm=MPI.comm_world):
    # Avoid MPI races where multiple ranks delete/create the same folder.
    if MPI.rank(comm) == 0:
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)
    MPI.barrier(comm)


def enforce_scalar_floor(scalar_function, floor_value):
    """Enforce scalar_function >= floor_value in-place."""
    values = scalar_function.vector().get_local()
    values = np.maximum(values, floor_value)
    scalar_function.vector().set_local(values)
    scalar_function.vector().apply("insert")


# -----------------------------------------------------------------------
# IMPORTANT: Calculating the wall-distance function (smoothed Eikonal)
# -----------------------------------------------------------------------

def calculate_distance_field(space, boundaries_data, wall_markers, custom_dx, relaxation=0.01):
    """Compute distance-to-wall field y with y=0 on no-slip walls."""
    wall_bc = [
        DirichletBC(space, Constant(0.0), boundaries_data, marker)
        for marker in _as_list(wall_markers)
    ]

    y = Function(space)
    dy = TrialFunction(space)
    z = TestFunction(space)

    # Linear initialization (Poission problem with wall BC y=0)
    linear_problem = inner(grad(dy), grad(z)) * custom_dx - Constant(1.0) * z * custom_dx
    solve(lhs(linear_problem) == rhs(linear_problem), y, wall_bc)

    # Smoothed Eikonal solve
    F = (
        sqrt(inner(grad(y), grad(y)) + DOLFIN_EPS) * z * custom_dx
        - Constant(1.0) * z * custom_dx
        + Constant(relaxation) * inner(grad(y), grad(z)) * custom_dx
    )
    problem = NonlinearVariationalProblem(F, y, bcs=wall_bc, J=derivative(F, y))
    solver = NonlinearVariationalSolver(problem)
    solver.solve()
    return y


def build_frozen_state_form(state_u, state_p, adj_u, adj_p, rho_eff, custom_dx, frozen_nu_tilde):
    """State form used for frozen-turbulence adjoint/sensitivity."""
    mu_effective_frozen = mu_fluid + rho_fluid * sa_positive_viscosity(frozen_nu_tilde)
    return (
        rho_fluid * inner(dot(state_u, nabla_grad(state_u)), adj_u)
        + mu_effective_frozen * inner(grad(state_u), grad(adj_u))
        + inner(grad(state_p), adj_u)
        + inner(div(state_u), adj_p)
        + alpha(rho_eff) * inner(state_u, adj_u)
    ) * custom_dx


# ------------------------------------------------------------
# Mesh, function spaces, and boundaries
# ------------------------------------------------------------
mesh = create_design_mesh_from_config()

U_h = VectorElement("CG", mesh.ufl_cell(), 2)
P_h = FiniteElement("CG", mesh.ufl_cell(), 1)
T_h = FiniteElement("CG", mesh.ufl_cell(), 1)
A_h = FiniteElement("DG", mesh.ufl_cell(), 0)

FrozenUPSpace = FunctionSpace(mesh, U_h * P_h)
TurbulenceSpace = FunctionSpace(mesh, T_h)

DensitySpace = FunctionSpace(mesh, A_h)

w_adj_frozen = Function(FrozenUPSpace)
w_state_frozen = Function(FrozenUPSpace)

(v_frozen, q_frozen) = split(w_adj_frozen)
(u_frozen_state, p_frozen_state) = split(w_state_frozen)

rho = Function(DensitySpace)
rho_f = Function(DensitySpace)
nu_tilde_frozen = Function(TurbulenceSpace)

rho_proj_plot = Function(DensitySpace)
unfiltered_gradient = Function(DensitySpace)
filtered_gradient = Function(DensitySpace)
unfiltered_s_vol = Function(DensitySpace)
filtered_s_vol = Function(DensitySpace)

mark = MARK
boundaries = build_marked_boundaries(mesh)
wall_markers = _as_list(mark["walls"])
inlet_markers = _as_list(mark["inlet"])
outlet_markers = _as_list(mark["outlet"])

# Avoid FFC auto-estimating an excessively high quadrature degree for SA nonlinear forms.
parameters["form_compiler"]["quadrature_degree"] = QUADRATURE_DEGREE
dx = Measure("dx", domain=mesh, metadata={"quadrature_degree": QUADRATURE_DEGREE})
ds = Measure(
    "ds",
    domain=mesh,
    subdomain_data=boundaries,
    metadata={"quadrature_degree": QUADRATURE_DEGREE},
)
dS = Measure("dS", domain=mesh, metadata={"quadrature_degree": QUADRATURE_DEGREE})

inlet_profiles, outlet_profiles = build_velocity_profile_sets_from_config()
inlet_profiles = _expand_to_match(inlet_profiles, len(inlet_markers), "inlet velocity profiles")
outlet_profiles = _expand_to_match(outlet_profiles, len(outlet_markers), "outlet velocity profiles")

u_noslip = Constant((0.0, 0.0))

sa_nu_tilde_targets = resolve_sa_nu_tilde_inlet_targets(len(inlet_markers))
nu_tilde_inlet_bc_values = [Constant(0.0) for _ in inlet_markers]
nu_tilde_wall_bc_value = Constant(0.0)
bcnt_inlet_turb = [
    DirichletBC(TurbulenceSpace, inlet_bc, boundaries, marker)
    for inlet_bc, marker in zip(nu_tilde_inlet_bc_values, inlet_markers)
]
bcnt_walls_turb = [
    DirichletBC(TurbulenceSpace, nu_tilde_wall_bc_value, boundaries, marker)
    for marker in wall_markers
]
bcn_turbulence = bcnt_inlet_turb + bcnt_walls_turb

bcu_walls_adj_frozen = [
    DirichletBC(FrozenUPSpace.sub(0), u_noslip, boundaries, marker)
    for marker in wall_markers
]
bcu_inlet_adj_frozen = [
    DirichletBC(FrozenUPSpace.sub(0), u_noslip, boundaries, marker)
    for marker in inlet_markers
]
bcu_outlet_adj_frozen = [
    DirichletBC(FrozenUPSpace.sub(0), u_noslip, boundaries, marker)
    for marker in outlet_markers
]
bc_NS_adj_frozen = bcu_walls_adj_frozen + bcu_inlet_adj_frozen + bcu_outlet_adj_frozen
if globals().get("ENABLE_PRESSURE_PIN", True):
    bcp_pin_adj_frozen = DirichletBC(
        FrozenUPSpace.sub(1),
        Constant(0.0),
        build_pressure_pin_expression(),
        "pointwise",
    )
    bc_NS_adj_frozen.append(bcp_pin_adj_frozen)

bcu_walls_frozen = [
    DirichletBC(FrozenUPSpace.sub(0), u_noslip, boundaries, marker)
    for marker in wall_markers
]
bcu_inlet_frozen = [
    DirichletBC(FrozenUPSpace.sub(0), profile, boundaries, marker)
    for profile, marker in zip(inlet_profiles, inlet_markers)
]
bcu_outlet_frozen = [
    DirichletBC(FrozenUPSpace.sub(0), profile, boundaries, marker)
    for profile, marker in zip(outlet_profiles, outlet_markers)
]
bc_NS_frozen = bcu_walls_frozen + bcu_inlet_frozen + bcu_outlet_frozen
if globals().get("ENABLE_PRESSURE_PIN", True):
    bcp_pin_frozen = DirichletBC(
        FrozenUPSpace.sub(1),
        Constant(0.0),
        build_pressure_pin_expression(),
        "pointwise",
    )
    bc_NS_frozen.append(bcp_pin_frozen)

wall_distance = calculate_distance_field(TurbulenceSpace, boundaries, wall_markers, dx, SA_DISTANCE_RELAXATION)
nu_laminar = Constant(MU_FLUID_VALUE / RHO_FLUID_VALUE)
sa_nu_tilde_init = _float_scalar(globals().get("SA_NU_TILDE_INITIAL", sa_nu_tilde_targets[0]))
sa_model = SpalartAllmarasSteadyState(
    TurbulenceSpace,
    bcn_turbulence,
    sa_nu_tilde_init,
    nu_laminar,
    Constant((0.0, 0.0)),
    dx,
    ds,
    wall_distance,
)


# ------------------------------------------------------------
# Design filter
# ------------------------------------------------------------
r_filter = compute_filter_base_length() * float(globals().get("FILTER_RADIUS_IN_CELLS", 2.0))
r = r_filter / (2.0 * 3.0**0.5)

u_filter = TrialFunction(DensitySpace)
v_filter = TestFunction(DensitySpace)
filter_in = Function(DensitySpace)
n = FacetNormal(mesh)
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


# ------------------------------------------------------------
# Optimization forms
# ------------------------------------------------------------
AreaOfInterest = interpolate(Constant(1.0), DensitySpace)
rho_effective = projection(rho_f, ETA_I)

mu_effective_obj_frozen = mu_fluid + rho_fluid * sa_positive_viscosity(nu_tilde_frozen)
ObjFunctional_frozen = AreaOfInterest * (
    0.5 * mu_effective_obj_frozen * inner(sym(nabla_grad(u_frozen_state)), sym(nabla_grad(u_frozen_state)))
    + alpha(rho_effective) * inner(u_frozen_state, u_frozen_state)
) * dx
state_form_frozen = build_frozen_state_form(
    u_frozen_state,
    p_frozen_state,
    v_frozen,
    q_frozen,
    rho_effective,
    dx,
    nu_tilde_frozen,
)
lagrangian_form_frozen = ObjFunctional_frozen + state_form_frozen
forward_form_frozen = derivative(state_form_frozen, w_adj_frozen, TestFunction(FrozenUPSpace))
adjoint_form_frozen = derivative(lagrangian_form_frozen, w_state_frozen, TestFunction(FrozenUPSpace))
ddx_frozen = derivative(lagrangian_form_frozen, rho_f)
vol_constraint = AreaOfInterest * rho_effective * dx - AreaOfInterest * VOL_FRAC * dx
sensitivities_vol_constraint = derivative(vol_constraint, rho_f)

StokesSpace = FunctionSpace(mesh, U_h * P_h)
u_lin, p_lin = TrialFunctions(StokesSpace)
v_lin, q_lin = TestFunctions(StokesSpace)
a_stokes = (
    mu_fluid * inner(grad(u_lin), grad(v_lin))
    + inner(grad(p_lin), v_lin)
    + inner(div(u_lin), q_lin)
    + alpha(rho_effective) * inner(u_lin, v_lin)
) * dx
l_stokes = Constant(0.0) * q_lin * dx

bcu_walls_stokes = [
    DirichletBC(StokesSpace.sub(0), u_noslip, boundaries, marker)
    for marker in wall_markers
]
bcu_inlet_stokes = [
    DirichletBC(StokesSpace.sub(0), profile, boundaries, marker)
    for profile, marker in zip(inlet_profiles, inlet_markers)
]
bcu_outlet_stokes = [
    DirichletBC(StokesSpace.sub(0), profile, boundaries, marker)
    for profile, marker in zip(outlet_profiles, outlet_markers)
]
bc_stokes = bcu_walls_stokes + bcu_inlet_stokes + bcu_outlet_stokes
if globals().get("ENABLE_PRESSURE_PIN", True):
    bcp_pin_stokes = DirichletBC(
        StokesSpace.sub(1),
        Constant(0.0),
        build_pressure_pin_expression(),
        "pointwise",
    )
    bc_stokes.append(bcp_pin_stokes)


def initialize_frozen_forward_guess_with_stokes():
    """Initialize frozen forward fields and SA state with a Stokes-Brinkman guess."""
    A = assemble(a_stokes)
    b = assemble(l_stokes)
    for bc in bc_stokes:
        bc.apply(A, b)
    w_stokes = Function(StokesSpace)
    solve(A, w_stokes.vector(), b, SNES_LINEAR_SOLVER)

    u_guess, p_guess = w_stokes.split(deepcopy=True)
    assign(w_state_frozen.sub(0), u_guess)
    assign(w_state_frozen.sub(1), p_guess)

    nu_guess_expr = Constant(sa_nu_tilde_init) * wall_distance / (
        wall_distance + Constant(SA_INIT_WALL_DIST_SCALE)
    )
    nu_guess = project(nu_guess_expr, TurbulenceSpace)
    nu_tilde_frozen.assign(nu_guess)
    sa_model.nu_tilde0.assign(nu_guess)
    sa_model.nu_tilde1.assign(nu_guess)


def build_forward_solver(forward_residual, state_function, boundary_conditions):
    """Create a forward SNES solver for a given residual/state/BC set."""
    jacobian = derivative(forward_residual, state_function)
    problem = NonlinearVariationalProblem(forward_residual, state_function, boundary_conditions, jacobian)
    solver = NonlinearVariationalSolver(problem)
    solver.parameters["nonlinear_solver"] = "snes"
    solver.parameters["snes_solver"]["linear_solver"] = SNES_LINEAR_SOLVER
    solver.parameters["snes_solver"]["method"] = FORWARD_SNES_METHOD
    if FORWARD_SNES_METHOD == "newtonls":
        solver.parameters["snes_solver"]["line_search"] = "l2"
    solver.parameters["snes_solver"]["relative_tolerance"] = FORWARD_SNES_RTOL
    solver.parameters["snes_solver"]["absolute_tolerance"] = FORWARD_SNES_ATOL
    solver.parameters["snes_solver"]["maximum_iterations"] = FORWARD_SNES_MAX_ITERS
    solver.parameters["snes_solver"]["error_on_nonconvergence"] = True
    return solver


def _get_attempt_value(attempt, key, legacy_key, default_value):
    if key in attempt:
        return attempt[key]
    if legacy_key in attempt:
        return attempt[legacy_key]
    return default_value


def resolve_forward_snes_attempts():
    """
    Resolve forward SNES attempts from config.

    Preferred config: FORWARD_SNES_ATTEMPTS = [ {method, line_search, rtol, atol, max_it, reinitialize}, ... ]
    Backward-compatible fallback uses FORWARD_SNES_* keys and preserves the old behavior:
    primary attempt + one reinitialized retry with the same settings.
    """
    attempts_cfg = globals().get("FORWARD_SNES_ATTEMPTS")
    if attempts_cfg is not None:
        if not isinstance(attempts_cfg, (list, tuple)) or len(attempts_cfg) == 0:
            raise ValueError("FORWARD_SNES_ATTEMPTS must be a non-empty list/tuple of dicts.")
        attempts = []
        for idx, attempt in enumerate(attempts_cfg):
            if not isinstance(attempt, dict):
                raise ValueError("FORWARD_SNES_ATTEMPTS[{}] must be a dict.".format(idx))
            method = _get_attempt_value(attempt, "method", "FORWARD_SNES_METHOD", FORWARD_SNES_METHOD)
            line_search = _get_attempt_value(
                attempt,
                "line_search",
                "FORWARD_SNES_LINE_SEARCH",
                globals().get("FORWARD_SNES_LINE_SEARCH", None),
            )
            relative_tolerance = _get_attempt_value(
                attempt, "rtol", "FORWARD_SNES_RTOL", FORWARD_SNES_RTOL
            )
            absolute_tolerance = _get_attempt_value(
                attempt, "atol", "FORWARD_SNES_ATOL", FORWARD_SNES_ATOL
            )
            maximum_iterations = _get_attempt_value(
                attempt, "max_it", "FORWARD_SNES_MAX_ITERS", FORWARD_SNES_MAX_ITERS
            )
            reinitialize = bool(attempt.get("reinitialize", idx > 0))
            attempts.append(
                {
                    "method": method,
                    "line_search": line_search,
                    "relative_tolerance": relative_tolerance,
                    "absolute_tolerance": absolute_tolerance,
                    "maximum_iterations": maximum_iterations,
                    "reinitialize": reinitialize,
                }
            )
        return attempts

    primary_attempt = {
        "method": FORWARD_SNES_METHOD,
        "line_search": globals().get("FORWARD_SNES_LINE_SEARCH", None),
        "relative_tolerance": FORWARD_SNES_RTOL,
        "absolute_tolerance": FORWARD_SNES_ATOL,
        "maximum_iterations": FORWARD_SNES_MAX_ITERS,
        "reinitialize": False,
    }
    # Preserve legacy behavior: if the primary solve fails, reinitialize and retry once.
    retry_attempt = dict(primary_attempt)
    retry_attempt["reinitialize"] = True
    attempts = [primary_attempt, retry_attempt]

    # Optional legacy fallback keys can append a final alternative attempt.
    fallback_method = globals().get("FORWARD_SNES_FALLBACK_METHOD")
    if fallback_method is not None:
        attempts.append(
            {
                "method": fallback_method,
                "line_search": globals().get("FORWARD_SNES_FALLBACK_LINE_SEARCH", None),
                "relative_tolerance": globals().get("FORWARD_SNES_FALLBACK_RTOL", FORWARD_SNES_RTOL),
                "absolute_tolerance": globals().get("FORWARD_SNES_FALLBACK_ATOL", FORWARD_SNES_ATOL),
                "maximum_iterations": globals().get("FORWARD_SNES_FALLBACK_MAX_ITERS", FORWARD_SNES_MAX_ITERS),
                "reinitialize": True,
            }
        )
    return attempts


FORWARD_SNES_ATTEMPTS_RESOLVED = resolve_forward_snes_attempts()


def solve_forward_with_attempts(forward_residual, state_function, boundary_conditions):
    """Solve forward residual using the configured ordered SNES attempts."""
    last_error = None
    total_attempts = len(FORWARD_SNES_ATTEMPTS_RESOLVED)
    for attempt_idx, attempt in enumerate(FORWARD_SNES_ATTEMPTS_RESOLVED):
        if attempt["reinitialize"]:
            initialize_frozen_forward_guess_with_stokes()
        solver = build_forward_solver(forward_residual, state_function, boundary_conditions)
        solver.parameters["snes_solver"]["method"] = attempt["method"]
        if attempt["method"] == "newtonls":
            if attempt["line_search"] is None:
                solver.parameters["snes_solver"]["line_search"] = "l2"
            else:
                solver.parameters["snes_solver"]["line_search"] = attempt["line_search"]
        solver.parameters["snes_solver"]["relative_tolerance"] = attempt["relative_tolerance"]
        solver.parameters["snes_solver"]["absolute_tolerance"] = attempt["absolute_tolerance"]
        solver.parameters["snes_solver"]["maximum_iterations"] = attempt["maximum_iterations"]
        try:
            solver.solve()
            if attempt_idx > 0:
                print(
                    "Forward SNES recovered on attempt {}/{} using method '{}'.".format(
                        attempt_idx + 1, total_attempts, attempt["method"]
                    )
                )
            return
        except RuntimeError as exc:
            last_error = exc
            print(
                "Forward SNES attempt {}/{} failed (method='{}', rtol={}, atol={}, max_it={}).".format(
                    attempt_idx + 1,
                    total_attempts,
                    attempt["method"],
                    attempt["relative_tolerance"],
                    attempt["absolute_tolerance"],
                    attempt["maximum_iterations"],
                )
            )
    raise RuntimeError("All forward SNES attempts failed. Last error: {}".format(last_error))


def resolve_adjoint_snes_attempts():
    """
    Resolve adjoint SNES attempts from config.

    Preferred config:
      ADJOINT_SNES_ATTEMPTS = [ {method, line_search, rtol, atol, max_it, reset_initial_guess}, ... ]

    Default behavior retries with progressively more robust settings when needed.
    """
    attempts_cfg = globals().get("ADJOINT_SNES_ATTEMPTS")
    default_max_iters = int(globals().get("ADJOINT_SNES_MAX_ITERS", 200))
    if attempts_cfg is not None:
        if not isinstance(attempts_cfg, (list, tuple)) or len(attempts_cfg) == 0:
            raise ValueError("ADJOINT_SNES_ATTEMPTS must be a non-empty list/tuple of dicts.")
        attempts = []
        for idx, attempt in enumerate(attempts_cfg):
            if not isinstance(attempt, dict):
                raise ValueError("ADJOINT_SNES_ATTEMPTS[{}] must be a dict.".format(idx))
            method = _get_attempt_value(attempt, "method", "ADJOINT_SNES_METHOD", "newtonls")
            line_search = _get_attempt_value(
                attempt,
                "line_search",
                "ADJOINT_SNES_LINE_SEARCH",
                globals().get("ADJOINT_SNES_LINE_SEARCH", "bt"),
            )
            relative_tolerance = _get_attempt_value(
                attempt, "rtol", "ADJOINT_SNES_RTOL", ADJOINT_SNES_RTOL
            )
            absolute_tolerance = _get_attempt_value(
                attempt, "atol", "ADJOINT_SNES_ATOL", ADJOINT_SNES_ATOL
            )
            maximum_iterations = int(
                _get_attempt_value(attempt, "max_it", "ADJOINT_SNES_MAX_ITERS", default_max_iters)
            )
            reset_initial_guess = bool(attempt.get("reset_initial_guess", idx > 0))
            attempts.append(
                {
                    "method": method,
                    "line_search": line_search,
                    "relative_tolerance": relative_tolerance,
                    "absolute_tolerance": absolute_tolerance,
                    "maximum_iterations": maximum_iterations,
                    "reset_initial_guess": reset_initial_guess,
                }
            )
        return attempts

    fallback_rtol = float(globals().get("ADJOINT_SNES_FALLBACK_RTOL", max(ADJOINT_SNES_RTOL, 5.0e-3)))
    fallback_atol = float(globals().get("ADJOINT_SNES_FALLBACK_ATOL", max(ADJOINT_SNES_ATOL, 1.0e-5)))
    fallback_max_iters = int(globals().get("ADJOINT_SNES_FALLBACK_MAX_ITERS", default_max_iters))

    return [
        {
            "method": "newtonls",
            "line_search": "bt",
            "relative_tolerance": ADJOINT_SNES_RTOL,
            "absolute_tolerance": ADJOINT_SNES_ATOL,
            "maximum_iterations": default_max_iters,
            "reset_initial_guess": False,
        },
        {
            "method": "newtonls",
            "line_search": "l2",
            "relative_tolerance": ADJOINT_SNES_RTOL,
            "absolute_tolerance": ADJOINT_SNES_ATOL,
            "maximum_iterations": default_max_iters,
            "reset_initial_guess": True,
        },
        {
            "method": "newtontr",
            "line_search": None,
            "relative_tolerance": fallback_rtol,
            "absolute_tolerance": fallback_atol,
            "maximum_iterations": fallback_max_iters,
            "reset_initial_guess": True,
        },
    ]


ADJOINT_SNES_ATTEMPTS_RESOLVED = resolve_adjoint_snes_attempts()


def solve_adjoint_with_attempts(adjoint_residual, adjoint_function, boundary_conditions):
    """Solve adjoint residual using configured ordered SNES attempts."""
    last_error = None
    total_attempts = len(ADJOINT_SNES_ATTEMPTS_RESOLVED)
    for attempt_idx, attempt in enumerate(ADJOINT_SNES_ATTEMPTS_RESOLVED):
        if attempt["reset_initial_guess"]:
            adjoint_function.vector().zero()
            adjoint_function.vector().apply("insert")

        jacobian = derivative(adjoint_residual, adjoint_function)
        problem = NonlinearVariationalProblem(adjoint_residual, adjoint_function, boundary_conditions, jacobian)
        solver = NonlinearVariationalSolver(problem)
        solver.parameters["nonlinear_solver"] = "snes"
        solver.parameters["snes_solver"]["linear_solver"] = SNES_LINEAR_SOLVER
        solver.parameters["snes_solver"]["method"] = attempt["method"]
        if attempt["method"] == "newtonls":
            if attempt["line_search"] is None:
                solver.parameters["snes_solver"]["line_search"] = "l2"
            else:
                solver.parameters["snes_solver"]["line_search"] = attempt["line_search"]
        solver.parameters["snes_solver"]["relative_tolerance"] = attempt["relative_tolerance"]
        solver.parameters["snes_solver"]["absolute_tolerance"] = attempt["absolute_tolerance"]
        solver.parameters["snes_solver"]["maximum_iterations"] = attempt["maximum_iterations"]
        solver.parameters["snes_solver"]["error_on_nonconvergence"] = True

        try:
            solver.solve()
            if attempt_idx > 0:
                print(
                    "Adjoint SNES recovered on attempt {}/{} using method '{}'.".format(
                        attempt_idx + 1, total_attempts, attempt["method"]
                    )
                )
            return
        except RuntimeError as exc:
            last_error = exc
            print(
                "Adjoint SNES attempt {}/{} failed (method='{}', rtol={}, atol={}, max_it={}).".format(
                    attempt_idx + 1,
                    total_attempts,
                    attempt["method"],
                    attempt["relative_tolerance"],
                    attempt["absolute_tolerance"],
                    attempt["maximum_iterations"],
                )
            )
    raise RuntimeError("All adjoint SNES attempts failed. Last error: {}".format(last_error))


# ------------------------------------------------------------
# Output setup
# ------------------------------------------------------------
results_root = os.path.join(
    THIS_DIR, globals().get("RESULTS_ROOT_NAME", "PipeBendTO_Results_Turbulent")
)
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

rho_out = File(os.path.join(rho_dir, "plot_rho.pvd"))
rhop_out = File(os.path.join(rho_p_dir, "plot_rho_projected.pvd"))
u_out = File(os.path.join(u_dir, "plot_u.pvd"))
p_out = File(os.path.join(p_dir, "plot_p.pvd"))
nu_tilde_out = File(os.path.join(nu_tilde_dir, "plot_nu_tilde.pvd"))

log_path = os.path.join(results_root, "OptimizationLogPipeBend.txt")
with open(log_path, "w") as txtout:
    txtout.write(
        "{} {} {} {} {}\r\n".format(
            "Iteration".ljust(12),
            "Objective".ljust(14),
            "ObjConv".ljust(12),
            "VolFrac".ljust(12),
            strftime("%a, %d %b %Y %H:%M:%S", localtime()),
        )
    )


# ------------------------------------------------------------
# MMA initialization
# ------------------------------------------------------------
assign(rho, interpolate(Constant(0.5), DensitySpace))

iter_count = 0
previous_objective = 0.0

num_mma = mesh.num_cells()
xval = np.zeros((num_mma, 1))
xval[:, 0] = rho.vector()
xold1 = np.zeros((num_mma, 1))
xold2 = np.zeros((num_mma, 1))
low = np.zeros((num_mma, 1))
upp = np.zeros((num_mma, 1))

mmma = 1
a0 = 1.0
a = np.zeros((mmma, 1))
c = 1.0e4 * np.ones((mmma, 1))
d = np.ones((mmma, 1))

xmin = np.zeros((num_mma, 1))
xmax = np.ones((num_mma, 1))

df0dx = np.zeros((num_mma, 1))
fval = np.zeros((mmma, 1))
dfdx = np.zeros((mmma, num_mma))

volume = assemble(AreaOfInterest * dx)
print("Frozen turbulence mode: segregated forward (NS + SA) with frozen adjoint/design derivatives")

if len(MOVE_LIMIT_SCHEDULE) != len(Q_PENAL_SCHEDULE):
    raise ValueError("MOVE_LIMIT_SCHEDULE must match Q_PENAL_SCHEDULE length.")
if len(BETA_PROJ_SCHEDULE) != len(Q_PENAL_SCHEDULE):
    raise ValueError("BETA_PROJ_SCHEDULE must match Q_PENAL_SCHEDULE length.")


# ------------------------------------------------------------
# Optimization loop
# ------------------------------------------------------------
for stage_idx, q_val in enumerate(Q_PENAL_SCHEDULE):
    beta_val = BETA_PROJ_SCHEDULE[stage_idx]
    BETA_PROJ.assign(beta_val)
    q_penal.assign(q_val)
    move_limit_now = MOVE_LIMIT_SCHEDULE[stage_idx]
    inner_count = 0
    convergence_history = 0
    objective_converged = False
    print(
        "Starting continuation stage {}/{}: q = {:.3f}, beta = {:.2f}, move = {:.4f}".format(
            stage_idx + 1,
            len(Q_PENAL_SCHEDULE),
            q_val,
            beta_val,
            move_limit_now,
        )
    )

    while inner_count < MAX_INNER_ITERATIONS and not objective_converged:
        ramp = min(1.0, float(iter_count + 1) / float(max(1, INLET_RAMP_STEPS)))
        apply_ramped_boundary_values(
            ramp,
            inlet_profiles,
            outlet_profiles,
            nu_tilde_inlet_bc_values,
            sa_nu_tilde_targets,
        )

        # Filter current design
        rho_f = pde_filter(rho, rho_f)
        rho_proj_plot.vector()[:] = project(rho_effective, DensitySpace).vector()[:]

        rho_out << rho
        rhop_out << rho_proj_plot

        # Forward solve
        print("Starting frozen forward solve (segregated NS + SA)")

        if iter_count == 0:
            initialize_frozen_forward_guess_with_stokes()

        for _ in range(max(1, FROZEN_PICARD_STEPS)):
            solve_forward_with_attempts(forward_form_frozen, w_state_frozen, bc_NS_frozen)

            velocity_for_sa = w_state_frozen.sub(0, deepcopy=True)
            sa_model.construct_forms(velocity_for_sa)
            sa_model.solve_turbulence_model()
            sa_model.update_variables(relaxation=NUT_RELAXATION_FACTOR)
            nu_tilde_frozen.assign(sa_model.nu_tilde0)
            enforce_scalar_floor(nu_tilde_frozen, SA_NU_TILDE_FLOOR)
            sa_model.nu_tilde0.assign(nu_tilde_frozen)
            sa_model.nu_tilde1.assign(nu_tilde_frozen)

        # Re-solve momentum with the updated frozen turbulence field.
        solve_forward_with_attempts(forward_form_frozen, w_state_frozen, bc_NS_frozen)

        # Adjoint solve
        print("Starting adjoint SNES solve")
        solve_adjoint_with_attempts(adjoint_form_frozen, w_adj_frozen, bc_NS_adj_frozen)

        u_out << w_state_frozen.sub(0)
        p_out << w_state_frozen.sub(1)
        nu_tilde_out << nu_tilde_frozen
        f0val = assemble(ObjFunctional_frozen)
        obj_conv = abs((f0val - previous_objective) / max(abs(f0val), 1e-12))

        if obj_conv < OBJECTIVE_CONVERGENCE_TOL:
            convergence_history += 1
            if convergence_history >= OBJECTIVE_STREAK_TO_STOP:
                objective_converged = True
        else:
            convergence_history = 0

        previous_objective = f0val

        # Objective gradient
        unfiltered_gradient.vector()[:] = assemble(ddx_frozen)[:]
        filtered_gradient = pde_filter(unfiltered_gradient, filtered_gradient)
        np.savetxt(os.path.join(design_dir, "rho_{:03}.txt".format(iter_count)), rho.vector()[:])

        # Constraint and constraint gradient
        fval[0, 0] = assemble(vol_constraint)
        unfiltered_s_vol.vector()[:] = assemble(sensitivities_vol_constraint)[:]
        filtered_s_vol = pde_filter(unfiltered_s_vol, filtered_s_vol)

        df0dx[:, 0] = filtered_gradient.vector()[:]
        dfdx[0, :] = filtered_s_vol.vector()[:]

        # MMA update
        (
            xmma,
            _ymma,
            _zmma,
            _lam,
            _xsi,
            _eta,
            _mu_mma,
            _zet,
            _s,
            low,
            upp,
        ) = mmasub(
            mmma,
            num_mma,
            iter_count,
            xval,
            xmin,
            xmax,
            xold1,
            xold2,
            f0val,
            df0dx,
            fval,
            dfdx,
            low,
            upp,
            a0,
            a,
            c,
            d,
            move_limit_now,
        )

        xold2 = xold1.copy()
        xold1 = xval.copy()
        xval = xmma.copy()
        rho.vector()[:] = xmma[:, 0].copy()

        vol_fraction_now = assemble(rho * dx) / volume
        with open(log_path, "a") as txtout:
            txtout.write(
                "{:03d}.{:03d}   {:.10e}   {:.10e}   {:.10e}   {}\r\n".format(
                    int(q_val * 1000),
                    inner_count,
                    f0val,
                    obj_conv,
                    vol_fraction_now,
                    strftime("%a, %d %b %Y %H:%M:%S", localtime()),
                )
            )

        print(
            "q = {:.3f}, beta = {:.2f}, iter = {:03d}, J = {:.4e}, obj_conv = {:.3e}, vol = {:.4f}".format(
                q_val, beta_val, inner_count, f0val, obj_conv, vol_fraction_now
            )
        )

        inner_count += 1
        iter_count += 1

print("Optimization finished. Results written to {}".format(results_root))
