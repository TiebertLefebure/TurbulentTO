from dolfin import *
import numpy as np
import os
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
from Utilities_TurbulentTO import (
    as_list,
    build_penalized_wall_distance_solver,
    ensure_clean_dir,
    enforce_scalar_floor,
    float_scalar,
    load_config_module_from_cli,
    match_count,
    positive_part,
)

CONFIG_MODULE_NAME, CONFIG = load_config_module_from_cli()
for _name, _value in vars(CONFIG).items():
    if not _name.startswith("_"):
        globals()[_name] = _value
print("Using config module: {}".format(CONFIG_MODULE_NAME))

BETA_PROJ = Constant(float_scalar(BETA_PROJ_VALUE))

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


def sa_positive_viscosity(state_nu_tilde):
    nu_lam = mu_fluid / rho_fluid
    return sa_turbulent_viscosity(
        state_nu_tilde,
        nu_lam,
        smooth_abs_eps=float(SA_SMOOTH_ABS_EPS),
    )


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
if "create_design_mesh" in globals():
    mesh = create_design_mesh()
else:
    x_min = float(globals().get("DOMAIN_X_MIN", 0.0))
    y_min = float(globals().get("DOMAIN_Y_MIN", 0.0))
    x_max = float(globals().get("DOMAIN_X_MAX", globals().get("L", 1.0)))
    y_max = float(globals().get("DOMAIN_Y_MAX", globals().get("L", x_max)))
    nx = int(globals().get("NX", globals().get("N", 96)))
    ny = int(globals().get("NY", globals().get("N", nx)))
    mesh_diagonal = globals().get("MESH_DIAGONAL", "crossed")
    mesh = Mesh(
        RectangleMesh(
            MPI.comm_world,
            Point(x_min, y_min),
            Point(x_max, y_max),
            nx,
            ny,
            mesh_diagonal,
        )
    )

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
if "mark_boundaries" in globals():
    boundaries = mark_boundaries(mesh)
else:
    inlet_y_min, inlet_y_max, outlet_x_min, outlet_x_max = compute_port_extents(
        L, INLET_TOP_OFFSET, INLET_WIDTH, OUTLET_RIGHT_OFFSET, OUTLET_WIDTH
    )
    boundaries = mark_pipe_bend_boundaries(
        mesh,
        L,
        TOL,
        inlet_y_min,
        inlet_y_max,
        outlet_x_min,
        outlet_x_max,
    )
wall_markers = as_list(mark["walls"])
inlet_markers = as_list(mark["inlet"])
outlet_markers = as_list(mark["outlet"])

pin_point = globals().get(
    "PRESSURE_PIN_POINT",
    (
        float(globals().get("DOMAIN_X_MIN", 0.0)),
        float(globals().get("DOMAIN_Y_MIN", 0.0)),
    ),
)
pressure_pin_expression = "near(x[0], {:.16g}) && near(x[1], {:.16g})".format(
    float(pin_point[0]),
    float(pin_point[1]),
)

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

if "build_velocity_profile_sets" in globals():
    inlet_profiles, outlet_profiles = build_velocity_profile_sets()
else:
    inlet_y_min, inlet_y_max, outlet_x_min, outlet_x_max = compute_port_extents(
        L, INLET_TOP_OFFSET, INLET_WIDTH, OUTLET_RIGHT_OFFSET, OUTLET_WIDTH
    )
    u_inlet, u_outlet = build_velocity_profiles(
        U_MAX_INLET,
        U_MAX_OUTLET,
        inlet_y_min,
        inlet_y_max,
        outlet_x_min,
        outlet_x_max,
        INLET_WIDTH,
        OUTLET_WIDTH,
    )
    inlet_profiles, outlet_profiles = [u_inlet], [u_outlet]
inlet_profiles = match_count(as_list(inlet_profiles), len(inlet_markers))
outlet_profiles = match_count(as_list(outlet_profiles), len(outlet_markers))

u_noslip = Constant((0.0, 0.0))

if "SA_NU_TILDE_INLETS" in globals():
    sa_nu_tilde_targets = match_count(SA_NU_TILDE_INLETS, len(inlet_markers))
else:
    sa_nu_tilde_targets = match_count([SA_NU_TILDE_INLET], len(inlet_markers))
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
        pressure_pin_expression,
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
        pressure_pin_expression,
        "pointwise",
    )
    bc_NS_frozen.append(bcp_pin_frozen)

# Use filtered/projected density to define Brinkman and design-dependent wall-distance fields.
AreaOfInterest = interpolate(Constant(1.0), DensitySpace)
rho_effective = projection(rho_f, ETA_I)

sa_wall_sigma = float(globals().get("SA_WALL_SIGMA", SA_DISTANCE_RELAXATION))
sa_wall_g0 = float(globals().get("SA_WALL_G0", 20.0))
sa_wall_penalty_alpha = float(globals().get("SA_WALL_PENALTY_ALPHA", 1.0e3))
sa_wall_penalty_power = float(globals().get("SA_WALL_PENALTY_N", 3.0))
sa_wall_g_floor = float(globals().get("SA_WALL_G_FLOOR", 1.0e-8))
sa_wall_newton_rtol = float(globals().get("SA_WALL_NEWTON_RTOL", 1.0e-8))
sa_wall_newton_atol = float(globals().get("SA_WALL_NEWTON_ATOL", 1.0e-10))
sa_wall_newton_max_iters = int(globals().get("SA_WALL_NEWTON_MAX_ITERS", 80))
sa_wall_newton_relax = float(globals().get("SA_WALL_NEWTON_RELAX", 1.0))
(
    wall_distance,
    _wall_distance_reciprocal,
    update_wall_distance_field,
) = build_penalized_wall_distance_solver(
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
    sa_wall_newton_rtol,
    sa_wall_newton_atol,
    sa_wall_newton_max_iters,
    sa_wall_newton_relax,
)
print(
    "SA wall equation with penalty: sigma={}, G0={}, alpha_G={}, n_G={}".format(
        sa_wall_sigma,
        sa_wall_g0,
        sa_wall_penalty_alpha,
        sa_wall_penalty_power,
    )
)

nu_laminar = Constant(MU_FLUID_VALUE / RHO_FLUID_VALUE)
sa_nu_tilde_init = float(globals().get("SA_NU_TILDE_INITIAL", sa_nu_tilde_targets[0]))
sa_nu_tilde_penalty_alpha = Constant(float(globals().get("SA_NU_TILDE_PENALTY_ALPHA", 1.0e3)))
sa_nu_tilde_penalty_power = float(globals().get("SA_NU_TILDE_PENALTY_N", 3.0))
# TO convention here: rho_effective=1 fluid, 0 solid.
sa_nu_tilde_penalty_reaction = sa_nu_tilde_penalty_alpha * (
    positive_part(Constant(1.0) - rho_effective) ** sa_nu_tilde_penalty_power
)

sa_model = SpalartAllmarasSteadyState(
    TurbulenceSpace,
    bcn_turbulence,
    sa_nu_tilde_init,
    nu_laminar,
    Constant((0.0, 0.0)),
    dx,
    ds,
    wall_distance,
    nu_tilde_penalty_reaction=sa_nu_tilde_penalty_reaction,
)


# ------------------------------------------------------------
# Design filter
# ------------------------------------------------------------
if "FILTER_BASE_LENGTH" in globals():
    filter_base_length = float(FILTER_BASE_LENGTH)
elif "L" in globals() and "N" in globals():
    filter_base_length = float(L) / float(N)
else:
    x_min = float(globals().get("DOMAIN_X_MIN", 0.0))
    y_min = float(globals().get("DOMAIN_Y_MIN", 0.0))
    x_max = float(globals().get("DOMAIN_X_MAX", 1.0))
    y_max = float(globals().get("DOMAIN_Y_MAX", 1.0))
    nx = int(globals().get("NX", globals().get("N", 1)))
    ny = int(globals().get("NY", globals().get("N", 1)))
    filter_base_length = min((x_max - x_min) / max(nx, 1), (y_max - y_min) / max(ny, 1))
r_filter = filter_base_length * float(globals().get("FILTER_RADIUS_IN_CELLS", 2.0))
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
        pressure_pin_expression,
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

# -------------------------
# SNES solver attempts
# -------------------------

_forward_attempts = list(globals().get("FORWARD_SNES_ATTEMPTS", []))
if not _forward_attempts:
    _forward_attempts = [
        {
            "method": FORWARD_SNES_METHOD,
            "line_search": globals().get("FORWARD_SNES_LINE_SEARCH", None),
            "rtol": FORWARD_SNES_RTOL,
            "atol": FORWARD_SNES_ATOL,
            "max_it": min(FORWARD_SNES_MAX_ITERS, 120),
            "reinitialize": False,
        },
        {
            "method": FORWARD_SNES_METHOD,
            "line_search": globals().get("FORWARD_SNES_LINE_SEARCH", None),
            "rtol": FORWARD_SNES_RTOL,
            "atol": FORWARD_SNES_ATOL,
            "max_it": min(FORWARD_SNES_MAX_ITERS, 120),
            "reinitialize": True,
        },
        {
            "method": "newtonls",
            "line_search": "bt",
            "rtol": max(FORWARD_SNES_RTOL, 5.0e-3),
            "atol": max(FORWARD_SNES_ATOL, 1.0e-5),
            "max_it": max(FORWARD_SNES_MAX_ITERS, 200),
            "reinitialize": True,
        }
    ]

FORWARD_SNES_ATTEMPTS_RESOLVED = []
for idx, attempt in enumerate(_forward_attempts):
    FORWARD_SNES_ATTEMPTS_RESOLVED.append(
        {
            "method": attempt.get("method", FORWARD_SNES_METHOD),
            "line_search": attempt.get("line_search", globals().get("FORWARD_SNES_LINE_SEARCH", None)),
            "relative_tolerance": attempt.get("rtol", FORWARD_SNES_RTOL),
            "absolute_tolerance": attempt.get("atol", FORWARD_SNES_ATOL),
            "maximum_iterations": int(attempt.get("max_it", FORWARD_SNES_MAX_ITERS)),
            "reinitialize": bool(attempt.get("reinitialize", idx > 0)),
        }
    )


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
    if last_error is not None:
        raise last_error

# -------------------------
# Adjoint solver attempts
# -------------------------

_adjoint_attempts = list(globals().get("ADJOINT_SNES_ATTEMPTS", []))
if not _adjoint_attempts:
    _adjoint_attempts = [
        {
            "method": "newtonls",
            "line_search": "bt",
            "rtol": ADJOINT_SNES_RTOL,
            "atol": ADJOINT_SNES_ATOL,
            "max_it": int(globals().get("ADJOINT_SNES_MAX_ITERS", 200)),
            "reset_initial_guess": False,
        },
        {
            "method": "newtontr",
            "line_search": None,
            "rtol": max(ADJOINT_SNES_RTOL, 5.0e-3),
            "atol": max(ADJOINT_SNES_ATOL, 1.0e-5),
            "max_it": int(globals().get("ADJOINT_SNES_MAX_ITERS", 200)),
            "reset_initial_guess": True,
        },
    ]

ADJOINT_SNES_ATTEMPTS_RESOLVED = []
for idx, attempt in enumerate(_adjoint_attempts):
    ADJOINT_SNES_ATTEMPTS_RESOLVED.append(
        {
            "method": attempt.get("method", "newtonls"),
            "line_search": attempt.get("line_search", "bt"),
            "relative_tolerance": attempt.get("rtol", ADJOINT_SNES_RTOL),
            "absolute_tolerance": attempt.get("atol", ADJOINT_SNES_ATOL),
            "maximum_iterations": int(attempt.get("max_it", int(globals().get("ADJOINT_SNES_MAX_ITERS", 200)))),
            "reset_initial_guess": bool(attempt.get("reset_initial_guess", idx > 0)),
        }
    )


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
# MMA iteration index must count design updates only (warm-up ramp steps do not call MMA).
mma_iter_count = 0
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

num_stages = min(len(Q_PENAL_SCHEDULE), len(MOVE_LIMIT_SCHEDULE), len(BETA_PROJ_SCHEDULE))
design_updates_after_full_ramp = bool(globals().get("DESIGN_UPDATES_AFTER_FULL_RAMP", True))
ramp_steps = max(1, int(globals().get("INLET_RAMP_STEPS", 1)))
mma_damping = min(max(float(globals().get("MMA_DAMPING", 1.0)), 0.0), 1.0)
mma_damping_on_spike = min(
    max(float(globals().get("MMA_DAMPING_ON_SPIKE", mma_damping)), 0.0),
    1.0,
)
objective_spike_rel_tol = max(float(globals().get("OBJECTIVE_SPIKE_REL_TOL", 0.3)), 0.0)
move_limit_reduction_on_spike = min(
    max(float(globals().get("MOVE_LIMIT_REDUCTION_ON_SPIKE", 0.7)), 0.05),
    1.0,
)
move_limit_recovery_factor = max(float(globals().get("MOVE_LIMIT_RECOVERY_FACTOR", 1.02)), 1.0)
move_limit_min = max(float(globals().get("MOVE_LIMIT_MIN", 5.0e-4)), 1.0e-6)


# ------------------------------------------------------------
# Optimization loop
# ------------------------------------------------------------
for stage_idx in range(num_stages):
    q_val = Q_PENAL_SCHEDULE[stage_idx]
    beta_val = BETA_PROJ_SCHEDULE[stage_idx]
    BETA_PROJ.assign(beta_val)
    q_penal.assign(q_val)
    base_move_limit = float(MOVE_LIMIT_SCHEDULE[stage_idx])
    move_limit_now = base_move_limit
    inner_count = 0
    convergence_history = 0
    objective_converged = False
    print(
            "Starting continuation stage {}/{}: q = {:.3f}, beta = {:.2f}, move = {:.4f}".format(
                stage_idx + 1,
                num_stages,
                q_val,
                beta_val,
                base_move_limit,
        )
    )

    while inner_count < MAX_INNER_ITERATIONS and not objective_converged:
        ramp = min(1.0, float(iter_count + 1) / float(ramp_steps))
        if "U_MAX_INLETS" in globals():
            inlet_targets = match_count(U_MAX_INLETS, len(inlet_profiles))
        else:
            inlet_targets = match_count([U_MAX_INLET], len(inlet_profiles))
        if "U_MAX_OUTLETS" in globals():
            outlet_targets = match_count(U_MAX_OUTLETS, len(outlet_profiles))
        else:
            outlet_targets = match_count([U_MAX_OUTLET], len(outlet_profiles))
        for profile, u_target in zip(inlet_profiles, inlet_targets):
            profile.u_max = ramp * float(u_target)
        for profile, u_target in zip(outlet_profiles, outlet_targets):
            profile.u_max = ramp * float(u_target)
        for constant_value, nu_target in zip(nu_tilde_inlet_bc_values, sa_nu_tilde_targets):
            constant_value.assign(ramp * float(nu_target))

        # Filter current design
        rho_f = pde_filter(rho, rho_f)
        rho_proj_plot.vector()[:] = project(rho_effective, DensitySpace).vector()[:]

        # Recompute penalized wall-distance with the current filtered design.
        update_wall_distance_field()

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

        u_out << w_state_frozen.sub(0)
        p_out << w_state_frozen.sub(1)
        nu_tilde_out << nu_tilde_frozen
        f0val = assemble(ObjFunctional_frozen)
        # Symmetric normalization avoids pathological spikes when the objective drops sharply.
        obj_conv = abs(f0val - previous_objective) / max(abs(f0val), abs(previous_objective), 1e-12)

        # Keep design fixed while inlet/turbulence ramp reaches full magnitude.
        if design_updates_after_full_ramp and ramp < 1.0:
            warmup_step = iter_count + 1
            previous_objective = f0val
            iter_count += 1
            print(
                "Warm-up ramp {}/{}: ramp = {:.3f}, J = {:.4e}".format(
                    warmup_step, ramp_steps, ramp, f0val
                )
            )
            continue

        # Adjoint solve
        print("Starting adjoint SNES solve")
        solve_adjoint_with_attempts(adjoint_form_frozen, w_adj_frozen, bc_NS_adj_frozen)

        if obj_conv < OBJECTIVE_CONVERGENCE_TOL:
            convergence_history += 1
            if convergence_history >= OBJECTIVE_STREAK_TO_STOP:
                objective_converged = True
        else:
            convergence_history = 0

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

        objective_spike = (
            inner_count > 0
            and
            previous_objective > 0.0
            and f0val > (1.0 + objective_spike_rel_tol) * previous_objective
        )
        if objective_spike:
            move_limit_now = max(move_limit_min, move_limit_now * move_limit_reduction_on_spike)
            update_damping = mma_damping_on_spike
        else:
            move_limit_now = min(base_move_limit, move_limit_now * move_limit_recovery_factor)
            update_damping = mma_damping

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
            mma_iter_count + 1,
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

        # Damped update improves robustness for SA-TO coupling under aggressive gradients.
        xmma = xval + update_damping * (xmma - xval)
        xmma = np.maximum(xmin, np.minimum(xmax, xmma))

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
            "q = {:.3f}, beta = {:.2f}, iter = {:03d}, J = {:.4e}, obj_conv = {:.3e}, vol = {:.4f}, move = {:.4f}, damping = {:.2f}".format(
                q_val,
                beta_val,
                inner_count,
                f0val,
                obj_conv,
                vol_fraction_now,
                move_limit_now,
                update_damping,
            )
        )

        previous_objective = f0val
        inner_count += 1
        mma_iter_count += 1
        iter_count += 1

print("Optimization finished. Results written to {}".format(results_root))
