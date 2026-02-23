from dolfin import *
import inspect
import numpy as np
import os
from time import localtime, strftime
from ufl import tanh

from mma import mmasub
from Utilities_LaminarTO import (
    as_list,
    build_pressure_pin_expression_from_config,
    compute_filter_base_length_from_config,
    compute_legacy_port_extents_from_config,
    ensure_clean_dir,
    float_scalar,
    load_config_module_from_cli,
    match_count,
)

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


CONFIG_MODULE_NAME, CONFIG = load_config_module_from_cli()
for _name, _value in vars(CONFIG).items():
    if not _name.startswith("_"):
        globals()[_name] = _value
print("Using config module: {}".format(CONFIG_MODULE_NAME))

BETA_PROJ = Constant(float_scalar(BETA_PROJ_VALUE))
if "BETA_PROJ_SCHEDULE" in globals():
    BETA_PROJ_SCHEDULE = [float(float_scalar(beta_value)) for beta_value in BETA_PROJ_SCHEDULE]
else:
    BETA_PROJ_SCHEDULE = [float(float_scalar(BETA_PROJ_VALUE))] * len(Q_PENAL_SCHEDULE)

# Brinkman penalization constants (same style as DiffuserBorrvallTO.py script)
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
    return compute_filter_base_length_from_config(globals())


def compute_legacy_port_extents():
    return compute_legacy_port_extents_from_config(globals())


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
        return as_list(inlet_profiles), as_list(outlet_profiles)

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
        return as_list(profile_data[0]), as_list(profile_data[1])

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


def apply_ramped_velocity_profiles(ramp, inlet_profiles, outlet_profiles):
    custom_ramp = globals().get("apply_velocity_ramp")
    if callable(custom_ramp):
        custom_ramp(ramp, inlet_profiles, outlet_profiles)
        return

    if "U_MAX_INLETS" in globals():
        inlet_targets = as_list(U_MAX_INLETS)
    elif "U_MAX_INLET" in globals():
        inlet_targets = [U_MAX_INLET]
    else:
        raise ValueError("Config must define U_MAX_INLETS or U_MAX_INLET.")

    if "U_MAX_OUTLETS" in globals():
        outlet_targets = as_list(U_MAX_OUTLETS)
    elif "U_MAX_OUTLET" in globals():
        outlet_targets = [U_MAX_OUTLET]
    else:
        raise ValueError("Config must define U_MAX_OUTLETS or U_MAX_OUTLET.")

    inlet_targets = match_count(inlet_targets, len(inlet_profiles))
    outlet_targets = match_count(outlet_targets, len(outlet_profiles))

    for profile, u_target in zip(inlet_profiles, inlet_targets):
        if not hasattr(profile, "u_max"):
            raise ValueError("Inlet profile is missing writable attribute 'u_max' for ramping.")
        profile.u_max = ramp * float_scalar(u_target)
    for profile, u_target in zip(outlet_profiles, outlet_targets):
        if not hasattr(profile, "u_max"):
            raise ValueError("Outlet profile is missing writable attribute 'u_max' for ramping.")
        profile.u_max = ramp * float_scalar(u_target)


def build_pressure_pin_expression():
    return build_pressure_pin_expression_from_config(globals())


def build_state_form(state_u, state_p, adj_u, adj_p, rho_eff, custom_dx):
    """Build the scalar UFL state form used in forward and adjoint derivations."""
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
mesh = create_design_mesh_from_config()

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

mark = MARK
boundaries = build_marked_boundaries(mesh)
wall_markers = as_list(mark["walls"])
inlet_markers = as_list(mark["inlet"])
outlet_markers = as_list(mark["outlet"])

dx = Measure("dx", domain=mesh)
ds = Measure("ds", domain=mesh, subdomain_data=boundaries)
dS = Measure("dS", domain=mesh)

inlet_profiles, outlet_profiles = build_velocity_profile_sets_from_config()
inlet_profiles = match_count(inlet_profiles, len(inlet_markers))
outlet_profiles = match_count(outlet_profiles, len(outlet_markers))

u_noslip = Constant((0.0, 0.0))

bcu_walls = [DirichletBC(FlowSpace.sub(0), u_noslip, boundaries, marker) for marker in wall_markers]
bcu_inlet = [
    DirichletBC(FlowSpace.sub(0), profile, boundaries, marker)
    for profile, marker in zip(inlet_profiles, inlet_markers)
]
bcu_outlet = [
    DirichletBC(FlowSpace.sub(0), profile, boundaries, marker)
    for profile, marker in zip(outlet_profiles, outlet_markers)
]
bc_NS = bcu_walls + bcu_inlet + bcu_outlet
if globals().get("ENABLE_PRESSURE_PIN", True):
    bcp_pin = DirichletBC(
        FlowSpace.sub(1),
        Constant(0.0),
        build_pressure_pin_expression(),
        "pointwise",
    )
    bc_NS.append(bcp_pin)

adjoint_velocity_bc_builder = globals().get("build_adjoint_velocity_bcs")
if callable(adjoint_velocity_bc_builder):
    bc_NS_adj = list(
        adjoint_velocity_bc_builder(
            FlowSpaceAdj,
            boundaries,
            wall_markers,
            inlet_markers,
            outlet_markers,
            u_noslip,
            inlet_profiles,
            outlet_profiles,
        )
    )
else:
    bcu_walls_adj = [
        DirichletBC(FlowSpaceAdj.sub(0), u_noslip, boundaries, marker) for marker in wall_markers
    ]
    bcu_inlet_adj = [
        DirichletBC(FlowSpaceAdj.sub(0), u_noslip, boundaries, marker) for marker in inlet_markers
    ]
    bcu_outlet_adj = [
        DirichletBC(FlowSpaceAdj.sub(0), u_noslip, boundaries, marker) for marker in outlet_markers
    ]
    bc_NS_adj = bcu_walls_adj + bcu_inlet_adj + bcu_outlet_adj
if globals().get("ENABLE_PRESSURE_PIN", True):
    bcp_pin_adj = DirichletBC(
        FlowSpaceAdj.sub(1),
        Constant(0.0),
        build_pressure_pin_expression(),
        "pointwise",
    )
    bc_NS_adj.append(bcp_pin_adj)


# ------------------------------------------------------------
# Design filter
# ------------------------------------------------------------
r_filter = compute_filter_base_length() * float(globals().get("FILTER_RADIUS_IN_CELLS", 3.0))
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

dissipation_density_builder = globals().get("build_dissipation_density")
if callable(dissipation_density_builder):
    dissipation_density = dissipation_density_builder(u, mu_fluid)
else:
    dissipation_density = 0.5 * mu_fluid * inner(sym(nabla_grad(u)), sym(nabla_grad(u)))

ObjFunctional = AreaOfInterest * (
    dissipation_density + alpha(rho_effective) * inner(u, u)
) * dx

state_form = build_state_form(u, p, v, q, rho_effective, dx)
lagrangian_form = ObjFunctional + state_form

forward_form = derivative(state_form, w_adj, TestFunction(FlowSpace))
adjoint_form = derivative(lagrangian_form, w_fwd, TestFunction(FlowSpaceAdj))

ddx = derivative(lagrangian_form, rho_f)
vol_constraint = AreaOfInterest * rho_effective * dx - AreaOfInterest * VOL_FRAC * dx
sensitivities_vol_constraint = derivative(vol_constraint, rho_f)

u_lin, p_lin = TrialFunctions(FlowSpace)
v_lin, q_lin = TestFunctions(FlowSpace)
a_stokes = (
    mu_fluid * inner(grad(u_lin), grad(v_lin))
    + inner(grad(p_lin), v_lin)
    + inner(div(u_lin), q_lin)
    + alpha(rho_effective) * inner(u_lin, v_lin)
) * dx
l_stokes = Constant(0.0) * q_lin * dx


def initialize_forward_guess_with_stokes():
    """Initialize w_fwd with a linear Stokes-Brinkman solve for robust Newton startup."""
    A = assemble(a_stokes)
    b = assemble(l_stokes)
    for bc in bc_NS:
        bc.apply(A, b)
    solve(A, w_fwd.vector(), b, SNES_LINEAR_SOLVER)


# ------------------------------------------------------------
# Output setup
# ------------------------------------------------------------
results_root = os.path.join(
    THIS_DIR, globals().get("RESULTS_ROOT_NAME", "PipeBendTO_Results_Laminar")
)
rho_dir = os.path.join(results_root, "rho")
rho_p_dir = os.path.join(results_root, "rho_projected")
u_dir = os.path.join(results_root, "u")
p_dir = os.path.join(results_root, "p")
design_dir = os.path.join(results_root, "design")

ensure_clean_dir(results_root)
ensure_clean_dir(rho_dir)
ensure_clean_dir(rho_p_dir)
ensure_clean_dir(u_dir)
ensure_clean_dir(p_dir)
ensure_clean_dir(design_dir)

rho_out = File(os.path.join(rho_dir, "plot_rho.pvd"))
rhop_out = File(os.path.join(rho_p_dir, "plot_rho_projected.pvd"))
u_out = File(os.path.join(u_dir, "plot_u.pvd"))
p_out = File(os.path.join(p_dir, "plot_p.pvd"))

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

if len(MOVE_LIMIT_SCHEDULE) != len(Q_PENAL_SCHEDULE):
    raise ValueError("MOVE_LIMIT_SCHEDULE must match Q_PENAL_SCHEDULE length.")
if len(BETA_PROJ_SCHEDULE) != len(Q_PENAL_SCHEDULE):
    raise ValueError("BETA_PROJ_SCHEDULE must match Q_PENAL_SCHEDULE length.")


# ------------------------------------------------------------
# Optimization loop
# ------------------------------------------------------------
for stage_idx, q_val in enumerate(Q_PENAL_SCHEDULE):
    beta_val = float(BETA_PROJ_SCHEDULE[stage_idx])
    BETA_PROJ.assign(beta_val)
    move_limit_now = MOVE_LIMIT_SCHEDULE[stage_idx]
    q_penal.assign(q_val)
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
        apply_ramped_velocity_profiles(ramp, inlet_profiles, outlet_profiles)

        # Filter current design
        rho_f = pde_filter(rho, rho_f)
        rho_proj_plot.vector()[:] = project(rho_effective, DensitySpace).vector()[:]

        rho_out << rho
        rhop_out << rho_proj_plot

        # Forward solve
        jac_fwd = derivative(forward_form, w_fwd)
        problem_fwd = NonlinearVariationalProblem(forward_form, w_fwd, bc_NS, jac_fwd)
        solver_fwd = NonlinearVariationalSolver(problem_fwd)
        solver_fwd.parameters["nonlinear_solver"] = "snes"
        solver_fwd.parameters["snes_solver"]["linear_solver"] = SNES_LINEAR_SOLVER
        solver_fwd.parameters["snes_solver"]["method"] = "newtonls"
        solver_fwd.parameters["snes_solver"]["line_search"] = "bt"
        solver_fwd.parameters["snes_solver"]["relative_tolerance"] = FORWARD_SNES_RTOL
        solver_fwd.parameters["snes_solver"]["absolute_tolerance"] = FORWARD_SNES_ATOL
        solver_fwd.parameters["snes_solver"]["maximum_iterations"] = SNES_MAX_ITERS
        solver_fwd.parameters["snes_solver"]["error_on_nonconvergence"] = True

        if iter_count == 0:
            initialize_forward_guess_with_stokes()

        try:
            solver_fwd.solve()
        except RuntimeError:
            initialize_forward_guess_with_stokes()
            solver_fwd.solve()

        # Adjoint solve
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
        solver_adj.parameters["snes_solver"]["error_on_nonconvergence"] = True
        solver_adj.solve()

        u_out << w_fwd.sub(0)
        p_out << w_fwd.sub(1)

        f0val = assemble(ObjFunctional)
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
            "q = {:.3f}, beta = {:.2f}, move = {:.3f}, iter = {:03d}, J = {:.4e}, obj_conv = {:.3e}, vol = {:.4f}".format(
                q_val, float(BETA_PROJ.values()[0]), move_limit_now, inner_count, f0val, obj_conv, vol_fraction_now
            )
        )

        inner_count += 1
        iter_count += 1

print("Optimization finished. Results written to {}".format(results_root))
