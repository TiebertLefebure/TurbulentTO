from dolfin import *
import numpy as np
import os
import shutil
import sys
from time import localtime, strftime
from ufl import tanh

from mma import mmasub

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
TURB_MODELS_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", "TurbulenceModels"))
if TURB_MODELS_DIR not in sys.path:
    sys.path.insert(0, TURB_MODELS_DIR)
from TurbulenceModel_SpalartAllmaras_TO import (
    SpalartAllmarasSteadyState,
    sa_turbulent_viscosity,
)
from Config_PipeBendBorrvall_TurbulentTO import *


BETA_PROJ = Constant(BETA_PROJ_VALUE)

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


def calculate_distance_field(space, boundaries_data, wall_marker, custom_dx, relaxation=0.01):
    """Compute distance-to-wall field y with y=0 on no-slip walls."""
    wall_bc = DirichletBC(space, Constant(0.0), boundaries_data, wall_marker)

    y = Function(space)
    dy = TrialFunction(space)
    z = TestFunction(space)

    # Linear initialization
    linear_problem = inner(grad(dy), grad(z)) * custom_dx - Constant(1.0) * z * custom_dx
    solve(lhs(linear_problem) == rhs(linear_problem), y, [wall_bc])

    # Smoothed Eikonal solve
    F = (
        sqrt(inner(grad(y), grad(y)) + DOLFIN_EPS) * z * custom_dx
        - Constant(1.0) * z * custom_dx
        + Constant(relaxation) * inner(grad(y), grad(z)) * custom_dx
    )
    problem = NonlinearVariationalProblem(F, y, bcs=[wall_bc], J=derivative(F, y))
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
mesh = Mesh(RectangleMesh(MPI.comm_world, Point(0.0, 0.0), Point(L, L), int(N), int(N), "crossed"))

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

inlet_y_min, inlet_y_max, outlet_x_min, outlet_x_max = compute_port_extents(
    L, INLET_TOP_OFFSET, INLET_WIDTH, OUTLET_RIGHT_OFFSET, OUTLET_WIDTH
)
mark = MARK
boundaries = mark_pipe_bend_boundaries(
    mesh, L, TOL, inlet_y_min, inlet_y_max, outlet_x_min, outlet_x_max
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

u_noslip = Constant((0.0, 0.0))

nu_tilde_inlet_bc_value = Constant(0.0)
nu_tilde_wall_bc_value = Constant(0.0)
bcnt_inlet_turb = DirichletBC(TurbulenceSpace, nu_tilde_inlet_bc_value, boundaries, mark["inlet"])
bcnt_walls_turb = DirichletBC(TurbulenceSpace, nu_tilde_wall_bc_value, boundaries, mark["walls"])
bcn_turbulence = [bcnt_inlet_turb, bcnt_walls_turb]

bcu_walls_adj_frozen = DirichletBC(FrozenUPSpace.sub(0), u_noslip, boundaries, mark["walls"])
bcu_inlet_adj_frozen = DirichletBC(FrozenUPSpace.sub(0), u_noslip, boundaries, mark["inlet"])
bcu_outlet_adj_frozen = DirichletBC(FrozenUPSpace.sub(0), u_noslip, boundaries, mark["outlet"])
bcp_pin_adj_frozen = DirichletBC(FrozenUPSpace.sub(1), Constant(0.0), "near(x[0], 0.0) && near(x[1], 0.0)", "pointwise")
bc_NS_adj_frozen = [bcu_walls_adj_frozen, bcu_inlet_adj_frozen, bcu_outlet_adj_frozen, bcp_pin_adj_frozen]

bcu_walls_frozen = DirichletBC(FrozenUPSpace.sub(0), u_noslip, boundaries, mark["walls"])
bcu_inlet_frozen = DirichletBC(FrozenUPSpace.sub(0), u_inlet, boundaries, mark["inlet"])
bcu_outlet_frozen = DirichletBC(FrozenUPSpace.sub(0), u_outlet, boundaries, mark["outlet"])
bcp_pin_frozen = DirichletBC(FrozenUPSpace.sub(1), Constant(0.0), "near(x[0], 0.0) && near(x[1], 0.0)", "pointwise")
bc_NS_frozen = [bcu_walls_frozen, bcu_inlet_frozen, bcu_outlet_frozen, bcp_pin_frozen]

wall_distance = calculate_distance_field(TurbulenceSpace, boundaries, mark["walls"], dx, SA_DISTANCE_RELAXATION)
nu_laminar = Constant(MU_FLUID_VALUE / RHO_FLUID_VALUE)
sa_model = SpalartAllmarasSteadyState(
    TurbulenceSpace,
    bcn_turbulence,
    SA_NU_TILDE_INLET,
    nu_laminar,
    Constant((0.0, 0.0)),
    dx,
    ds,
    wall_distance,
)


# ------------------------------------------------------------
# Design filter
# ------------------------------------------------------------
r_filter = L * 2.0 / float(N)
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

bcu_walls_stokes = DirichletBC(StokesSpace.sub(0), u_noslip, boundaries, mark["walls"])
bcu_inlet_stokes = DirichletBC(StokesSpace.sub(0), u_inlet, boundaries, mark["inlet"])
bcu_outlet_stokes = DirichletBC(StokesSpace.sub(0), u_outlet, boundaries, mark["outlet"])
bcp_pin_stokes = DirichletBC(StokesSpace.sub(1), Constant(0.0), "near(x[0], 0.0) && near(x[1], 0.0)", "pointwise")
bc_stokes = [bcu_walls_stokes, bcu_inlet_stokes, bcu_outlet_stokes, bcp_pin_stokes]


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

    nu_guess_expr = Constant(SA_NU_TILDE_INLET) * wall_distance / (wall_distance + Constant(SA_INIT_WALL_DIST_SCALE))
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


# ------------------------------------------------------------
# Output setup
# ------------------------------------------------------------
results_root = os.path.join(THIS_DIR, "PipeBendTO_Results_Turbulent")
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
        u_inlet.u_max = ramp * U_MAX_INLET
        u_outlet.u_max = ramp * U_MAX_OUTLET
        nu_tilde_inlet_bc_value.assign(ramp * SA_NU_TILDE_INLET)

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
            solver_fwd_frozen = build_forward_solver(forward_form_frozen, w_state_frozen, bc_NS_frozen)
            try:
                solver_fwd_frozen.solve()
            except RuntimeError:
                initialize_frozen_forward_guess_with_stokes()
                solver_fwd_frozen.solve()

            velocity_for_sa = w_state_frozen.sub(0, deepcopy=True)
            sa_model.construct_forms(velocity_for_sa)
            sa_model.solve_turbulence_model()
            sa_model.update_variables(relaxation=NUT_RELAXATION_FACTOR)
            nu_tilde_frozen.assign(sa_model.nu_tilde0)
            enforce_scalar_floor(nu_tilde_frozen, SA_NU_TILDE_FLOOR)
            sa_model.nu_tilde0.assign(nu_tilde_frozen)
            sa_model.nu_tilde1.assign(nu_tilde_frozen)

        # Re-solve momentum with the updated frozen turbulence field.
        solver_fwd_frozen = build_forward_solver(forward_form_frozen, w_state_frozen, bc_NS_frozen)
        try:
            solver_fwd_frozen.solve()
        except RuntimeError:
            initialize_frozen_forward_guess_with_stokes()
            solver_fwd_frozen.solve()

        # Adjoint solve
        print("Starting adjoint SNES solve")
        jac_adj = derivative(adjoint_form_frozen, w_adj_frozen)
        problem_adj = NonlinearVariationalProblem(adjoint_form_frozen, w_adj_frozen, bc_NS_adj_frozen, jac_adj)

        solver_adj = NonlinearVariationalSolver(problem_adj)
        solver_adj.parameters["nonlinear_solver"] = "snes"
        solver_adj.parameters["snes_solver"]["linear_solver"] = SNES_LINEAR_SOLVER
        solver_adj.parameters["snes_solver"]["method"] = "newtonls"
        solver_adj.parameters["snes_solver"]["line_search"] = "bt"
        solver_adj.parameters["snes_solver"]["relative_tolerance"] = ADJOINT_SNES_RTOL
        solver_adj.parameters["snes_solver"]["absolute_tolerance"] = ADJOINT_SNES_ATOL
        solver_adj.parameters["snes_solver"]["maximum_iterations"] = 200
        solver_adj.parameters["snes_solver"]["error_on_nonconvergence"] = True
        solver_adj.solve()

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
