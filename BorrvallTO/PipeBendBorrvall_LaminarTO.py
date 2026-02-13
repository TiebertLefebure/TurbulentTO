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


# ------------------------------------------------------------
# User parameters
# ------------------------------------------------------------

L = 1.0
N = 120
TOL = DOLFIN_EPS

# Figure-6 geometry parameters (Borrvall 2003 pipe bend case)
INLET_WIDTH = 0.2
INLET_TOP_OFFSET = 0.2
OUTLET_WIDTH = 0.2
OUTLET_RIGHT_OFFSET = 0.2

# Flow settings
MU_FLUID_VALUE = 1e-3 # dynamic viscosity 
RHO_FLUID_VALUE = 1.0 # mass density
U_MAX_INLET = 1.0
U_MAX_OUTLET = 1.0

### Reynolds number Re = U_MAX_INLET * INLET_WIDTH * RHO_FLUID_VALUE / MU_FLUID_VALUE = 200

# Topology optimization settings
VOL_FRAC = 0.50
MAX_INNER_ITERATIONS = 60  # per continuation stage
OBJECTIVE_CONVERGENCE_TOL = 5e-5 # OBJECTIVE_CONVERGENCE_TOL = 1e-5
OBJECTIVE_STREAK_TO_STOP = 5

Q_PENAL_SCHEDULE = [0.005, 0.01, 0.03, 0.05, 0.1]
MOVE_LIMIT_SCHEDULE = [0.03, 0.03, 0.02, 0.015, 0.01]

SNES_LINEAR_SOLVER = "mumps"  # use "mumps" if available in your PETSc/FEniCS build
INLET_RAMP_STEPS = 40
FILTER_RADIUS_IN_CELLS = 3.0
FORWARD_SNES_RTOL = 5.0e-7
FORWARD_SNES_ATOL = 1.0e-9
ADJOINT_SNES_RTOL = 5.0e-7
ADJOINT_SNES_ATOL = 1.0e-9
SNES_MAX_ITERS = 200

BETA_PROJ = Constant(0.1)
ETA_I = 0.50

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


def between(value, limits, eps=DOLFIN_EPS):
    return (limits[0] - eps <= value) and (value <= limits[1] + eps)


def ensure_clean_dir(path, comm=MPI.comm_world):
    # Avoid MPI races where multiple ranks delete/create the same folder.
    if MPI.rank(comm) == 0:
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)
    MPI.barrier(comm)


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
mesh = Mesh(RectangleMesh(MPI.comm_world, Point(0.0, 0.0), Point(L, L), int(N), int(N), "crossed"))

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

inlet_y_max = L - INLET_TOP_OFFSET
inlet_y_min = inlet_y_max - INLET_WIDTH
outlet_x_max = L - OUTLET_RIGHT_OFFSET
outlet_x_min = outlet_x_max - OUTLET_WIDTH


class Inlet(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 0.0, TOL) and between(x[1], (inlet_y_min, inlet_y_max), TOL)


class Outlet(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], 0.0, TOL) and between(x[0], (outlet_x_min, outlet_x_max), TOL)


class Walls(SubDomain):
    def inside(self, x, on_boundary):
        left_wall_outside_inlet = near(x[0], 0.0, TOL) and not between(x[1], (inlet_y_min, inlet_y_max), TOL)
        bottom_wall_outside_outlet = near(x[1], 0.0, TOL) and not between(x[0], (outlet_x_min, outlet_x_max), TOL)
        right_wall = near(x[0], L, TOL)
        top_wall = near(x[1], L, TOL)
        return on_boundary and (left_wall_outside_inlet or bottom_wall_outside_outlet or right_wall or top_wall)


mark = {"generic": 0, "walls": 1, "inlet": 2, "outlet": 3}
boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
boundaries.set_all(mark["generic"])
Walls().mark(boundaries, mark["walls"])
Inlet().mark(boundaries, mark["inlet"])
Outlet().mark(boundaries, mark["outlet"])

dx = Measure("dx", domain=mesh)
ds = Measure("ds", domain=mesh, subdomain_data=boundaries)
dS = Measure("dS", domain=mesh)

# Inlet profile: +x direction
y_inlet_center = 0.5 * (inlet_y_min + inlet_y_max)
u_inlet = Expression(
    ("u_max * (1 - pow(2.0 * (x[1] - y_c) / width, 2))", "0.0"),
    degree=2,
    u_max=U_MAX_INLET,
    y_c=y_inlet_center,
    width=INLET_WIDTH,
)

# Outlet profile: -y direction
x_outlet_center = 0.5 * (outlet_x_min + outlet_x_max)
u_outlet = Expression(
    ("0.0", "-u_max * (1 - pow(2.0 * (x[0] - x_c) / width, 2))"),
    degree=2,
    u_max=U_MAX_OUTLET,
    x_c=x_outlet_center,
    width=OUTLET_WIDTH,
)

u_noslip = Constant((0.0, 0.0))

bcu_walls = DirichletBC(FlowSpace.sub(0), u_noslip, boundaries, mark["walls"])
bcu_inlet = DirichletBC(FlowSpace.sub(0), u_inlet, boundaries, mark["inlet"])
bcu_outlet = DirichletBC(FlowSpace.sub(0), u_outlet, boundaries, mark["outlet"])
bcp_pin = DirichletBC(FlowSpace.sub(1), Constant(0.0), "near(x[0], 0.0) && near(x[1], 0.0)", "pointwise")
bc_NS = [bcu_walls, bcu_inlet, bcu_outlet, bcp_pin]

bcu_walls_adj = DirichletBC(FlowSpaceAdj.sub(0), u_noslip, boundaries, mark["walls"])
bcu_inlet_adj = DirichletBC(FlowSpaceAdj.sub(0), u_noslip, boundaries, mark["inlet"])
bcu_outlet_adj = DirichletBC(FlowSpaceAdj.sub(0), u_noslip, boundaries, mark["outlet"])
bcp_pin_adj = DirichletBC(FlowSpaceAdj.sub(1), Constant(0.0), "near(x[0], 0.0) && near(x[1], 0.0)", "pointwise")
bc_NS_adj = [bcu_walls_adj, bcu_inlet_adj, bcu_outlet_adj, bcp_pin_adj]


# ------------------------------------------------------------
# Design filter
# ------------------------------------------------------------
r_filter = L * FILTER_RADIUS_IN_CELLS / float(N)
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

ObjFunctional = AreaOfInterest * (
    0.5 * mu_fluid * inner(sym(nabla_grad(u)), sym(nabla_grad(u)))
    + alpha(rho_effective) * inner(u, u)
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
results_root = os.path.join(THIS_DIR, "PipeBendTO_Results_Laminar")
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


# ------------------------------------------------------------
# Optimization loop
# ------------------------------------------------------------
for stage_idx, q_val in enumerate(Q_PENAL_SCHEDULE):
    move_limit_now = MOVE_LIMIT_SCHEDULE[stage_idx]
    q_penal.assign(q_val)
    inner_count = 0
    convergence_history = 0
    objective_converged = False

    while inner_count < MAX_INNER_ITERATIONS and not objective_converged:
        ramp = min(1.0, float(iter_count + 1) / float(max(1, INLET_RAMP_STEPS)))
        u_inlet.u_max = ramp * U_MAX_INLET
        u_outlet.u_max = ramp * U_MAX_OUTLET

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
