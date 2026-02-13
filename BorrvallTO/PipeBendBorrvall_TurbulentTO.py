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
MU_FLUID_VALUE = 1.0e-4 # dynamic viscosity 
RHO_FLUID_VALUE = 1.0 # mass density
U_MAX_INLET = 1.0
U_MAX_OUTLET = 1.0

# Spalart-Allmaras settings
SA_NU_TILDE_INLET = 1.0e-3 # SA_NU_TILDE_INLET about 10*nu = 10 * MU_FLUID_VALUE / RHO_FLUID_VALUE
SA_DISTANCE_RELAXATION = 0.01


# Topology optimization settings
VOL_FRAC = 0.50
MAX_INNER_ITERATIONS = 100
OBJECTIVE_CONVERGENCE_TOL = 1e-3
OBJECTIVE_STREAK_TO_STOP = 5

Q_PENAL_SCHEDULE = [0.005, 0.01, 0.03, 0.05, 0.1]  # continuation schedule
MOVE_LIMIT = 0.02
SNES_LINEAR_SOLVER = "mumps"  # use "mumps" if available in your PETSc/FEniCS build
INLET_RAMP_STEPS = 60

BETA_PROJ = Constant(0.1)
ETA_I = 0.50
QUADRATURE_DEGREE = 6

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


def sa_min(a, b):
    return (a + b - abs(a - b)) / Constant(2.0)


def sa_turbulent_viscosity(state_nu_tilde):
    nu_lam = mu_fluid / rho_fluid
    chi = state_nu_tilde / (nu_lam + DOLFIN_EPS)
    f_v1 = chi**3 / (chi**3 + Constant(7.1) ** 3)
    nu_t_raw = state_nu_tilde * f_v1
    return 0.5 * (nu_t_raw + abs(nu_t_raw))


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


def build_state_form(
    state_u,
    state_p,
    state_nu_tilde,
    adj_u,
    adj_p,
    adj_nu_tilde,
    rho_eff,
    custom_dx,
    wall_distance=None,
):
    """Build the scalar UFL state form used in forward and adjoint derivations."""
    if wall_distance is None:
        raise ValueError("wall_distance is required.")

    nu_lam = mu_fluid / rho_fluid
    sigma = Constant(2.0 / 3.0)
    cb1 = Constant(0.1355)
    cb2 = Constant(0.622)
    kappa = Constant(0.41)
    cw2 = Constant(0.3)
    cw3 = Constant(2.0)
    cw1 = cb1 / kappa**2 + (Constant(1.0) + cb2) / sigma

    # SA closure quantities
    chi = state_nu_tilde / (nu_lam + DOLFIN_EPS)
    f_v1 = chi**3 / (chi**3 + Constant(7.1) ** 3)
    f_v2 = Constant(1.0) - chi / (Constant(1.0) + chi * f_v1)

    S_sq = Constant(2.0) * inner(sym(nabla_grad(state_u)), sym(nabla_grad(state_u)))
    S = sqrt(S_sq + DOLFIN_EPS)
    y_safe = wall_distance + DOLFIN_EPS

    S_tilde = S + state_nu_tilde / (kappa**2 * y_safe**2) * f_v2
    r_arg = state_nu_tilde / (S_tilde * kappa**2 * y_safe**2 + DOLFIN_EPS)
    r = sa_min(r_arg, Constant(10.0))

    g = r + cw2 * (r**6 - r)
    f_w = g * ((Constant(1.0) + cw3**6) / (g**6 + cw3**6)) ** (Constant(1.0) / Constant(6.0))

    nu_t = state_nu_tilde * f_v1
    nu_t_positive = 0.5 * (nu_t + abs(nu_t))
    mu_effective = mu_fluid + rho_fluid * nu_t_positive

    prod_nt = cb1 * S_tilde * state_nu_tilde
    react_nt = cw1 * f_w * (state_nu_tilde / y_safe**2)
    cross_diff_nt = (cb2 / sigma) * inner(nabla_grad(state_nu_tilde), nabla_grad(state_nu_tilde))
    source_nt = prod_nt + cross_diff_nt

    momentum = (
        rho_fluid * inner(dot(state_u, nabla_grad(state_u)), adj_u)
        + mu_effective * inner(grad(state_u), grad(adj_u))
        + inner(grad(state_p), adj_u)
        + inner(div(state_u), adj_p)
        + alpha(rho_eff) * inner(state_u, adj_u)
    ) * custom_dx

    turbulence_transport = (
        dot(state_u, nabla_grad(state_nu_tilde)) * adj_nu_tilde
        + inner(((nu_lam + state_nu_tilde) / sigma) * grad(state_nu_tilde), grad(adj_nu_tilde))
        + react_nt * state_nu_tilde * adj_nu_tilde
        - source_nt * adj_nu_tilde
    ) * custom_dx

    return momentum + turbulence_transport


# ------------------------------------------------------------
# Mesh, function spaces, and boundaries
# ------------------------------------------------------------
mesh = Mesh(RectangleMesh(MPI.comm_world, Point(0.0, 0.0), Point(L, L), int(N), int(N), "crossed"))

U_h = VectorElement("CG", mesh.ufl_cell(), 2)
P_h = FiniteElement("CG", mesh.ufl_cell(), 1)
T_h = FiniteElement("CG", mesh.ufl_cell(), 1)
A_h = FiniteElement("DG", mesh.ufl_cell(), 0)

FlowElement = MixedElement([U_h, P_h, T_h])
FlowSpace = FunctionSpace(mesh, FlowElement)
FlowSpaceAdj = FunctionSpace(mesh, FlowElement)
TurbulenceSpace = FunctionSpace(mesh, T_h)

DensitySpace = FunctionSpace(mesh, A_h)

w_fwd = Function(FlowSpace)
w_adj = Function(FlowSpaceAdj)

(u, p, nu_tilde) = split(w_fwd)
(v, q, psi) = split(w_adj)

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

bcu_walls_adj = DirichletBC(FlowSpaceAdj.sub(0), u_noslip, boundaries, mark["walls"])
bcu_inlet_adj = DirichletBC(FlowSpaceAdj.sub(0), u_inlet, boundaries, mark["inlet"])
bcu_outlet_adj = DirichletBC(FlowSpaceAdj.sub(0), u_outlet, boundaries, mark["outlet"])
bcp_pin_adj = DirichletBC(FlowSpaceAdj.sub(1), Constant(0.0), "near(x[0], 0.0) && near(x[1], 0.0)", "pointwise")

nu_tilde_inlet_bc_value = Constant(SA_NU_TILDE_INLET)
nu_tilde_wall_bc_value = Constant(0.0)

bcnt_inlet = DirichletBC(FlowSpace.sub(2), nu_tilde_inlet_bc_value, boundaries, mark["inlet"])
bcnt_walls = DirichletBC(FlowSpace.sub(2), nu_tilde_wall_bc_value, boundaries, mark["walls"])

bcnt_inlet_adj = DirichletBC(FlowSpaceAdj.sub(2), nu_tilde_inlet_bc_value, boundaries, mark["inlet"])
bcnt_walls_adj = DirichletBC(FlowSpaceAdj.sub(2), nu_tilde_wall_bc_value, boundaries, mark["walls"])

bc_NS = [bcu_walls, bcu_inlet, bcu_outlet, bcp_pin, bcnt_inlet, bcnt_walls]
bc_NS_adj = [bcu_walls_adj, bcu_inlet_adj, bcu_outlet_adj, bcp_pin_adj, bcnt_inlet_adj, bcnt_walls_adj]

wall_distance = calculate_distance_field(TurbulenceSpace, boundaries, mark["walls"], dx, SA_DISTANCE_RELAXATION)


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

mu_effective_obj = mu_fluid + rho_fluid * sa_turbulent_viscosity(nu_tilde)
ObjFunctional = AreaOfInterest * (
    0.5 * mu_effective_obj * inner(sym(nabla_grad(u)), sym(nabla_grad(u)))
    + alpha(rho_effective) * inner(u, u)
) * dx

state_form = build_state_form(
    u,
    p,
    nu_tilde,
    v,
    q,
    psi,
    rho_effective,
    dx,
    wall_distance,
)
lagrangian_form = ObjFunctional + state_form

forward_form = derivative(state_form, w_adj, TestFunction(FlowSpace))
adjoint_form = derivative(lagrangian_form, w_fwd, TestFunction(FlowSpaceAdj))

ddx = derivative(lagrangian_form, rho_f)
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


def initialize_forward_guess_with_stokes():
    """Initialize w_fwd with a linear Stokes-Brinkman solve for robust Newton startup."""
    A = assemble(a_stokes)
    b = assemble(l_stokes)
    for bc in bc_stokes:
        bc.apply(A, b)
    w_stokes = Function(StokesSpace)
    solve(A, w_stokes.vector(), b, SNES_LINEAR_SOLVER)

    u_guess, p_guess = w_stokes.split(deepcopy=True)
    nu_guess = interpolate(Constant(SA_NU_TILDE_INLET), TurbulenceSpace)
    mixed_assigner = FunctionAssigner(
        FlowSpace,
        [u_guess.function_space(), p_guess.function_space(), nu_guess.function_space()],
    )
    mixed_assigner.assign(w_fwd, [u_guess, p_guess, nu_guess])


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


# ------------------------------------------------------------
# Optimization loop
# ------------------------------------------------------------
for q_val in Q_PENAL_SCHEDULE:
    q_penal.assign(q_val)
    inner_count = 0
    convergence_history = 0
    objective_converged = False

    while inner_count <= MAX_INNER_ITERATIONS and not objective_converged:
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
        solver_fwd.parameters["snes_solver"]["relative_tolerance"] = 1.0e-6
        solver_fwd.parameters["snes_solver"]["absolute_tolerance"] = 1.0e-9
        solver_fwd.parameters["snes_solver"]["maximum_iterations"] = 200
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
        solver_adj.parameters["snes_solver"]["relative_tolerance"] = 1.0e-6
        solver_adj.parameters["snes_solver"]["absolute_tolerance"] = 1.0e-9
        solver_adj.parameters["snes_solver"]["maximum_iterations"] = 200
        solver_adj.parameters["snes_solver"]["error_on_nonconvergence"] = True
        solver_adj.solve()

        u_out << w_fwd.sub(0)
        p_out << w_fwd.sub(1)
        nu_tilde_out << w_fwd.sub(2)

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
            MOVE_LIMIT,
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
            "q = {:.3f}, iter = {:03d}, J = {:.4e}, obj_conv = {:.3e}, vol = {:.4f}".format(
                q_val, inner_count, f0val, obj_conv, vol_fraction_now
            )
        )

        inner_count += 1
        iter_count += 1

print("Optimization finished. Results written to {}".format(results_root))
