from dolfin import *
import numpy as np
import os
import time
try:
    from ufl import tanh
except ModuleNotFoundError:
    from ufl_legacy import tanh

from mma import mmasub
from Utilities_SharedTO import (
    append_optimization_log_entry,
    as_list,
    build_pressure_pin_expression_from_config,
    compute_filter_base_length_from_config,
    create_design_mesh_from_config,
    ensure_clean_dir,
    initialize_optimization_log,
    load_config_module_from_cli,
    reset_vtk_series,
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

# Brinkman penalization constants
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

mark = MARK
boundaries = globals()["mark_boundaries"](mesh)
wall_markers = as_list(mark["walls"])
inlet_markers = as_list(mark["inlet"])
outlet_markers = as_list(mark["outlet"])
density_lower_bound, density_upper_bound = build_density_bounds_from_config()
density_lower_values = density_lower_bound.vector().get_local()
density_upper_values = density_upper_bound.vector().get_local()

dx = Measure("dx", domain=mesh)
ds = Measure("ds", domain=mesh, subdomain_data=boundaries)
dS = Measure("dS", domain=mesh)

inlet_profiles, outlet_profiles = globals()["build_velocity_profile_sets"]()
inlet_profiles = as_list(inlet_profiles)
outlet_profiles = as_list(outlet_profiles)

u_noslip = Constant((0.0, 0.0))

bcu_walls = [DirichletBC(FlowSpace.sub(0), u_noslip, boundaries, m) for m in wall_markers]
bcu_inlet = [DirichletBC(FlowSpace.sub(0), prof, boundaries, m) for prof, m in zip(inlet_profiles, inlet_markers)]
bcu_outlet = [DirichletBC(FlowSpace.sub(0), prof, boundaries, m) for prof, m in zip(outlet_profiles, outlet_markers)]
bc_NS = bcu_walls + bcu_inlet + bcu_outlet
if globals().get("ENABLE_PRESSURE_PIN", True):
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
    bcu_outlet_adj = [DirichletBC(FlowSpaceAdj.sub(0), u_noslip, boundaries, m) for m in outlet_markers]
    bc_NS_adj = bcu_walls_adj + bcu_inlet_adj + bcu_outlet_adj
if globals().get("ENABLE_PRESSURE_PIN", True):
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
    deformation = nabla_grad(u) + nabla_grad(u).T
    dissipation_density = 0.5 * mu_fluid * inner(deformation, deformation)

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


# ------------------------------------------------------------
# Output setup
# ------------------------------------------------------------
results_root = os.path.join(THIS_DIR, globals().get("RESULTS_ROOT_NAME", "LaminarTO_Results"))
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

rho_out = File(reset_vtk_series(os.path.join(rho_dir, "plot_rho.pvd"), COMM))
rhop_out = File(reset_vtk_series(os.path.join(rho_p_dir, "plot_rho_projected.pvd"), COMM))
u_out = File(reset_vtk_series(os.path.join(u_dir, "plot_u.pvd"), COMM))
p_out = File(reset_vtk_series(os.path.join(p_dir, "plot_p.pvd"), COMM))

log_path = os.path.join(results_root, "OptimizationLog.txt")
initialize_optimization_log(log_path)


# ------------------------------------------------------------
# MMA initialization
# ------------------------------------------------------------
initial_density = float(globals().get("INITIAL_DENSITY_VALUE", VOL_FRAC))
assign(rho, interpolate(Constant(initial_density), DensitySpace))
rho.vector().set_local(np.clip(rho.vector().get_local(), density_lower_values, density_upper_values))
rho.vector().apply("insert")

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
rho_f = pde_filter(rho, rho_f)
initialize_forward_guess_with_stokes()


# ------------------------------------------------------------
# Optimization loop
# ------------------------------------------------------------
optimization_start_time = time.perf_counter()
for stage_idx, q_val in enumerate(Q_PENAL_SCHEDULE):
    beta_val = float(BETA_PROJ_SCHEDULE[stage_idx])
    BETA_PROJ.assign(beta_val)
    move_limit_now = MOVE_LIMIT_SCHEDULE[stage_idx]
    q_penal.assign(q_val)
    inner_count = 0
    convergence_history = 0
    objective_converged = False
    root_print("Starting continuation stage {}/{}: q = {:.3f}, beta = {:.2f}, move = {:.4f}".format(
        stage_idx + 1, len(Q_PENAL_SCHEDULE), q_val, beta_val, move_limit_now,
    ))

    while inner_count < MAX_INNER_ITERATIONS and not objective_converged:
        iteration_start_time = time.perf_counter()

        root_print("--- Stage {}/{} | iter {:03d} (global {:03d}) ---".format(
            stage_idx + 1, len(Q_PENAL_SCHEDULE), inner_count, iter_count,
        ))

        # Filter current design
        rho_f = pde_filter(rho, rho_f)
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
        vol_fraction_now = assemble(AreaOfInterest * rho_effective * dx) / volume
        vol_residual_now = float(fval[0, 0]) / max(volume, 1.0e-12)

        df0dx[:, 0] = filtered_gradient.vector()[:]
        dfdx[0, :] = filtered_s_vol.vector()[:]

        # MMA update
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

        iteration_elapsed = time.perf_counter() - iteration_start_time
        optimization_elapsed = time.perf_counter() - optimization_start_time
        root_print("q={:.3f} beta={:.2f} move={:.3f} iter={:03d} iter_time={:.1f}s elapsed={:.1f}s J={:.4e} conv={:.3e} vol={:.4f} streak={}/{}".format(
            q_val, float(BETA_PROJ.values()[0]), move_limit_now,
            inner_count, iteration_elapsed, optimization_elapsed, f0val, obj_conv, vol_fraction_now,
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
            stage_idx + 1, len(Q_PENAL_SCHEDULE), MAX_INNER_ITERATIONS,
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
