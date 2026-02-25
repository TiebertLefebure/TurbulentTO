from dolfin import *
from Utilities import *
from Configs.ConfigUBend_SpalartAllmaras import *
from TurbulenceModel_SpalartAllmaras import SpalartAllmarasTransient as SpalartAllmaras
import os
import time


def _l2_norm_diff(f1, f0, dx_measure):
    """True L2 norm of the difference between two FEniCS Functions."""
    diff = f1 - f0
    if f1.ufl_shape == ():
        return sqrt(assemble(diff**2 * dx_measure))
    return sqrt(assemble(dot(diff, diff) * dx_measure))


def _try_load_warm_start(target_function, space, path, label):
    """Load a warm-start field into target_function if the H5 path exists."""
    if not path:
        return False
    if not os.path.exists(path):
        print(f'Warm-start {label} skipped (file not found): {path}')
        return False

    try:
        loaded = load_H5_files(space, path)
    except Exception as exc:
        print(f'Warm-start {label} skipped (failed to load {path}): {type(exc).__name__}: {exc}')
        return False
    target_function.assign(loaded)
    print(f'Warm-start loaded for {label}: {path}')
    return True

# Use SA-specific parameters from the config file if available
if 'simulation_prm_SA' in globals():
    simulation_prm = simulation_prm_SA
if 'saving_directory_SA' in globals():
    saving_directory = saving_directory_SA

# Load mesh 
[mesh, marked_facets] = load_mesh_from_file(mesh_files['MESH_DIRECTORY'], mesh_files['FACET_DIRECTORY'])

# Custom integration measures
quadrature_degree = simulation_prm['QUADRATURE_DEGREE']
dx = Measure("dx", domain=mesh, metadata={"quadrature_degree": quadrature_degree})
ds = Measure("ds", domain=mesh, metadata={"quadrature_degree": quadrature_degree})

# Construct function spaces
V = VectorFunctionSpace(mesh, "CG", 2)       
Q = FunctionSpace(mesh, "CG", 1)                                        
K = FunctionSpace(mesh, "CG", 1)  

# Construct boundary conditions
bcu=[]; bcp=[]; bcn=[]

for boundary_name, markers in boundary_markers.items():
    if markers is None:
        continue  

    for marker in markers:
        for variable, bc_list, function_space in zip(['U','P','NU_TILDE'], [bcu,bcp,bcn], [V,Q,K,K]):
                
            condition_value = boundary_conditions[boundary_name].get(variable)
            if condition_value != None:
                bc_list.append(DirichletBC(function_space, condition_value, marked_facets, marker))

# Initialize constants and expressions
nu = Constant(physical_prm['VISCOSITY'])
force = Constant(physical_prm['FORCE'])
dt = Constant(simulation_prm_SA['STEP_SIZE'])
wall_distance_method = simulation_prm.get('WALL_DISTANCE_METHOD', 'OriginalEikonal')
wall_distance_relax = simulation_prm.get('WALL_DISTANCE_EIKONAL_RELAXATION', 0.01)
wall_distance_sigma_w = simulation_prm.get('WALL_DISTANCE_YOON_SIGMA_W', 0.1)
wall_distance_g0 = simulation_prm.get('WALL_DISTANCE_YOON_G0', 20.0)
wall_distance_g_floor = simulation_prm.get('WALL_DISTANCE_YOON_G_FLOOR', 1.0e-12)
y = calculate_Distance_field(
    K,
    marked_facets,
    boundary_markers['WALLS'],
    wall_distance_relax,
    method=wall_distance_method,
    sigma_w=wall_distance_sigma_w,
    g0=wall_distance_g0,
    g_floor=wall_distance_g_floor,
)

# Initialize functions
u, v, u1, u0 = initialize_functions(V, Constant(initial_conditions['U']))
p, q, p1, p0 = initialize_functions(Q, Constant(initial_conditions['P']))

# Initialize turbulence model
turbulence_model = SpalartAllmaras(K, bcn, initial_conditions['NU_TILDE'],
                            nu, force, dx, ds, dt, y)
turbulence_model.construct_forms(u1)

# Optional warm-start from saved H5 fields (same mesh/function spaces required).
if simulation_prm.get('WARM_START_ENABLED', False):
    _try_load_warm_start(u0, V, simulation_prm.get('WARM_START_U_H5', None), 'u0')
    _try_load_warm_start(u1, V, simulation_prm.get('WARM_START_U_H5', None), 'u1')
    _try_load_warm_start(p0, Q, simulation_prm.get('WARM_START_P_H5', None), 'p0')
    _try_load_warm_start(p1, Q, simulation_prm.get('WARM_START_P_H5', None), 'p1')
    nu_tilde_path = simulation_prm.get('WARM_START_NU_TILDE_H5', None)
    if _try_load_warm_start(turbulence_model.nu_tilde0, K, nu_tilde_path, 'nu_tilde0'):
        turbulence_model.nu_tilde1.assign(turbulence_model.nu_tilde0)

# Construct RANS forms
# SUPG (Streamline Upwind Petrov-Galerkin) Stabilization
h = CellDiameter(mesh)
u_mag = sqrt(dot(u0, u0) + 1e-10)
tau = h / (2.0 * u_mag)
residual = (u - u0) / dt + dot(u0, nabla_grad(u)) - force
F_supg = inner(tau * dot(u0, nabla_grad(v)), residual) * dx

F1  = dot((u - u0) / dt, v)*dx \
    + dot(dot(u0, nabla_grad(u)), v)*dx \
    + inner((nu + turbulence_model.nu_t) * grad(u), grad(v))*dx \
    - dot(force, v)*dx \
    + F_supg

F2  = dot(grad(p), grad(q))*dx + dot(div(u1) / dt, q)*dx
F3  = dot(u, v)*dx - dot(u1, v)*dx + dt * dot(grad(p1), v)*dx

# Precompute lhs and rhs
a_1, l_1 = lhs(F1), rhs(F1)
a_2, l_2 = lhs(F2), rhs(F2)
a_3, l_3 = lhs(F3), rhs(F3)

# Relaxation factors
U_RELAXATION_FACTOR = simulation_prm.get('U_RELAXATION_FACTOR', 1.0)
NUT_RELAXATION_FACTOR = simulation_prm.get('NUT_RELAXATION_FACTOR', 1.0)
SA_INNER_ITERS_MAX = max(1, int(simulation_prm.get('SA_INNER_ITERS', 1)))
SA_INNER_ITERS_MIN = max(1, int(simulation_prm.get('SA_INNER_ITERS_MIN', SA_INNER_ITERS_MAX)))
SA_INNER_ITERS_MIN = min(SA_INNER_ITERS_MIN, SA_INNER_ITERS_MAX)
SA_INNER_ITERS_REDUCE_NT_FACTOR = float(simulation_prm.get('SA_INNER_ITERS_REDUCE_NU_TILDE_FACTOR', 0.1))
SA_INNER_ITERS_REDUCE_STREAK = max(1, int(simulation_prm.get('SA_INNER_ITERS_REDUCE_STREAK', 20)))

# Field-specific convergence tolerances (defaults preserve old behavior)
TOLERANCE_GLOBAL = float(simulation_prm['TOLERANCE'])
TOLERANCE_U = float(simulation_prm.get('TOLERANCE_U', TOLERANCE_GLOBAL))
TOLERANCE_P = float(simulation_prm.get('TOLERANCE_P', TOLERANCE_GLOBAL))
TOLERANCE_NU_TILDE = float(simulation_prm.get('TOLERANCE_NU_TILDE', TOLERANCE_GLOBAL))

# Main loop
residuals = {key: [] for key in ['u', 'p', 'nu_tilde']}
start_time = time.time()
nu_tilde0_outer_prev = Function(K)
sa_inner_iters_current = SA_INNER_ITERS_MAX
sa_inner_reduced = False
sa_inner_reduce_counter = 0
for iter in range(simulation_prm['MAX_ITERATIONS']):
    # Dynamic time-stepping
    if iter > 0:
        h_x = MaxCellEdgeLength(mesh); h_y = MinCellEdgeLength(mesh)
        step_size = calculate_cfl_time_step(u0, h_x, h_y, simulation_prm['CFL_RELAXATION'], mesh)
        min_step_size = simulation_prm.get('MIN_STEP_SIZE', None)
        max_step_size = simulation_prm.get('MAX_STEP_SIZE', None)
        if min_step_size is not None:
            step_size = max(step_size, min_step_size)
        if max_step_size is not None:
            step_size = min(step_size, max_step_size)
        dt.assign(Constant(step_size))

    # Solve NS
    A_1 = assemble(a_1); b_1 = assemble(l_1)
    [bc.apply(A_1,b_1) for bc in bcu]
    solve(A_1, u1.vector(), b_1, 'mumps')

    A_2 = assemble(a_2); b_2 = assemble(l_2)
    [bc.apply(A_2,b_2) for bc in bcp]
    solve(A_2, p1.vector(), b_2, 'mumps')

    A_3 = assemble(a_3); b_3 = assemble(l_3)
    [bc.apply(A_3,b_3) for bc in bcu]
    solve(A_3, u1.vector(), b_3, 'mumps')

    # Solve SA turbulence model with optional inner iterations so nu_tilde can
    # catch up to the current velocity field before the next momentum solve.
    nu_tilde0_outer_prev.assign(turbulence_model.nu_tilde0)
    sa_inner_iters_this_iter = sa_inner_iters_current
    for sa_iter in range(sa_inner_iters_this_iter):
        turbulence_model.solve_turbulence_model()
        if sa_iter < sa_inner_iters_this_iter - 1:
            turbulence_model.update_variables(relaxation=NUT_RELAXATION_FACTOR)

    errors = [
        _l2_norm_diff(u1, u0, dx),
        _l2_norm_diff(p1, p0, dx),
        _l2_norm_diff(turbulence_model.nu_tilde1, nu_tilde0_outer_prev, dx),
    ]
    break_flag = (
        errors[0] <= TOLERANCE_U and
        errors[1] <= TOLERANCE_P and
        errors[2] <= TOLERANCE_NU_TILDE
    )

    # One-way reduction of SA inner iterations once nu_tilde is consistently in
    # the convergence tail. This cuts runtime without changing the early robust phase.
    if not sa_inner_reduced and SA_INNER_ITERS_MIN < SA_INNER_ITERS_MAX:
        if errors[2] <= SA_INNER_ITERS_REDUCE_NT_FACTOR * TOLERANCE_NU_TILDE:
            sa_inner_reduce_counter += 1
        else:
            sa_inner_reduce_counter = 0

        if sa_inner_reduce_counter >= SA_INNER_ITERS_REDUCE_STREAK:
            sa_inner_iters_current = SA_INNER_ITERS_MIN
            sa_inner_reduced = True
            print(
                f'Reducing SA inner iterations from {SA_INNER_ITERS_MAX} to {SA_INNER_ITERS_MIN} '
                f'after {SA_INNER_ITERS_REDUCE_STREAK} iterations with '
                f'|nu_tilde1-nu_tilde0| <= {SA_INNER_ITERS_REDUCE_NT_FACTOR:.2e} * tol_nu_tilde.'
            )

    # Update residuals and print summary
    print(f'iter: {iter+1} ({time.time() - start_time:.2f}s, dt = {float(dt):.2e}s, sa_inner={sa_inner_iters_this_iter}) --- L2 norms: '
          f'|u1-u0|= {errors[0]:.2e}, |p1-p0|= {errors[1]:.2e}, '
          f'|nu_tilde1-nu_tilde0|= {errors[2]:.2e} '
          f'(req: u<{TOLERANCE_U:.2e}, p<{TOLERANCE_P:.2e}, nu_tilde<{TOLERANCE_NU_TILDE:.2e})')

    for key, error in zip(residuals.keys(), errors):
        residuals[key].append(error)

    # Update variables for next iteration
    u0.assign((1.0 - U_RELAXATION_FACTOR) * u0 + U_RELAXATION_FACTOR * u1)
    p0.assign(p1)
    turbulence_model.update_variables(relaxation=NUT_RELAXATION_FACTOR)

    # Check for convergence
    if break_flag:
        print(f'Simulation converged in {iter+1} iterations ({time.time() - start_time:.2f} seconds)')
        break

solutions = {'u':u1, 'p':p1, 'nu_tilde':turbulence_model.nu_tilde1}

# Visualize
if post_processing['PLOT']==True:
    visualize_functions(solutions)
    visualize_convergence(residuals)

# Save results and residuals
if post_processing['SAVE']==True:
    for (key, f) in solutions.items():
        save_pvd_file(f, saving_directory['PVD_FILES'] + key + '.pvd')
        save_h5_file( f, saving_directory['H5_FILES']  + key + '.h5')

    for (key, f) in residuals.items():
        save_list(f, saving_directory['RESIDUALS'] + key + '.txt')
