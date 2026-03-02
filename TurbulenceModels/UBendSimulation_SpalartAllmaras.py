from dolfin import *
from Utilities import *
from Configs.ConfigUBend_SpalartAllmaras import *
from TurbulenceModel_SpalartAllmaras import SpalartAllmarasTransient as SpalartAllmaras
import os
import time

parameters["std_out_all_processes"] = False
IS_ROOT = True


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
        if IS_ROOT:
            print(f'Warm-start {label} skipped (file not found): {path}')
        return False

    try:
        loaded = load_H5_files(space, path)
    except Exception as exc:
        if IS_ROOT:
            print(f'Warm-start {label} skipped (failed to load {path}): {type(exc).__name__}: {exc}')
        return False
    target_function.assign(loaded)
    if IS_ROOT:
        print(f'Warm-start loaded for {label}: {path}')
    return True


def _same_path(path_a, path_b):
    if not path_a or not path_b:
        return False
    return os.path.abspath(os.path.normpath(path_a)) == os.path.abspath(os.path.normpath(path_b))

# ---------------------------------------
# Warm-start with different meshes
# ---------------------------------------

def _try_transfer_warm_start_between_meshes(target_function, target_space, source_space, path, label):
    """Load a field on source_space and transfer it onto target_space."""
    if not path:
        return False
    if not os.path.exists(path):
        if IS_ROOT:
            print(f'Warm-start {label} skipped (file not found): {path}')
        return False

    try:
        source_function = load_H5_files(source_space, path)
    except Exception as exc:
        if IS_ROOT:
            print(f'Warm-start {label} skipped (failed to load source {path}): {type(exc).__name__}: {exc}')
        return False

    # Cross-mesh interpolation can fail on boundary points due to tiny geometric
    # mismatches/tolerances. Allow extrapolation on the source field so points
    # that are numerically just outside still evaluate.
    try:
        source_function.set_allow_extrapolation(True)
    except Exception:
        pass

    transfer_method = None
    transferred = None
    lagrange_error = None
    interpolate_error = None
    try:
        transferred = Function(target_space)
        LagrangeInterpolator.interpolate(transferred, source_function)
        transfer_method = 'LagrangeInterpolator'
    except Exception as exc:
        lagrange_error = exc
        try:
            transferred = interpolate(source_function, target_space)
            transfer_method = 'interpolate'
        except Exception as exc:
            interpolate_error = exc
            try:
                # Fallback for cases where direct interpolation between meshes fails.
                transferred = project(source_function, target_space)
                transfer_method = 'project'
            except Exception as exc_project:
                if IS_ROOT:
                    print(
                        f'Warm-start {label} skipped (failed to transfer from source mesh): '
                        f'LagrangeInterpolator -> {type(lagrange_error).__name__}: {lagrange_error}; '
                        f'interpolate -> {type(interpolate_error).__name__}: {interpolate_error}; '
                        f'project -> {type(exc_project).__name__}: {exc_project}'
                    )
                return False

    target_function.assign(transferred)
    if IS_ROOT:
        print(f'Warm-start loaded for {label} from other mesh via {transfer_method}: {path}')
    return True




# Use SA-specific parameters from the config file if available
if 'simulation_prm_SA' in globals():
    simulation_prm = simulation_prm_SA
if 'saving_directory_SA' in globals():
    saving_directory = saving_directory_SA

# Load mesh 
[mesh, marked_facets] = load_mesh_from_file(mesh_files['MESH_DIRECTORY'], mesh_files['FACET_DIRECTORY'])
IS_ROOT = (MPI.COMM_WORLD.Get_rank() == 0)

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
wall_distance_method = simulation_prm.get('WALL_DISTANCE_METHOD', 'RelaxedWallEikonal')
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


# ---------------------------------------------
# Warm-start 
# ---------------------------------------------

# Optional warm-start from saved H5 fields.
# If the warm-start source mesh differs from the active mesh, fields are
# transferred via interpolation/projection onto the active function spaces.
if simulation_prm.get('WARM_START_ENABLED', False):
    warm_start_source_mesh_xdmf = simulation_prm.get('WARM_START_SOURCE_MESH_XDMF', mesh_files['MESH_DIRECTORY'])
    warm_start_source_facet_xdmf = simulation_prm.get('WARM_START_SOURCE_FACET_XDMF', mesh_files['FACET_DIRECTORY'])
    warm_start_same_mesh = _same_path(warm_start_source_mesh_xdmf, mesh_files['MESH_DIRECTORY'])

    if warm_start_same_mesh:
        _try_load_warm_start(u0, V, simulation_prm.get('WARM_START_U_H5', None), 'u0')
        _try_load_warm_start(u1, V, simulation_prm.get('WARM_START_U_H5', None), 'u1')
        _try_load_warm_start(p0, Q, simulation_prm.get('WARM_START_P_H5', None), 'p0')
        _try_load_warm_start(p1, Q, simulation_prm.get('WARM_START_P_H5', None), 'p1')
        nu_tilde_path = simulation_prm.get('WARM_START_NU_TILDE_H5', None)
        if _try_load_warm_start(turbulence_model.nu_tilde0, K, nu_tilde_path, 'nu_tilde0'):
            turbulence_model.nu_tilde1.assign(turbulence_model.nu_tilde0)
    else:
        if IS_ROOT:
            print(
                'Warm-start source mesh differs from active mesh; '
                'loading source fields and transferring to active mesh.'
            )
            print(f'  source mesh: {warm_start_source_mesh_xdmf}')
            print(f'  active mesh: {mesh_files["MESH_DIRECTORY"]}')

        source_spaces = None
        try:
            source_mesh, _ = load_mesh_from_file(warm_start_source_mesh_xdmf, warm_start_source_facet_xdmf)
            source_spaces = {
                'V': VectorFunctionSpace(source_mesh, "CG", 2),
                'Q': FunctionSpace(source_mesh, "CG", 1),
                'K': FunctionSpace(source_mesh, "CG", 1),
            }
        except Exception as exc:
            if IS_ROOT:
                print(
                    f'Cross-mesh warm-start skipped (failed to load source mesh {warm_start_source_mesh_xdmf}): '
                    f'{type(exc).__name__}: {exc}'
                )

        if source_spaces is not None:
            _try_transfer_warm_start_between_meshes(
                u0, V, source_spaces['V'], simulation_prm.get('WARM_START_U_H5', None), 'u0'
            )
            _try_transfer_warm_start_between_meshes(
                u1, V, source_spaces['V'], simulation_prm.get('WARM_START_U_H5', None), 'u1'
            )
            _try_transfer_warm_start_between_meshes(
                p0, Q, source_spaces['Q'], simulation_prm.get('WARM_START_P_H5', None), 'p0'
            )
            _try_transfer_warm_start_between_meshes(
                p1, Q, source_spaces['Q'], simulation_prm.get('WARM_START_P_H5', None), 'p1'
            )
            nu_tilde_path = simulation_prm.get('WARM_START_NU_TILDE_H5', None)
            if _try_transfer_warm_start_between_meshes(
                turbulence_model.nu_tilde0, K, source_spaces['K'], nu_tilde_path, 'nu_tilde0'
            ):
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
SA_INNER_ITERS_MID = simulation_prm.get('SA_INNER_ITERS_MID', None)
if SA_INNER_ITERS_MID is not None:
    SA_INNER_ITERS_MID = int(SA_INNER_ITERS_MID)
    if not (SA_INNER_ITERS_MAX > SA_INNER_ITERS_MID > SA_INNER_ITERS_MIN):
        SA_INNER_ITERS_MID = None
SA_INNER_ITERS_REDUCE_NT_FACTOR = float(simulation_prm.get('SA_INNER_ITERS_REDUCE_NU_TILDE_FACTOR', 0.1))
SA_INNER_ITERS_REDUCE_NT_FACTOR_FINAL = float(
    simulation_prm.get('SA_INNER_ITERS_REDUCE_NU_TILDE_FACTOR_FINAL', SA_INNER_ITERS_REDUCE_NT_FACTOR)
)
SA_INNER_ITERS_REDUCE_STREAK = max(1, int(simulation_prm.get('SA_INNER_ITERS_REDUCE_STREAK', 20)))
SA_INNER_ITERS_REDUCE_STREAK_FINAL = max(
    1, int(simulation_prm.get('SA_INNER_ITERS_REDUCE_STREAK_FINAL', SA_INNER_ITERS_REDUCE_STREAK))
)

sa_inner_reduction_stages = []
if SA_INNER_ITERS_MIN < SA_INNER_ITERS_MAX:
    if SA_INNER_ITERS_MID is not None:
        sa_inner_reduction_stages.append(
            (SA_INNER_ITERS_MID, SA_INNER_ITERS_REDUCE_NT_FACTOR, SA_INNER_ITERS_REDUCE_STREAK)
        )
        sa_inner_reduction_stages.append(
            (SA_INNER_ITERS_MIN, SA_INNER_ITERS_REDUCE_NT_FACTOR_FINAL, SA_INNER_ITERS_REDUCE_STREAK_FINAL)
        )
    else:
        sa_inner_reduction_stages.append(
            (SA_INNER_ITERS_MIN, SA_INNER_ITERS_REDUCE_NT_FACTOR, SA_INNER_ITERS_REDUCE_STREAK)
        )

# Field-specific convergence tolerances (defaults preserve old behavior)
TOLERANCE_GLOBAL = float(simulation_prm['TOLERANCE'])
TOLERANCE_U = float(simulation_prm.get('TOLERANCE_U', TOLERANCE_GLOBAL))
TOLERANCE_P = float(simulation_prm.get('TOLERANCE_P', TOLERANCE_GLOBAL))
TOLERANCE_NU_TILDE = float(simulation_prm.get('TOLERANCE_NU_TILDE', TOLERANCE_GLOBAL))
MIN_OUTER_ITERS = max(0, int(simulation_prm.get('MIN_OUTER_ITERS', 0)))
MIN_PSEUDO_TIME = max(0.0, float(simulation_prm.get('MIN_PSEUDO_TIME', 0.0)))
RUNTIME_WRITE_INTERVAL = max(0, int(simulation_prm.get('RUNTIME_WRITE_INTERVAL', 0)))
RUNTIME_WRITE_PVD = bool(simulation_prm.get('RUNTIME_WRITE_PVD', True))
RUNTIME_WRITE_RESIDUALS = bool(simulation_prm.get('RUNTIME_WRITE_RESIDUALS', True))

# Main loop
residuals = {key: [] for key in ['u', 'p', 'nu_tilde']}
start_time = time.time()
nu_tilde0_outer_prev = Function(K)
sa_inner_iters_current = SA_INNER_ITERS_MAX
sa_inner_reduce_stage_index = 0
sa_inner_reduce_counter = 0
pseudo_time = 0.0
interrupted = False
last_completed_iter = 0
runtime_pvd_files = {}
if post_processing.get('SAVE', False) and RUNTIME_WRITE_INTERVAL > 0 and RUNTIME_WRITE_PVD:
    runtime_pvd_files = {
        'u': File(saving_directory['PVD_FILES'] + 'u_runtime.pvd'),
        'p': File(saving_directory['PVD_FILES'] + 'p_runtime.pvd'),
        'nu_tilde': File(saving_directory['PVD_FILES'] + 'nu_tilde_runtime.pvd'),
    }
try:
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
        pseudo_time += float(dt)
        converged_fields = (
            errors[0] <= TOLERANCE_U and
            errors[1] <= TOLERANCE_P and
            errors[2] <= TOLERANCE_NU_TILDE
        )
        convergence_gate = ((iter + 1) >= MIN_OUTER_ITERS and pseudo_time >= MIN_PSEUDO_TIME)
        break_flag = converged_fields and convergence_gate

        # One-way staged reduction of SA inner iterations once nu_tilde is consistently
        # in the convergence tail. This preserves the early robust phase and speeds up the tail.
        if sa_inner_reduce_stage_index < len(sa_inner_reduction_stages):
            next_sa_inner_iters, next_nt_factor, next_reduce_streak = sa_inner_reduction_stages[sa_inner_reduce_stage_index]
            if errors[2] <= next_nt_factor * TOLERANCE_NU_TILDE:
                sa_inner_reduce_counter += 1
            else:
                sa_inner_reduce_counter = 0

            if sa_inner_reduce_counter >= next_reduce_streak:
                prev_sa_inner_iters = sa_inner_iters_current
                sa_inner_iters_current = next_sa_inner_iters
                sa_inner_reduce_stage_index += 1
                sa_inner_reduce_counter = 0
                if IS_ROOT:
                    print(
                        f'Reducing SA inner iterations from {prev_sa_inner_iters} to {sa_inner_iters_current} '
                        f'after {next_reduce_streak} iterations with '
                        f'|nu_tilde1-nu_tilde0| <= {next_nt_factor:.2e} * tol_nu_tilde.'
                    )

        # Update residuals and print summary
        if IS_ROOT:
            gate_msg = ''
            if not convergence_gate and (MIN_OUTER_ITERS > 0 or MIN_PSEUDO_TIME > 0.0):
                gate_msg = f' [gate: iter>={MIN_OUTER_ITERS}, t>={MIN_PSEUDO_TIME:.2e}; t={pseudo_time:.2e}]'
            print(f'iter: {iter+1} ({time.time() - start_time:.2f}s, dt = {float(dt):.2e}s, t = {pseudo_time:.2e}s, sa_inner = {sa_inner_iters_this_iter}) ----- L2 norms: '
                  f'|u1-u0| = {errors[0]:.2e}, |p1-p0| = {errors[1]:.2e}, '
                  f'|nu_tilde1-nu_tilde0| = {errors[2]:.2e} '
                  f'(req: u < {TOLERANCE_U:.2e}, p < {TOLERANCE_P:.2e}, nu_tilde < {TOLERANCE_NU_TILDE:.2e}){gate_msg}')

        for key, error in zip(residuals.keys(), errors):
            residuals[key].append(error)

        # Runtime snapshots for live monitoring (ParaView + residual tails).
        if post_processing.get('SAVE', False) and RUNTIME_WRITE_INTERVAL > 0:
            if ((iter + 1) % RUNTIME_WRITE_INTERVAL == 0) or break_flag:
                if runtime_pvd_files:
                    runtime_pvd_files['u'] << (u1, pseudo_time)
                    runtime_pvd_files['p'] << (p1, pseudo_time)
                    runtime_pvd_files['nu_tilde'] << (turbulence_model.nu_tilde1, pseudo_time)
                if RUNTIME_WRITE_RESIDUALS:
                    for (key, values) in residuals.items():
                        save_list(values, saving_directory['RESIDUALS'] + key + '.txt')
                if IS_ROOT:
                    print(
                        f'Runtime snapshot written at iter {iter+1} '
                        f'(t = {pseudo_time:.2e}s).'
                    )

        # Update variables for next iteration
        u0.assign((1.0 - U_RELAXATION_FACTOR) * u0 + U_RELAXATION_FACTOR * u1)
        p0.assign(p1)
        turbulence_model.update_variables(relaxation=NUT_RELAXATION_FACTOR)
        last_completed_iter = iter + 1

        # Check for convergence
        if break_flag:
            if IS_ROOT:
                print(f'Simulation converged in {iter+1} iterations ({time.time() - start_time:.2f} seconds, pseudo-time = {pseudo_time:.2e}s)')
            break
except KeyboardInterrupt:
    interrupted = True
    if IS_ROOT:
        print(
            f'KeyboardInterrupt received after {last_completed_iter} completed iterations '
            f'(pseudo-time = {pseudo_time:.2e}s). Saving current state...'
        )

solutions = {'u':u1, 'p':p1, 'nu_tilde':turbulence_model.nu_tilde1}

# Visualize
if post_processing['PLOT']==True and not interrupted:
    visualize_functions(solutions)
    visualize_convergence(residuals)

# Save results and residuals
if post_processing['SAVE']==True:
    for (key, f) in solutions.items():
        save_pvd_file(f, saving_directory['PVD_FILES'] + key + '.pvd')
        save_h5_file( f, saving_directory['H5_FILES']  + key + '.h5')

    for (key, f) in residuals.items():
        save_list(f, saving_directory['RESIDUALS'] + key + '.txt')

    if IS_ROOT and interrupted:
        print('Interrupted run state saved to PVD/H5/residual files.')
