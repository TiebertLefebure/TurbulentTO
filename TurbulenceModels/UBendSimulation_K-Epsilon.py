from dolfin import *
from Utilities import *
import importlib.util as _ilu
import os
import time

# importlib is required because hyphenated filenames are not valid Python identifiers
_tm = _ilu.spec_from_file_location("TurbulenceModel_KE", os.path.join(os.path.dirname(os.path.abspath(__file__)), "TurbulenceModel_K-Epsilon.py"))
_tm = _ilu.module_from_spec(_tm); _tm.__spec__.loader.exec_module(_tm)
KEpsilonTransient = _tm.KEpsilonTransient; del _tm

_cfg = _ilu.spec_from_file_location("ConfigUBend_KE", os.path.join(os.path.dirname(os.path.abspath(__file__)), "Configs", "ConfigUBend_K-Epsilon.py"))
_cfg = _ilu.module_from_spec(_cfg); _cfg.__spec__.loader.exec_module(_cfg)
globals().update({k: v for k, v in vars(_cfg).items() if not k.startswith('_')}); del _cfg, _ilu

parameters["std_out_all_processes"] = False
IS_ROOT = True


def _mesh_save_dirs(d, mesh_path):
    mesh_name = os.path.basename(os.path.dirname(os.path.normpath(mesh_path)))
    if not mesh_name:
        return d
    out = {}
    for key, path in d.items():
        path = os.path.normpath(path)
        parent = os.path.dirname(path)
        out[key] = os.path.join(path, '') if os.path.basename(parent) == mesh_name else os.path.join(parent, mesh_name, os.path.basename(path), '')
    return out


def _solve_linear_system(A, x, b, linear_solver, linear_preconditioner):
    if linear_solver in (None, '', 'default'):
        if linear_preconditioner in (None, '', 'default'):
            solve(A, x, b)
        else:
            solve(A, x, b, 'default', linear_preconditioner)
    else:
        if linear_preconditioner in (None, '', 'default'):
            solve(A, x, b, linear_solver)
        else:
            solve(A, x, b, linear_solver, linear_preconditioner)


def _squared_l2_error(f1, f0, dx_measure):
    diff = f1 - f0
    if f1.ufl_shape == ():
        return assemble(diff**2 * dx_measure)
    return assemble(dot(diff, diff) * dx_measure)

saving_directory = _mesh_save_dirs(saving_directory, mesh_files['MESH_DIRECTORY'])
[mesh, marked_facets] = load_mesh_from_file(mesh_files['MESH_DIRECTORY'], mesh_files['FACET_DIRECTORY'])
IS_ROOT = (MPI.COMM_WORLD.Get_rank() == 0)
quadrature_degree = simulation_prm['QUADRATURE_DEGREE']
dx = Measure('dx', domain=mesh, metadata={'quadrature_degree': quadrature_degree})
ds = Measure('ds', domain=mesh, metadata={'quadrature_degree': quadrature_degree})
V = VectorFunctionSpace(mesh, 'CG', 2)
Q = FunctionSpace(mesh, 'CG', 1)
K = FunctionSpace(mesh, 'CG', 1)
bcu = []
bcp = []
bck = []
bce = []
variables = ['U', 'P', 'K', 'E']
bc_lists = [bcu, bcp, bck, bce]
spaces = [V, Q, K, K]
for boundary_name, markers in boundary_markers.items():
    if markers is None:
        continue
    for marker in markers:
        for variable, bc_list, function_space in zip(variables, bc_lists, spaces):
            condition_value = boundary_conditions[boundary_name].get(variable)
            if condition_value is not None:
                bc_list.append(DirichletBC(function_space, condition_value, marked_facets, marker))
nu = Constant(physical_prm['VISCOSITY'])
force = Constant(physical_prm['FORCE'])
dt = Constant(simulation_prm['STEP_SIZE'])
NS_LINEAR_SOLVER = simulation_prm.get('NS_LINEAR_SOLVER', simulation_prm.get('LINEAR_SOLVER', 'mumps'))
NS_LINEAR_PRECONDITIONER = simulation_prm.get(
    'NS_LINEAR_PRECONDITIONER', simulation_prm.get('LINEAR_PRECONDITIONER', None)
)
KE_LINEAR_SOLVER = simulation_prm.get('KE_LINEAR_SOLVER', simulation_prm.get('LINEAR_SOLVER', 'default'))
KE_LINEAR_PRECONDITIONER = simulation_prm.get(
    'KE_LINEAR_PRECONDITIONER', simulation_prm.get('LINEAR_PRECONDITIONER', 'default')
)
y = calculate_Distance_field(K, marked_facets, boundary_markers['WALLS'], 0.01)
u, v, u1, u0 = initialize_functions(V, Constant(initial_conditions['U']))
p, q, p1, p0 = initialize_functions(Q, Constant(initial_conditions['P']))
ke_options = {
    'LINEAR_SOLVER': KE_LINEAR_SOLVER,
    'LINEAR_PRECONDITIONER': KE_LINEAR_PRECONDITIONER,
}
turbulence_model = KEpsilon(
    K,
    bck,
    bce,
    initial_conditions['K'],
    initial_conditions['E'],
    nu,
    force,
    dx,
    ds,
    dt,
    y,
    ke_options=ke_options,
)

turbulence_model.construct_forms(u1)
nu_t = turbulence_model.nu_t
h = CellDiameter(mesh)
u_mag = sqrt(dot(u0, u0) + 1e-10)
tau = h / (2.0 * u_mag)
residual = (u - u0) / dt + dot(u0, nabla_grad(u)) - force
F_supg = inner(tau * dot(u0, nabla_grad(v)), residual) * dx
F1 = (
    dot((u - u0) / dt, v) * dx
    + dot(dot(u0, nabla_grad(u)), v) * dx
    + inner((nu + nu_t) * grad(u), grad(v)) * dx
    - dot(force, v) * dx
    + F_supg
)
F2 = dot(grad(p), grad(q)) * dx + dot(div(u1) / dt, q) * dx
F3 = dot(u, v) * dx - dot(u1, v) * dx + dt * dot(grad(p1), v) * dx
a_1, l_1 = lhs(F1), rhs(F1)
a_2, l_2 = lhs(F2), rhs(F2)
a_3, l_3 = lhs(F3), rhs(F3)

u_relax = simulation_prm.get('U_RELAXATION_FACTOR', 1.0)
turb_relax = simulation_prm.get('TURB_RELAXATION_FACTOR', 1.0)
TOLERANCE_GLOBAL = float(simulation_prm['TOLERANCE'])
TOLERANCE_U = float(simulation_prm.get('TOLERANCE_U', TOLERANCE_GLOBAL))
TOLERANCE_P = float(simulation_prm.get('TOLERANCE_P', TOLERANCE_GLOBAL))
TOLERANCE_K = float(simulation_prm.get('TOLERANCE_K', TOLERANCE_GLOBAL))
TOLERANCE_E = float(simulation_prm.get('TOLERANCE_E', TOLERANCE_GLOBAL))
RUNTIME_WRITE_INTERVAL = max(0, int(simulation_prm.get('RUNTIME_WRITE_INTERVAL', 0)))
RUNTIME_WRITE_PVD = bool(simulation_prm.get('RUNTIME_WRITE_PVD', True))
RUNTIME_WRITE_RESIDUALS = bool(simulation_prm.get('RUNTIME_WRITE_RESIDUALS', True))
h_x = MaxCellEdgeLength(mesh)
h_y = MinCellEdgeLength(mesh)
cfl_projection_space = FunctionSpace(mesh, 'DG', 0)

residual_keys = ['u', 'p', 'k', 'e']
residuals = {key: [] for key in residual_keys}
start_time = time.time()
interrupted = False
last_completed_iter = 0
runtime_pvd_files = {}
runtime_pvd_directory = None
if post_processing.get('SAVE', False) and RUNTIME_WRITE_INTERVAL > 0 and RUNTIME_WRITE_PVD:
    runtime_pvd_directory = os.path.join(
        saving_directory['PVD_FILES'],
        f'runtime_{time.strftime("%Y%m%d_%H%M%S")}',
    )
    os.makedirs(runtime_pvd_directory, exist_ok=True)
    runtime_pvd_files = {
        'u': File(os.path.join(runtime_pvd_directory, 'u_runtime.pvd')),
        'p': File(os.path.join(runtime_pvd_directory, 'p_runtime.pvd')),
        'k': File(os.path.join(runtime_pvd_directory, 'k_runtime.pvd')),
        'e': File(os.path.join(runtime_pvd_directory, 'e_runtime.pvd')),
    }
    if IS_ROOT:
        print(f'Runtime PVD snapshots will be written under: {runtime_pvd_directory}')
if post_processing.get('SAVE', False) and RUNTIME_WRITE_INTERVAL > 0 and RUNTIME_WRITE_RESIDUALS:
    os.makedirs(saving_directory['RESIDUALS'], exist_ok=True)

try:
    if IS_ROOT:
        if NS_LINEAR_PRECONDITIONER in (None, '', 'default'):
            print(f'NS linear solver: {NS_LINEAR_SOLVER}')
        else:
            print(f'NS linear solver: {NS_LINEAR_SOLVER} (preconditioner: {NS_LINEAR_PRECONDITIONER})')
        if KE_LINEAR_PRECONDITIONER in (None, '', 'default'):
            print(f'k-epsilon linear solver: {KE_LINEAR_SOLVER}')
        else:
            print(
                f'k-epsilon linear solver: {KE_LINEAR_SOLVER} '
                f'(preconditioner: {KE_LINEAR_PRECONDITIONER})'
            )

    for iteration in range(simulation_prm['MAX_ITERATIONS']):
        if iteration > 0:
            step_size = calculate_cfl_time_step(
                u0, h_x, h_y, simulation_prm['CFL_RELAXATION'], mesh, cfl_space=cfl_projection_space
            )
            min_step_size = simulation_prm.get('MIN_STEP_SIZE', None)
            max_step_size = simulation_prm.get('MAX_STEP_SIZE', None)
            if min_step_size is not None:
                step_size = max(step_size, min_step_size)
            if max_step_size is not None:
                step_size = min(step_size, max_step_size)
            dt.assign(step_size)
        dt_value = float(dt.values()[0])

        A_1 = assemble(a_1)
        b_1 = assemble(l_1)
        [bc.apply(A_1, b_1) for bc in bcu]
        _solve_linear_system(A_1, u1.vector(), b_1, NS_LINEAR_SOLVER, NS_LINEAR_PRECONDITIONER)

        A_2 = assemble(a_2)
        b_2 = assemble(l_2)
        [bc.apply(A_2, b_2) for bc in bcp]
        _solve_linear_system(A_2, p1.vector(), b_2, NS_LINEAR_SOLVER, NS_LINEAR_PRECONDITIONER)

        A_3 = assemble(a_3)
        b_3 = assemble(l_3)
        [bc.apply(A_3, b_3) for bc in bcu]
        _solve_linear_system(A_3, u1.vector(), b_3, NS_LINEAR_SOLVER, NS_LINEAR_PRECONDITIONER)

        turbulence_model.solve_turbulence_model()
        errors = [
            _squared_l2_error(u1, u0, dx),
            _squared_l2_error(p1, p0, dx),
            _squared_l2_error(turbulence_model.k1, turbulence_model.k0, dx),
            _squared_l2_error(turbulence_model.e1, turbulence_model.e0, dx),
        ]
        break_flag = (
            errors[0] <= TOLERANCE_U and
            errors[1] <= TOLERANCE_P and
            errors[2] <= TOLERANCE_K and
            errors[3] <= TOLERANCE_E
        )

        if IS_ROOT:
            print(
                f'iter: {iteration + 1} ({time.time() - start_time:.2f}s, dt = {dt_value:.2e}s) --- L2 errors: '
                f'|u1-u0|= {errors[0]:.2e}, |p1-p0|= {errors[1]:.2e}, '
                f'|k1-k0|= {errors[2]:.2e}, |e1-e0|= {errors[3]:.2e} '
                f'(req: u < {TOLERANCE_U:.2e}, p < {TOLERANCE_P:.2e}, '
                f'k < {TOLERANCE_K:.2e}, e < {TOLERANCE_E:.2e})'
            )
        for key, error in zip(residuals.keys(), errors):
            residuals[key].append(error)

        if post_processing.get('SAVE', False) and RUNTIME_WRITE_INTERVAL > 0:
            if ((iteration + 1) % RUNTIME_WRITE_INTERVAL == 0) or break_flag:
                output_step = float(iteration + 1)
                if runtime_pvd_files:
                    try:
                        runtime_pvd_files['u'] << (u1, output_step)
                        runtime_pvd_files['p'] << (p1, output_step)
                        runtime_pvd_files['k'] << (turbulence_model.k1, output_step)
                        runtime_pvd_files['e'] << (turbulence_model.e1, output_step)
                    except Exception as exc:
                        runtime_pvd_files = {}
                        if IS_ROOT:
                            print(
                                'Warning: runtime PVD snapshot write failed; '
                                'disabling further intermediate PVD writes for this run. '
                                f'{type(exc).__name__}: {exc}'
                            )
                if RUNTIME_WRITE_RESIDUALS:
                    for key, values in residuals.items():
                        save_list(values, saving_directory['RESIDUALS'] + key + '.txt')
                if IS_ROOT:
                    print(f'Runtime snapshot written at iter {iteration + 1} (step = {output_step:.0f}).')

        u0.assign((1.0 - u_relax) * u0 + u_relax * u1)
        p0.assign(p1)
        turbulence_model.update_variables(relaxation=turb_relax)
        last_completed_iter = iteration + 1

        if break_flag:
            if IS_ROOT:
                print(
                    f'Simulation converged in {iteration + 1} iterations '
                    f'({time.time() - start_time:.2f} seconds)'
                )
            break
except KeyboardInterrupt:
    interrupted = True
    if IS_ROOT:
        print(
            f'KeyboardInterrupt received after {last_completed_iter} completed iterations. '
            f'Saving current state...'
        )

nu_t_final = project(
    turbulence_model.nu_t,
    K,
    form_compiler_parameters={'quadrature_degree': quadrature_degree},
)
solutions = {
    'u': u1,
    'p': p1,
    'k': turbulence_model.k1,
    'e': turbulence_model.e1,
    'nu_t': nu_t_final,
}
if post_processing['SAVE'] is True:
    for key, f in solutions.items():
        save_pvd_file(f, saving_directory['PVD_FILES'] + key + '.pvd')
        save_h5_file(f, saving_directory['H5_FILES'] + key + '.h5')
    for key, f in residuals.items():
        save_list(f, saving_directory['RESIDUALS'] + key + '.txt')
    if IS_ROOT and interrupted:
        print('Interrupted run state saved to PVD/H5/residual files.')
if post_processing['PLOT'] is True and not interrupted:
    visualize_functions(solutions)
    visualize_convergence(residuals)
