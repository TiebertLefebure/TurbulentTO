from dolfin import *
from Utilities import *
from Configs.ConfigChannel_Laminar import *
import time

setup_simulation_log(saving_directory, __file__)

# Load mesh
[mesh, marked_facets] = load_mesh_from_file(mesh_files['MESH_DIRECTORY'], mesh_files['FACET_DIRECTORY'])

# Custom integration measures
quadrature_degree = simulation_prm['QUADRATURE_DEGREE']
dx = Measure("dx", domain=mesh, metadata={"quadrature_degree": quadrature_degree})
ds = Measure("ds", domain=mesh, metadata={"quadrature_degree": quadrature_degree})

# Construct periodic boundary condition
mesh_width = mesh.coordinates()[:, 0].max() - mesh.coordinates()[:, 0].min()


class Periodic(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 0)

    def map(self, x, y):
        y[0] = x[0] - mesh_width
        y[1] = x[1]


periodic = Periodic(1E-5)

# Construct function spaces
V = VectorFunctionSpace(mesh, "CG", 2, constrained_domain=periodic)
Q = FunctionSpace(mesh, "CG", 1)
K = FunctionSpace(mesh, "CG", 1, constrained_domain=periodic)

# Construct boundary conditions
bcu = []
bcp = []
bck = []
bce = []

for boundary_name, markers in boundary_markers.items():
    if markers is None:
        continue

    for marker in markers:
        for variable, bc_list, function_space in zip(['U', 'P', 'K', 'E'], [bcu, bcp, bck, bce], [V, Q, K, K]):
            condition_value = boundary_conditions[boundary_name].get(variable)
            if condition_value is not None:
                bc_list.append(DirichletBC(function_space, condition_value, marked_facets, marker))

# Initialize constants and expressions
nu = Constant(physical_prm['VISCOSITY'])
force = Constant(physical_prm['FORCE'])
dt = Constant(simulation_prm['STEP_SIZE'])
nu_t = Constant(0.0)

# Initialize functions
u, v, u1, u0 = initialize_functions(V, Constant(initial_conditions['U']))
p, q, p1, p0 = initialize_functions(Q, Constant(initial_conditions['P']))

# Construct flow forms
F1 = (
    dot((u - u0) / dt, v) * dx
    + dot(dot(u0, nabla_grad(u)), v) * dx
    + inner((nu + nu_t) * grad(u), grad(v)) * dx
    - dot(force, v) * dx
)
F2 = dot(grad(p), grad(q)) * dx + dot(div(u1) / dt, q) * dx
F3 = dot(u, v) * dx - dot(u1, v) * dx + dt * dot(grad(p1), v) * dx

# Precompute lhs and rhs
a_1, l_1 = lhs(F1), rhs(F1)
a_2, l_2 = lhs(F2), rhs(F2)
a_3, l_3 = lhs(F3), rhs(F3)

# Main loop
residual_keys = ['u', 'p']
residuals = {key: [] for key in residual_keys}
start_time = time.time()

for iter in range(simulation_prm['MAX_ITERATIONS']):
    # Dynamic time-stepping
    if iter > 0:
        h_x = MaxCellEdgeLength(mesh)
        h_y = MinCellEdgeLength(mesh)
        step_size = calculate_cfl_time_step(u0, h_x, h_y, simulation_prm['CFL_RELAXATION'], mesh)
        min_step_size = simulation_prm.get('MIN_STEP_SIZE', None)
        max_step_size = simulation_prm.get('MAX_STEP_SIZE', None)
        if min_step_size is not None:
            step_size = max(step_size, min_step_size)
        if max_step_size is not None:
            step_size = min(step_size, max_step_size)
        dt.assign(Constant(step_size))
    dt_value = float(dt.values()[0])

    # Solve NS
    A_1 = assemble(a_1)
    b_1 = assemble(l_1)
    [bc.apply(A_1, b_1) for bc in bcu]
    solve(A_1, u1.vector(), b_1, 'mumps')

    A_2 = assemble(a_2)
    b_2 = assemble(l_2)
    [bc.apply(A_2, b_2) for bc in bcp]
    solve(A_2, p1.vector(), b_2, 'mumps')

    A_3 = assemble(a_3)
    b_3 = assemble(l_3)
    [bc.apply(A_3, b_3) for bc in bcu]
    solve(A_3, u1.vector(), b_3, 'mumps')

    break_flag, errors = are_close_all([u1, p1], [u0, p0], simulation_prm['TOLERANCE'])
    print(
        f'iter: {iter + 1} ({time.time() - start_time:.2f}s, dt = {dt_value:.3e}s) --- L2 errors: '
        f'|u1-u0|= {errors[0]:.2e}, |p1-p0|= {errors[1]:.2e} '
        f'(required: {simulation_prm["TOLERANCE"]:.2e})'
    )

    for key, error in zip(residuals.keys(), errors):
        residuals[key].append(error)

    # Update variables for next iteration
    u0.assign(u1)
    p0.assign(p1)

    # Check for convergence
    if break_flag:
        print(f'Simulation converged in {iter + 1} iterations ({time.time() - start_time:.2f} seconds)')
        break

solutions = {'u': u1, 'p': p1}

# Visualize
if post_processing['PLOT'] is True:
    visualize_functions(solutions)
    visualize_convergence(residuals)

# Save results and residuals
if post_processing['SAVE'] is True:
    for (key, f) in solutions.items():
        save_pvd_file(f, saving_directory['PVD_FILES'] + key + '.pvd')
        save_h5_file(f, saving_directory['H5_FILES'] + key + '.h5')

    for (key, f) in residuals.items():
        save_list(f, saving_directory['RESIDUALS'] + key + '.txt')
