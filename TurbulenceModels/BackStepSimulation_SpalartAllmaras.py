from dolfin import *
from Utilities import *
from Configs.ConfigBackStep_SpalartAllmaras import *
from TurbulenceModel_SpalartAllmaras import SpalartAllmarasSteadyState as SpalartAllmaras
import time

# Use SA-specific parameters from the config file if available
if 'simulation_prm_SA' in globals():
    simulation_prm = simulation_prm_SA
if 'saving_directory_SA' in globals():
    saving_directory = saving_directory_SA

setup_simulation_log(saving_directory, __file__)

# Load mesh
[mesh, marked_facets] = load_mesh_from_file(mesh_files['MESH_DIRECTORY'], mesh_files['FACET_DIRECTORY'])

# Custom integration measures
quadrature_degree = simulation_prm['QUADRATURE_DEGREE']
dx = Measure("dx", domain=mesh, metadata={"quadrature_degree": quadrature_degree})
ds = Measure("ds", domain=mesh, metadata={"quadrature_degree": quadrature_degree})

# Construct function spaces
Element1 = VectorElement("CG", mesh.ufl_cell(), 2)
Element2 = FiniteElement("CG", mesh.ufl_cell(), 1)
W_elem = MixedElement([Element1, Element2])
W = FunctionSpace(mesh, W_elem)
K = FunctionSpace(mesh, "CG", 1)

# Construct boundary conditions
bcw = []
bcn = []

for boundary_name, markers in boundary_markers.items():
    if markers is None:
        continue

    for marker in markers:
        for variable, bc_list, function_space in zip(
            ['U', 'P', 'NU_TILDE'],
            [bcw, bcw, bcn],
            [W.sub(0), W.sub(1), K],
        ):
            condition_value = boundary_conditions[boundary_name].get(variable)
            if condition_value is None:
                continue

            # Back-step symmetry boundary: constrain only the normal velocity component.
            if boundary_name == 'SYMMETRY' and variable == 'U':
                bc_list.append(DirichletBC(function_space.sub(1), condition_value, marked_facets, marker))
            else:
                bc_list.append(DirichletBC(function_space, condition_value, marked_facets, marker))

# Initialize constants and expressions
nu = Constant(physical_prm['VISCOSITY'])
force = Constant(physical_prm['FORCE'])
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
u, v, u1, u0, p, q, p1, p0, w1, w0 = initialize_mixed_functions(
    W, Constant((*initial_conditions['U'], initial_conditions['P']))
)

# Initialize turbulence model
turbulence_model = SpalartAllmaras(
    K,
    bcn,
    initial_conditions['NU_TILDE'],
    nu,
    force,
    dx,
    ds,
    y,
)
turbulence_model.construct_forms(u1)

# Construct RANS form (same mixed steady Picard logic as BackStepSimulation.py)
FW = dot(dot(u0, nabla_grad(u)), v) * dx \
    + (nu + turbulence_model.nu_t) * inner(nabla_grad(u), nabla_grad(v)) * dx \
    - div(v) * p * dx \
    - div(u) * q * dx \
    + dot(p * FacetNormal(mesh), v) * ds \
    - dot((nu + turbulence_model.nu_t) * nabla_grad(u) * FacetNormal(mesh), v) * ds \
    - dot(force, v) * dx

# Precompute lhs and rhs
a_w = lhs(FW)
l_w = rhs(FW)

# Main loop
residuals = {key: [] for key in ['u', 'p', 'nu_tilde']}
start_time = time.time()
for iter in range(simulation_prm['MAX_ITERATIONS']):
    # Solve NS
    A_W = assemble(a_w)
    b_w = assemble(l_w)
    [bc.apply(A_W, b_w) for bc in bcw]
    solve(A_W, w1.vector(), b_w, 'mumps')

    # Solve turbulence model
    turbulence_model.solve_turbulence_model()

    break_flag, errors = are_close_all(
        [u1, p1, turbulence_model.nu_tilde1],
        [u0, p0, turbulence_model.nu_tilde0],
        simulation_prm['TOLERANCE'],
    )

    print(
        f'iter: {iter+1} ({time.time() - start_time:.2f}s) - L2 errors: '
        f'|u1-u0|= {errors[0]:.2e}, |p1-p0|= {errors[1]:.2e}, '
        f'|nu_tilde1-nu_tilde0|= {errors[2]:.2e} (required: {simulation_prm["TOLERANCE"]:.2e})'
    )

    for key, error in zip(residuals.keys(), errors):
        residuals[key].append(error)

    # Update variables for next iteration
    w0.assign(simulation_prm['PICARD_RELAXATION'] * w1 + (1 - simulation_prm['PICARD_RELAXATION']) * w0)
    turbulence_model.update_variables(simulation_prm['PICARD_RELAXATION'])

    # Check for convergence
    if break_flag:
        print(f'Simulation converged in {iter+1} iterations ({time.time() - start_time:.2f} seconds)')
        break

# Store solutions in one dictionary
u1, p1 = w1.split(deepcopy=True)
solutions = {'u': u1, 'p': p1, 'nu_tilde': turbulence_model.nu_tilde1}

# Visualize
if post_processing['PLOT'] == True:
    visualize_functions(solutions)
    visualize_convergence(residuals)

# Save results and residuals
if post_processing['SAVE'] == True:
    for (key, f) in solutions.items():
        save_pvd_file(f, saving_directory['PVD_FILES'] + key + '.pvd')
        save_h5_file(f, saving_directory['H5_FILES'] + key + '.h5')

    for (key, f) in residuals.items():
        save_list(f, saving_directory['RESIDUALS'] + key + '.txt')
