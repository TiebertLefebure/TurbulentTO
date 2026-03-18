from dolfin import *
from Utilities import *
from Configs.ConfigUBend_SpalartAllmaras import *
from TurbulenceModel_SpalartAllmaras import SpalartAllmarasSteadyState as SpalartAllmaras
import time

parameters["std_out_all_processes"] = False
IS_ROOT = True


def _l2_norm_diff(f1, f0, dx_measure):
    """True L2 norm of the difference between two FEniCS Functions."""
    diff = f1 - f0
    if f1.ufl_shape == ():
        return sqrt(assemble(diff**2 * dx_measure))
    return sqrt(assemble(dot(diff, diff) * dx_measure))


# Use SA-specific parameters from the config file if available
if "simulation_prm_SA" in globals():
    simulation_prm = simulation_prm_SA
if "saving_directory_SA_STEADY" in globals():
    saving_directory = saving_directory_SA_STEADY
elif "saving_directory_SA" in globals():
    saving_directory = saving_directory_SA

# Load mesh
[mesh, marked_facets] = load_mesh_from_file(mesh_files["MESH_DIRECTORY"], mesh_files["FACET_DIRECTORY"])
IS_ROOT = (MPI.COMM_WORLD.Get_rank() == 0)

# Custom integration measures
quadrature_degree = simulation_prm["QUADRATURE_DEGREE"]
dx = Measure("dx", domain=mesh, metadata={"quadrature_degree": quadrature_degree})
ds = Measure("ds", domain=mesh, metadata={"quadrature_degree": quadrature_degree})

# Construct function spaces (mixed steady NS + scalar SA)
Element_u = VectorElement("CG", mesh.ufl_cell(), 2)
Element_p = FiniteElement("CG", mesh.ufl_cell(), 1)
W_elem = MixedElement([Element_u, Element_p])
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
            ["U", "P", "NU_TILDE"],
            [bcw, bcw, bcn],
            [W.sub(0), W.sub(1), K],
        ):
            condition_value = boundary_conditions[boundary_name].get(variable)
            if condition_value is not None:
                bc_list.append(DirichletBC(function_space, condition_value, marked_facets, marker))

# Initialize constants and expressions
nu = Constant(physical_prm["VISCOSITY"])
force = Constant(physical_prm["FORCE"])
NS_LINEAR_SOLVER = simulation_prm.get("NS_LINEAR_SOLVER", simulation_prm.get("LINEAR_SOLVER", "mumps"))
NS_LINEAR_PRECONDITIONER = simulation_prm.get(
    "NS_LINEAR_PRECONDITIONER", simulation_prm.get("LINEAR_PRECONDITIONER", None)
)
SA_LINEAR_SOLVER = simulation_prm.get("SA_LINEAR_SOLVER", "default")
SA_LINEAR_PRECONDITIONER = simulation_prm.get("SA_LINEAR_PRECONDITIONER", "default")
wall_distance_method = simulation_prm.get("WALL_DISTANCE_METHOD", "OriginalEikonal")
wall_distance_relax = simulation_prm.get("WALL_DISTANCE_EIKONAL_RELAXATION", 0.01)
wall_distance_sigma_w = simulation_prm.get("WALL_DISTANCE_YOON_SIGMA_W", 0.1)
wall_distance_g0 = simulation_prm.get("WALL_DISTANCE_YOON_G0", 20.0)
wall_distance_g_floor = simulation_prm.get("WALL_DISTANCE_YOON_G_FLOOR", 1.0e-12)
y = calculate_Distance_field(
    K,
    marked_facets,
    boundary_markers["WALLS"],
    wall_distance_relax,
    method=wall_distance_method,
    sigma_w=wall_distance_sigma_w,
    g0=wall_distance_g0,
    g_floor=wall_distance_g_floor,
)

# Initialize mixed NS functions and SA field
u, v, u1, u0, p, q, p1, p0, w1, w0 = initialize_mixed_functions(
    W, Constant((*initial_conditions["U"], initial_conditions["P"]))
)

sa_options = dict(simulation_prm.get("SA_OPTIONS", {}))
sa_options["LINEAR_SOLVER"] = SA_LINEAR_SOLVER
sa_options["LINEAR_PRECONDITIONER"] = SA_LINEAR_PRECONDITIONER
turbulence_model = SpalartAllmaras(
    K,
    bcn,
    initial_conditions["NU_TILDE"],
    nu,
    force,
    dx,
    ds,
    y,
    sa_options=sa_options,
)
turbulence_model.construct_forms(u1)

# Steady Picard form (frozen convection with u0 and frozen nu_t from nu_tilde0)
n = FacetNormal(mesh)
FW = (
    dot(dot(u0, nabla_grad(u)), v) * dx
    + (nu + turbulence_model.nu_t) * inner(nabla_grad(u), nabla_grad(v)) * dx
    - div(v) * p * dx
    - div(u) * q * dx
    + dot(p * n, v) * ds
    - dot((nu + turbulence_model.nu_t) * nabla_grad(u) * n, v) * ds
    - dot(force, v) * dx
)

a_w = lhs(FW)
l_w = rhs(FW)

PICARD_RELAXATION = float(simulation_prm.get("PICARD_RELAXATION", 0.2))
NUT_RELAXATION_FACTOR = float(simulation_prm.get("NUT_RELAXATION_FACTOR", PICARD_RELAXATION))
SA_INNER_ITERS = max(1, int(simulation_prm.get("SA_INNER_ITERS", 1)))

TOLERANCE_GLOBAL = float(simulation_prm["TOLERANCE"])
TOLERANCE_U = float(simulation_prm.get("TOLERANCE_U", TOLERANCE_GLOBAL))
TOLERANCE_P = float(simulation_prm.get("TOLERANCE_P", TOLERANCE_GLOBAL))
TOLERANCE_NU_TILDE = float(simulation_prm.get("TOLERANCE_NU_TILDE", TOLERANCE_GLOBAL))


def _solve_linear_system(A, x, b):
    if NS_LINEAR_PRECONDITIONER in (None, "", "default"):
        solve(A, x, b, NS_LINEAR_SOLVER)
    else:
        solve(A, x, b, NS_LINEAR_SOLVER, NS_LINEAR_PRECONDITIONER)


# Main loop
residuals = {key: [] for key in ["u", "p", "nu_tilde"]}
start_time = time.time()

for iter in range(simulation_prm["MAX_ITERATIONS"]):
    A_W = assemble(a_w)
    b_w = assemble(l_w)
    [bc.apply(A_W, b_w) for bc in bcw]
    _solve_linear_system(A_W, w1.vector(), b_w)

    nu_tilde0_outer_prev = Function(K)
    nu_tilde0_outer_prev.assign(turbulence_model.nu_tilde0)
    for sa_iter in range(SA_INNER_ITERS):
        turbulence_model.solve_turbulence_model()
        if sa_iter < SA_INNER_ITERS - 1:
            turbulence_model.update_variables(relaxation=NUT_RELAXATION_FACTOR)

    errors = [
        _l2_norm_diff(u1, u0, dx),
        _l2_norm_diff(p1, p0, dx),
        _l2_norm_diff(turbulence_model.nu_tilde1, nu_tilde0_outer_prev, dx),
    ]
    break_flag = (
        errors[0] <= TOLERANCE_U
        and errors[1] <= TOLERANCE_P
        and errors[2] <= TOLERANCE_NU_TILDE
    )

    if IS_ROOT:
        print(
            f"iter: {iter+1} ({time.time() - start_time:.2f}s, sa_inner = {SA_INNER_ITERS}) "
            f"----- L2 norms: |u1-u0| = {errors[0]:.2e}, |p1-p0| = {errors[1]:.2e}, "
            f"|nu_tilde1-nu_tilde0| = {errors[2]:.2e} "
            f"(req: u < {TOLERANCE_U:.2e}, p < {TOLERANCE_P:.2e}, nu_tilde < {TOLERANCE_NU_TILDE:.2e})"
        )

    for key, error in zip(residuals.keys(), errors):
        residuals[key].append(error)

    # Picard update
    w0.assign(PICARD_RELAXATION * w1 + (1.0 - PICARD_RELAXATION) * w0)
    turbulence_model.update_variables(relaxation=NUT_RELAXATION_FACTOR)

    if break_flag:
        if IS_ROOT:
            print(
                f"Steady simulation converged in {iter+1} iterations "
                f"({time.time() - start_time:.2f} seconds)"
            )
        break

# Store solutions
u_out, p_out = w1.split(deepcopy=True)
solutions = {"u": u_out, "p": p_out, "nu_tilde": turbulence_model.nu_tilde1}

# Visualize
if post_processing["PLOT"] == True:
    visualize_functions(solutions)
    visualize_convergence(residuals)

# Save results and residuals
if post_processing["SAVE"] == True:
    for (key, f) in solutions.items():
        save_pvd_file(f, saving_directory["PVD_FILES"] + key + ".pvd")
        save_h5_file(f, saving_directory["H5_FILES"] + key + ".h5")

    for (key, f) in residuals.items():
        save_list(f, saving_directory["RESIDUALS"] + key + ".txt")
