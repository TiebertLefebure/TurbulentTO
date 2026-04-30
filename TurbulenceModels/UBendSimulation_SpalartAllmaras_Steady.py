from dolfin import *
from Utilities import *
from Configs.ConfigUBend_SpalartAllmaras_Steady import *
from TurbulenceModel_SpalartAllmaras import SpalartAllmarasSteadyState as SpalartAllmaras
from SA_Steady_IPCS_Picard_Solver import run_steady_sa_ipcs_picard


parameters["std_out_all_processes"] = False
IS_ROOT = MPI.COMM_WORLD.Get_rank() == 0


# Use steady SA-specific parameters from the config file.
simulation_prm = steady_sa_solver_parameters
setup_simulation_log(saving_directory, __file__)

# Load mesh.
mesh, marked_facets = load_mesh_from_file(mesh_files["MESH_DIRECTORY"], mesh_files["FACET_DIRECTORY"])

# Custom integration measures.
quadrature_degree = simulation_prm["QUADRATURE_DEGREE"]
dx = Measure("dx", domain=mesh, metadata={"quadrature_degree": quadrature_degree})
ds = Measure(
    "ds",
    domain=mesh,
    subdomain_data=marked_facets,
    metadata={"quadrature_degree": quadrature_degree},
)

# Construct function spaces.
V = VectorFunctionSpace(mesh, "CG", 2)
Q = FunctionSpace(mesh, "CG", 1)
K = FunctionSpace(mesh, "CG", 1)

# Construct boundary conditions.
bcu = []
bcp = []
bcn = []

for boundary_name, markers in boundary_markers.items():
    if markers is None:
        continue

    for marker in markers:
        for variable, bc_list, function_space in zip(
            ["U", "P", "NU_TILDE"],
            [bcu, bcp, bcn],
            [V, Q, K],
        ):
            condition_value = boundary_conditions[boundary_name].get(variable)
            if condition_value is not None:
                bc_list.append(DirichletBC(function_space, condition_value, marked_facets, marker))

# Initialize constants and wall distance.
nu = Constant(physical_prm["VISCOSITY"])
force = Constant(physical_prm["FORCE"])
dt = Constant(float(simulation_prm["FLOW_IPCS_TIME_STEP"]))
y = calculate_relaxed_wall_distance_field_yoon_eq19(
    K,
    marked_facets,
    boundary_markers["WALLS"],
    relax=WALL_DISTANCE_SIGMA_W,
    sigma_w=WALL_DISTANCE_SIGMA_W,
    g0=WALL_DISTANCE_G0,
    g_floor=WALL_DISTANCE_G_FLOOR,
    newton_rtol=WALL_DISTANCE_NEWTON_RTOL,
    newton_atol=WALL_DISTANCE_NEWTON_ATOL,
    newton_max_iters=WALL_DISTANCE_NEWTON_MAX_ITERATIONS,
    newton_relax=WALL_DISTANCE_NEWTON_RELAXATION,
    custom_dx=dx,
)

# Initialize functions.
u, v, u1, u0 = initialize_functions(V, Constant(initial_conditions["U"]))
p, q, p1, p0 = initialize_functions(Q, Constant(initial_conditions["P"]))

# Initialize steady SA model.  The local steady class is kept algebraically
# aligned with the frozen TO SA model, without TO penalty terms.
sa_options = {
    "LINEAR_SOLVER": simulation_prm.get("SA_TRANSPORT_LINEAR_SOLVER", "default"),
    "LINEAR_PRECONDITIONER": simulation_prm.get("SA_TRANSPORT_LINEAR_PRECONDITIONER", "default"),
}
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
turbulence_model.construct_forms(u0)

# Pseudo-time incremental IPCS flow forms, without momentum SUPG.
F1 = (
    dot((u - u0) / dt, v) * dx
    + dot(dot(u0, nabla_grad(u)), v) * dx
    + inner((nu + turbulence_model.nu_t) * grad(u), grad(v)) * dx
    + dot(grad(p0), v) * dx
    - dot(force, v) * dx
)
F2 = dot(grad(p - p0), grad(q)) * dx + dot(div(u1) / dt, q) * dx
F3 = dot(u, v) * dx - dot(u1, v) * dx + dt * dot(grad(p1 - p0), v) * dx

a_1, l_1 = lhs(F1), rhs(F1)
a_2, l_2 = lhs(F2), rhs(F2)
a_3, l_3 = lhs(F3), rhs(F3)

run_steady_sa_ipcs_picard(
    simulation_prm=simulation_prm,
    post_processing=post_processing,
    saving_directory=saving_directory,
    dx=dx,
    dt=dt,
    a_1=a_1,
    l_1=l_1,
    a_2=a_2,
    l_2=l_2,
    a_3=a_3,
    l_3=l_3,
    bcu=bcu,
    bcp=bcp,
    u0=u0,
    u1=u1,
    p0=p0,
    p1=p1,
    velocity_space=V,
    pressure_space=Q,
    turbulence_space=K,
    turbulence_model=turbulence_model,
    normalize_pressure_mean=bool(simulation_prm.get("FLOW_IPCS_NORMALIZE_PRESSURE_MEAN", False)),
    ds=ds,
    pressure_drop_metric=pressure_drop_metric,
    is_root=IS_ROOT,
)
