from dolfin import *
from Utilities import *
from Configs.ConfigChannel_SpalartAllmaras_Steady import *
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

# Construct periodic boundary condition.
mesh_width = mesh.coordinates()[:, 0].max() - mesh.coordinates()[:, 0].min()


class Periodic(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 0)

    def map(self, x, y):
        y[0] = x[0] - mesh_width
        y[1] = x[1]


periodic = Periodic(1e-5)


def build_midline_bulk_velocity_evaluator(mesh, x_location, channel_height, quadrature_degree):
    """Build an evaluator for U_bulk = integral_midline(u_x dS) / H."""
    mesh.init(mesh.topology().dim() - 1, mesh.topology().dim())
    facet_markers = MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)
    midpoint_tolerance = max(1.0e-10, DOLFIN_EPS)

    class MidLine(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[0], x_location, midpoint_tolerance)

    MidLine().mark(facet_markers, 1)
    dS_midline = Measure(
        "dS",
        domain=mesh,
        subdomain_data=facet_markers,
        metadata={"quadrature_degree": quadrature_degree},
    )
    line_length = assemble(Constant(1.0) * dS_midline(1))
    if float(line_length) <= 1.0e-14:
        raise RuntimeError(
            "Could not find an internal mesh line at x={:.6e} m for U_bulk_midline.".format(
                x_location
            )
        )

    def evaluate(velocity):
        line_integral = assemble(avg(velocity[0]) * dS_midline(1))
        return line_integral / channel_height, line_integral, line_length

    return evaluate


def build_channel_wall_distance(mesh, space, marked_facets, wall_markers, dx_measure):
    """Build the configured wall-distance field for the periodic channel."""
    mode = WALL_DISTANCE_MODE.strip().lower()
    if mode in ("exact", "analytic", "analytical"):
        y_coordinate = SpatialCoordinate(mesh)[1]
        y_min = Constant(float(mesh.coordinates()[:, 1].min()))
        y_max = Constant(float(mesh.coordinates()[:, 1].max()))
        distance_lower = y_coordinate - y_min
        distance_upper = y_max - y_coordinate
        distance = conditional(
            le(distance_lower, distance_upper),
            distance_lower,
            distance_upper,
        )
        return distance, "exact analytical channel distance"

    if mode in ("relaxed", "relaxed_yoon", "yoon", "yoon_eq19"):
        distance = calculate_relaxed_wall_distance_field_yoon_eq19(
            space,
            marked_facets,
            wall_markers,
            relax=WALL_DISTANCE_SIGMA_W,
            sigma_w=WALL_DISTANCE_SIGMA_W,
            g0=WALL_DISTANCE_G0,
            g_floor=WALL_DISTANCE_G_FLOOR,
            newton_rtol=WALL_DISTANCE_NEWTON_RTOL,
            newton_atol=WALL_DISTANCE_NEWTON_ATOL,
            newton_max_iters=WALL_DISTANCE_NEWTON_MAX_ITERATIONS,
            newton_relax=WALL_DISTANCE_NEWTON_RELAXATION,
            custom_dx=dx_measure,
        )
        return distance, "relaxed Yoon Eq. (19) reciprocal-distance field"

    raise ValueError(
        "Unknown WALL_DISTANCE_MODE '{}'. Use 'exact' or 'relaxed_yoon'.".format(
            WALL_DISTANCE_MODE
        )
    )

# Construct function spaces.
V = VectorFunctionSpace(mesh, "CG", 2, constrained_domain=periodic)
if CHANNEL_DRIVE_MODE_NORMALIZED == "body_force":
    Q = FunctionSpace(mesh, "CG", 1, constrained_domain=periodic)
else:
    Q = FunctionSpace(mesh, "CG", 1)
K = FunctionSpace(mesh, "CG", 1, constrained_domain=periodic)

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
y, wall_distance_description = build_channel_wall_distance(
    mesh,
    K,
    marked_facets,
    boundary_markers["WALLS"],
    dx,
)
if IS_ROOT:
    print(
        "Channel drive mode: {} with body force {} and pressure BCs p_in={}, p_out={}.".format(
            CHANNEL_DRIVE_MODE_NORMALIZED,
            BODY_FORCE,
            boundary_conditions["INFLOW"].get("P"),
            boundary_conditions["OUTFLOW"].get("P"),
        )
    )
    print("Channel wall-distance mode: {}.".format(wall_distance_description))

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

midline_x = mesh.coordinates()[:, 0].min() + 0.5 * mesh_width
evaluate_midline_bulk_velocity = build_midline_bulk_velocity_evaluator(
    mesh,
    midline_x,
    CHANNEL_HEIGHT,
    quadrature_degree,
)


def evaluate_midline_u_bulk(state):
    u_bulk, _line_integral, _line_length = evaluate_midline_bulk_velocity(state["u"])
    return u_bulk


solutions, residuals = run_steady_sa_ipcs_picard(
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
    normalize_pressure_mean=bool(simulation_prm.get("FLOW_IPCS_NORMALIZE_PRESSURE_MEAN", True)),
    ds=ds,
    picard_diagnostics=[
        {
            "key": "u_bulk_midline",
            "label": "U_bulk_midline",
            "unit": "m/s",
            "evaluator": evaluate_midline_u_bulk,
        },
    ],
    picard_convergence={
        "type": "diagnostic_window_relative_change",
        "diagnostic_key": "u_bulk_midline",
        "window": CHANNEL_BULK_CONVERGENCE_WINDOW,
        "relative_tolerance": CHANNEL_BULK_CONVERGENCE_RELATIVE_TOLERANCE,
        "require_flow_convergence": True,
        "require_pressure_tolerance": True,
        "require_nu_tilde_tolerance": True,
        "require_velocity_tolerance": True,
    },
    is_root=IS_ROOT,
)

domain_area = assemble(Constant(1.0) * dx)
u_bulk_domain = assemble(solutions["u"][0] * dx) / domain_area
u_bulk, _midline_integral, midline_length = evaluate_midline_bulk_velocity(solutions["u"])
re_actual = u_bulk * REYNOLDS_LENGTH / KINEMATIC_VISCOSITY
pressure_drop = pressure_drop_metric["PRESSURE_DROP"]
channel_length = mesh_width
rho_kinematic_pressure = 1.0
skin_friction_coefficient = (
    pressure_drop * CHANNEL_HEIGHT
    / (rho_kinematic_pressure * channel_length * u_bulk**2)
)

if IS_ROOT:
    print(
        "Channel bulk diagnostics: U_bulk_midline={:.6e} m/s, "
        "U_bulk_domain={:.6e} m/s, Re_H_actual={:.6e}, C_f={:.6e}".format(
            u_bulk,
            u_bulk_domain,
            re_actual,
            skin_friction_coefficient,
        )
    )
    print(
        "Channel friction definition: C_f=Delta_p*H/(rho*L*U_bulk^2), "
        "Delta_p={:.6e} Pa, H={:.6e} m, L={:.6e} m, rho={:.1f} "
        "(kinematic-pressure convention). U_bulk is the midline average "
        "at x={:.6e} m over a marked line length of {:.6e} m.".format(
            pressure_drop,
            CHANNEL_HEIGHT,
            channel_length,
            rho_kinematic_pressure,
            midline_x,
            midline_length,
        )
    )
