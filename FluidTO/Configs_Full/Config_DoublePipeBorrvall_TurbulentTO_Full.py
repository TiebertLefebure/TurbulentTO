import os
import numpy as np
from dolfin import DOLFIN_EPS, Expression, Function, MeshFunction, MPI, SubDomain, cells, near
from Utilities_SharedTO import load_mesh_from_xdmf


# =======================================================================
# Configuration: Borrvall Double Pipe - Turbulent Full (SA, Re = 1,660)
#
# Two symmetric ports on the left and right boundaries.
# This config targets the Full adjoint: the reciprocal wall-distance G enters
# the monolithic primal state and the adjoint system together with
# (u, p, nu_tilde).
# =======================================================================

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Mesh path for this benchmark geometry.
# Generate via: cd Meshes/DoublePipeBorrvall && python3 double_pipe_gmsh.py && python3 gmsh_to_xdmf.py
mesh_files = {
    "MESH_DIRECTORY": os.path.join(REPO_ROOT, "Meshes/DoublePipeBorrvall/mesh.xdmf"),
}


def create_design_mesh():
    return load_mesh_from_xdmf(mesh_files["MESH_DIRECTORY"], MPI.comm_world)


# Geometry and reference meshing parameters for the two-port design box.
DOMAIN_X_MIN = 0.0
DOMAIN_Y_MIN = 0.0
DOMAIN_X_MAX = 1.5
DOMAIN_Y_MAX = 1.0
NX = 150  # reference resolution used to generate the Gmsh mesh
NY = 100
TOL = DOLFIN_EPS

# Port layout on left/right boundaries
PORT_WIDTH = 1.0 / 6.0
PORT_TOP_MARGIN = 1.0 / 4.0
PORT_BOTTOM_MARGIN = 1.0 / 4.0

TOP_PORT_Y_MAX = DOMAIN_Y_MAX - PORT_TOP_MARGIN
TOP_PORT_Y_MIN = TOP_PORT_Y_MAX - PORT_WIDTH
BOTTOM_PORT_Y_MIN = PORT_BOTTOM_MARGIN
BOTTOM_PORT_Y_MAX = BOTTOM_PORT_Y_MIN + PORT_WIDTH

INLET_SEGMENTS = [
    (TOP_PORT_Y_MIN, TOP_PORT_Y_MAX),
    (BOTTOM_PORT_Y_MIN, BOTTOM_PORT_Y_MAX),
]
OUTLET_SEGMENTS = [
    (TOP_PORT_Y_MIN, TOP_PORT_Y_MAX),
    (BOTTOM_PORT_Y_MIN, BOTTOM_PORT_Y_MAX),
]

# Optional passive inlet strip near the left boundary. The default
# "fluid_only" mode only pins the inlet-window cells to fluid and leaves the
# rest of the strip free; "block" reproduces the older behavior that also
# forces the non-port part of the strip to solid.
ENABLE_INLET_PASSIVE_STRIP = False
INLET_PASSIVE_STRIP_MODE = "fluid_only"
INLET_PASSIVE_STRIP_CELLS = 3
INLET_PASSIVE_STRIP_LENGTH = INLET_PASSIVE_STRIP_CELLS * ((DOMAIN_X_MAX - DOMAIN_X_MIN) / NX)

# Flow and Brinkman parameters for the monolithic primal solve.
MU_FLUID_VALUE = 1.0e-4
RHO_FLUID_VALUE = 1.0
U_MAX_INLETS = [1.0, 1.0]
U_MAX_OUTLETS = [1.0, 1.0]

# =============================================================================================
# Reynolds number: Re = U_MAX_INLET * PORT_WIDTH * RHO_FLUID_VALUE / MU_FLUID_VALUE = 1,660
# =============================================================================================



# SA transport parameters for the monolithic primal state.
SA_MUT_RATIO = 5.0
SA_SMOOTH_ABS_EPS = 1.0e-12
SA_INIT_WALL_DIST_SCALE = 0.05 * DOMAIN_Y_MAX
SA_NU_TILDE_FLOOR = 1.0e-12
SA_NU_TILDE_PENALTY_ALPHA = 1.0e3
SA_NU_TILDE_PENALTY_N = 3.0

# Relaxed wall equation parameters for the coupled reciprocal distance state.
SA_WALL_DENSITY_SOURCE = "design"
SA_WALL_SIGMA = 0.01
SA_WALL_G0 = 20.0
SA_WALL_PENALTY_ALPHA = 1.0e3
SA_WALL_PENALTY_N = 3.0
SA_WALL_G_FLOOR = 1.0e-8

# Only treat near-solid cells as artificial walls; the initial rho=1/3 gray
# field should not trigger wall penalties across the whole domain.
SA_WALL_SOLID_THRESHOLD = 0.10
SA_WALL_DISTANCE_FLOOR = 1.0e-6 * PORT_WIDTH

# MMA objective and continuation parameters for the topology update.
VOL_FRAC = 1.0 / 3.0
INITIAL_DENSITY_VALUE = 1.0 / 3.0
MAX_INNER_ITERATIONS_SCHEDULE = [35, 80, 100, 120, 120, 120, 100, 100]
OBJECTIVE_CONVERGENCE_TOL = 5e-5
OBJECTIVE_STREAK_TO_STOP = 5

Q_PENAL_SCHEDULE = [0.05, 0.1, 0.1, 0.2, 0.5, 1.0, 1.0, 1.0]
MOVE_LIMIT_SCHEDULE = [0.08, 0.06, 0.04, 0.02, 0.01, 0.005, 0.003, 0.002]
BETA_PROJ_SCHEDULE = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]

# Full primal-state solve parameters.
LINEAR_SOLVER = "mumps"
STATE_SOLVE_METHOD = "newtontr"
STATE_RTOL = 1.0e-6
STATE_ATOL = 1.0e-8
STATE_MAX_ITERS = 120
STATE_INITIAL_SA_SWEEPS = 4
# Ramp the turbulent-viscosity feedback into the momentum equations gradually
# so the first monolithic solve does not jump straight from Stokes/SA startup
# to a strongly coupled (u, p, nu_tilde, G) state.
STATE_TURBULENCE_COUPLING_SCHEDULE = [
    {"weight": 0.0, "max_iters": 220, "atol": 8.0e-4},
    {"weight": 0.20, "max_iters": 180, "atol": 6.0e-4},
    {"weight": 0.45, "max_iters": 180, "atol": 4.0e-4},
    {"weight": 0.65, "max_iters": 200, "atol": 3.0e-4},
    {"weight": 0.82, "max_iters": 220, "atol": 2.0e-4},
    {"weight": 1.0, "max_iters": 250, "atol": 1.0e-4},
]
# Keep one alternate-globalization retry from the current iterate, then one
# clean Stokes rebuild retry with the same safer line-search globalization.
STATE_RECOVERY_ATTEMPTS = [
    {
        "label": "current-iterate line-search retry",
        "method": "newtonls",
        "line_search": "bt",
        "max_iters": 220,
        "restart_with_stokes": False,
    },
    {
        "label": "Stokes rebuild line-search retry",
        "method": "newtonls",
        "line_search": "bt",
        "max_iters": 250,
        "restart_with_stokes": True,
    },
]

# Projection, boundary-condition, and output settings for the optimization loop.
BETA_PROJ_VALUE = BETA_PROJ_SCHEDULE[0]
ETA_I = 0.50
QUADRATURE_DEGREE = 6
FILTER_RADIUS_IN_CELLS = 2.0

OUTLET_BC_TYPE = "pressure"
OUTLET_PRESSURE_VALUE = 0.0

ENABLE_PRESSURE_PIN = False
PRESSURE_PIN_POINT = (DOMAIN_X_MIN, DOMAIN_Y_MIN)
RESULTS_ROOT_NAME = "Results_Full/Results_DoublePipeBorrvall_TurbulentTO_Full"

MARK = {"generic": 0, "walls": 1, "inlet": (2, 3), "outlet": (4, 5)}


def between(value, limits, eps=DOLFIN_EPS):
    return (limits[0] - eps <= value) and (value <= limits[1] + eps)


class VerticalPortBoundary(SubDomain):
    def __init__(self, x_location, tol, y_min, y_max):
        super().__init__()
        self._x_location = x_location
        self._tol = tol
        self._y_min = y_min
        self._y_max = y_max

    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], self._x_location, self._tol) and between(
            x[1], (self._y_min, self._y_max), self._tol
        )


class WallsBoundary(SubDomain):
    def __init__(self, x_min, x_max, y_min, y_max, tol, inlet_segments, outlet_segments):
        super().__init__()
        self._x_min = x_min
        self._x_max = x_max
        self._y_min = y_min
        self._y_max = y_max
        self._tol = tol
        self._inlet_segments = inlet_segments
        self._outlet_segments = outlet_segments

    def _inside_any_segment(self, y_value, segments):
        return any(between(y_value, segment, self._tol) for segment in segments)

    def inside(self, x, on_boundary):
        left_wall_outside_inlets = near(x[0], self._x_min, self._tol) and not self._inside_any_segment(
            x[1], self._inlet_segments
        )
        right_wall_outside_outlets = near(x[0], self._x_max, self._tol) and not self._inside_any_segment(
            x[1], self._outlet_segments
        )
        top_wall = near(x[1], self._y_max, self._tol)
        bottom_wall = near(x[1], self._y_min, self._tol)
        return on_boundary and (
            left_wall_outside_inlets or right_wall_outside_outlets or top_wall or bottom_wall
        )


def _validate_port_configuration():
    inlet_markers = list(MARK["inlet"])
    outlet_markers = list(MARK["outlet"])
    if len(inlet_markers) != len(INLET_SEGMENTS):
        raise ValueError("MARK['inlet'] length must match INLET_SEGMENTS length.")
    if len(outlet_markers) != len(OUTLET_SEGMENTS):
        raise ValueError("MARK['outlet'] length must match OUTLET_SEGMENTS length.")
    if len(U_MAX_INLETS) != len(INLET_SEGMENTS):
        raise ValueError("U_MAX_INLETS length must match INLET_SEGMENTS length.")
    if len(U_MAX_OUTLETS) != len(OUTLET_SEGMENTS):
        raise ValueError("U_MAX_OUTLETS length must match OUTLET_SEGMENTS length.")


_validate_port_configuration()


def mark_boundaries(mesh):
    boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    boundaries.set_all(MARK["generic"])

    WallsBoundary(
        DOMAIN_X_MIN,
        DOMAIN_X_MAX,
        DOMAIN_Y_MIN,
        DOMAIN_Y_MAX,
        TOL,
        INLET_SEGMENTS,
        OUTLET_SEGMENTS,
    ).mark(boundaries, MARK["walls"])

    for marker, segment in zip(MARK["inlet"], INLET_SEGMENTS):
        VerticalPortBoundary(DOMAIN_X_MIN, TOL, segment[0], segment[1]).mark(boundaries, marker)
    for marker, segment in zip(MARK["outlet"], OUTLET_SEGMENTS):
        VerticalPortBoundary(DOMAIN_X_MAX, TOL, segment[0], segment[1]).mark(boundaries, marker)
    return boundaries


def _build_horizontal_profile(u_max, y_min, y_max):
    y_center = 0.5 * (y_min + y_max)
    width = y_max - y_min
    return Expression(
        ("u_max * (1 - pow(2.0 * (x[1] - y_c) / width, 2))", "0.0"),
        degree=2,
        u_max=u_max,
        y_c=y_center,
        width=width,
    )


def build_velocity_profile_sets():
    inlet_profiles = [
        _build_horizontal_profile(u_max, segment[0], segment[1])
        for u_max, segment in zip(U_MAX_INLETS, INLET_SEGMENTS)
    ]
    if OUTLET_BC_TYPE == "pressure":
        return inlet_profiles, []

    outlet_profiles = [
        _build_horizontal_profile(u_max, segment[0], segment[1])
        for u_max, segment in zip(U_MAX_OUTLETS, OUTLET_SEGMENTS)
    ]
    return inlet_profiles, outlet_profiles


def _inside_any_segment(y_value, segments, tol=TOL):
    return any(between(y_value, segment, tol) for segment in segments)


def _passive_strip_mode():
    return str(globals().get("INLET_PASSIVE_STRIP_MODE", "fluid_only")).strip().lower()


def build_density_bounds(mesh, density_space):
    if not ENABLE_INLET_PASSIVE_STRIP:
        return 0.0, 1.0

    lower = Function(density_space)
    upper = Function(density_space)

    lower_values = np.zeros(density_space.dim())
    upper_values = np.ones(density_space.dim())
    dofmap = density_space.dofmap()
    inlet_block_x_max = DOMAIN_X_MIN + INLET_PASSIVE_STRIP_LENGTH
    strip_mode = _passive_strip_mode()
    if strip_mode not in {"fluid_only", "block"}:
        raise ValueError(
            "INLET_PASSIVE_STRIP_MODE must be either 'fluid_only' or 'block'. "
            "Got {!r}.".format(strip_mode)
        )

    for cell in cells(mesh):
        dof = dofmap.cell_dofs(cell.index())[0]
        midpoint = cell.midpoint()
        x_coord = midpoint.x()
        y_coord = midpoint.y()

        if DOMAIN_X_MIN - DOLFIN_EPS <= x_coord <= inlet_block_x_max + DOLFIN_EPS:
            if _inside_any_segment(y_coord, INLET_SEGMENTS):
                lower_values[dof] = 1.0
                upper_values[dof] = 1.0
            elif strip_mode == "block":
                lower_values[dof] = 0.0
                upper_values[dof] = 0.0

    lower.vector().set_local(lower_values)
    lower.vector().apply("insert")
    upper.vector().set_local(upper_values)
    upper.vector().apply("insert")
    return lower, upper


def build_volume_region(mesh, density_space):
    if not ENABLE_INLET_PASSIVE_STRIP:
        return 1.0

    volume_region = Function(density_space)
    region_values = np.ones(density_space.dim())
    dofmap = density_space.dofmap()
    inlet_block_x_max = DOMAIN_X_MIN + INLET_PASSIVE_STRIP_LENGTH
    strip_mode = _passive_strip_mode()

    for cell in cells(mesh):
        dof = dofmap.cell_dofs(cell.index())[0]
        midpoint = cell.midpoint()
        x_coord = midpoint.x()
        y_coord = midpoint.y()
        if (
            DOMAIN_X_MIN - DOLFIN_EPS <= x_coord <= inlet_block_x_max + DOLFIN_EPS
            and (
                strip_mode == "block"
                or _inside_any_segment(y_coord, INLET_SEGMENTS)
            )
        ):
            region_values[dof] = 0.0

    volume_region.vector().set_local(region_values)
    volume_region.vector().apply("insert")
    return volume_region
