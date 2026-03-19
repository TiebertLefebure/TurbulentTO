import os
from dolfin import DOLFIN_EPS, Expression, Mesh, MeshFunction, MPI, SubDomain, XDMFFile, near
from Utilities_LaminarTO import load_mesh_from_xdmf


# -----------------------------------------------------------------
# Configuration file for Borrvall Double Pipe case (Laminar)
# -----------------------------------------------------------------

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# Mesh files
# Generate via: cd Meshes/DoublePipe && python3 double_pipe_gmsh.py && python3 gmsh_to_xdmf.py
mesh_files = {
    'MESH_DIRECTORY': os.path.join(THIS_DIR, 'Meshes/DoublePipe/mesh.xdmf'),
}


def create_design_mesh():
    return load_mesh_from_xdmf(mesh_files['MESH_DIRECTORY'], MPI.comm_world)


# Domain and mesh
DOMAIN_X_MIN = 0.0
DOMAIN_Y_MIN = 0.0
DOMAIN_X_MAX = 1.5  # DOMAIN_X_MAX = 1.0 for square design domain
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

# Flow settings
MU_FLUID_VALUE = 1.0e-1
RHO_FLUID_VALUE = 1.0
U_MAX_INLETS = [1.0, 1.0]
U_MAX_OUTLETS = [1.0, 1.0]


# ----------------------------------------------------------------------------------------------
# Reynolds number: Re = U_MAX_INLET * PORT_WIDTH * RHO_FLUID_VALUE / MU_FLUID_VALUE = 1.66
# ----------------------------------------------------------------------------------------------


# Topology optimization settings
VOL_FRAC = 1.0 / 3.0
MAX_INNER_ITERATIONS = 80
OBJECTIVE_CONVERGENCE_TOL = 5e-5
OBJECTIVE_STREAK_TO_STOP = 5

Q_PENAL_SCHEDULE = [0.005, 0.01, 0.03, 0.05, 0.1]
MOVE_LIMIT_SCHEDULE = [0.03, 0.03, 0.02, 0.015, 0.01]
BETA_PROJ_SCHEDULE = [0.3, 0.5, 1.0, 2.0, 4.0]

SNES_LINEAR_SOLVER = "mumps"
FILTER_RADIUS_IN_CELLS = 3.0
FORWARD_SNES_RTOL = 5.0e-7
FORWARD_SNES_ATOL = 1.0e-9
ADJOINT_SNES_RTOL = 5.0e-7
ADJOINT_SNES_ATOL = 1.0e-9
SNES_MAX_ITERS = 200

BETA_PROJ_VALUE = BETA_PROJ_SCHEDULE[0] # Initial value = 0.3
ETA_I = 0.50

ENABLE_PRESSURE_PIN = True
PRESSURE_PIN_POINT = (DOMAIN_X_MIN, DOMAIN_Y_MIN)
RESULTS_ROOT_NAME = "Results_DoublePipe_LaminarTO"

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
    outlet_profiles = [
        _build_horizontal_profile(u_max, segment[0], segment[1])
        for u_max, segment in zip(U_MAX_OUTLETS, OUTLET_SEGMENTS)
    ]
    return inlet_profiles, outlet_profiles
