import os
import numpy as np
from dolfin import DOLFIN_EPS, Expression, Function, MeshFunction, MPI, SubDomain, cells, near
from Utilities_SharedTO import load_mesh_from_xdmf


# =======================================================================
# Configuration: Borrvall Double Pipe - Turbulent Full (SA, Re = 1,660)
#
# Two symmetric ports on the left and right boundaries.
# This config targets the Full adjoint: the reciprocal wall-distance G enters
# the monolithic primal state and adjoint system as (u, p, nu_tilde, G).
# =======================================================================

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Mesh path for this benchmark geometry.
# Generate via: python3 Meshes/generate_borrvall_guided_meshes.py
mesh_files = {
    "MESH_DIRECTORY": os.path.join(REPO_ROOT, "Meshes/DoublePipeBorrvall/mesh_guided.xdmf"),
}


def create_design_mesh():
    return load_mesh_from_xdmf(mesh_files["MESH_DIRECTORY"], MPI.comm_world)


# Geometry and reference meshing parameters.
L = 1.0
INLET_EXTENSION_WIDTH = 0.2 * L
OUTLET_EXTENSION_WIDTH = 0.2 * L
DESIGN_X_MIN = 0.0
DESIGN_Y_MIN = 0.0
DESIGN_X_MAX = 1.5 * L
DESIGN_Y_MAX = L
DOMAIN_X_MIN = DESIGN_X_MIN - INLET_EXTENSION_WIDTH
DOMAIN_Y_MIN = DESIGN_Y_MIN
DOMAIN_X_MAX = DESIGN_X_MAX + OUTLET_EXTENSION_WIDTH
DOMAIN_Y_MAX = DESIGN_Y_MAX
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

# Flow and Brinkman parameters for the monolithic primal solve.
MU_FLUID_VALUE = 1.0e-4
RHO_FLUID_VALUE = 1.0
U_MAX_INLETS = [1.0, 1.0]
U_MAX_OUTLETS = [1.0, 1.0]


# =============================================================================================
# Reynolds number: Re = U_MAX_INLET * PORT_WIDTH * RHO_FLUID_VALUE / MU_FLUID_VALUE = 1,600
# =============================================================================================


# SA transport parameters for the monolithic primal state.
SA_TURBULENCE_INTENSITY = 0.075
SA_TURBULENCE_LENGTH_SCALE_RATIO = 0.06
SA_REFERENCE_LENGTH = PORT_WIDTH
SA_SMOOTH_ABS_EPS = 1.0e-12
SA_INIT_WALL_DIST_SCALE = 0.05 * DOMAIN_Y_MAX
SA_NU_TILDE_FLOOR = 1.0e-12
SA_NU_TILDE_PENALTY_ALPHA = 1.0e3
SA_NU_TILDE_PENALTY_N = 3.0

# Relaxed wall equation parameters for the coupled reciprocal distance state.
SA_WALL_SIGMA = 0.01
SA_WALL_G0 = 20.0
SA_WALL_PENALTY_ALPHA = 1.0e3
SA_WALL_PENALTY_N = 3.0
SA_WALL_G_FLOOR = 1.0e-8

# MMA objective and continuation parameters for the topology update.
VOL_FRAC = 1.0 / 3.0
MAX_INNER_ITERATIONS_SCHEDULE = [35, 80, 100, 120, 120, 120, 100, 100]
OBJECTIVE_CONVERGENCE_TOL = 1e-5
OBJECTIVE_STREAK_TO_STOP = 5

Q_PENAL_SCHEDULE = [0.05, 0.1, 0.1, 0.2, 0.5, 1.0, 1.0, 1.0]
MOVE_LIMIT_SCHEDULE = [0.08, 0.06, 0.04, 0.02, 0.01, 0.005, 0.003, 0.002]
BETA_PROJ_SCHEDULE = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]

# Full primal-state (u, p, nu_tilde, G) solve parameters.
LINEAR_SOLVER = "mumps"
STATE_SOLVE_METHOD = "newtonls"
STATE_LINE_SEARCH = "bt"
STATE_RTOL = 1.0e-6
STATE_ATOL = 1.0e-8
STATE_MAX_ITERS = 120
STATE_ERROR_ON_NONCONVERGENCE = False
STATE_ACCEPTED_RESIDUAL_FACTOR = 1.0
STATE_INITIAL_SA_SWEEPS = 8
STATE_INITIAL_SA_RELAXATION = 0.35
# Ramp both Navier-Stokes convection and turbulent-viscosity feedback into the
# momentum equations. The first full double-pipe solve starts from a linear
# Stokes-Brinkman field, so jumping directly to full convection at Re ~= 1600
# can stall before the turbulence homotopy has a chance to help.
STATE_TURBULENCE_COUPLING_SCHEDULE = [
    {"convection_weight": 0.00, "weight": 0.00, "max_iters": 160, "atol": 2.0e-4, "accept_norm": 1.0e-1},
    {"convection_weight": 0.20, "weight": 0.00, "max_iters": 180, "atol": 2.0e-4, "accept_norm": 3.0e-1},
    {"convection_weight": 0.45, "weight": 0.00, "max_iters": 200, "atol": 2.0e-4, "accept_norm": 7.5e-1},
    {"convection_weight": 0.70, "weight": 0.20, "max_iters": 220, "atol": 2.0e-4, "accept_norm": 1.0e0},
    {"convection_weight": 0.90, "weight": 0.45, "max_iters": 240, "atol": 1.5e-4, "accept_norm": 1.0e0},
    {"convection_weight": 1.00, "weight": 0.70, "max_iters": 260, "atol": 1.2e-4, "accept_norm": 7.5e-1},
    {"convection_weight": 1.00, "weight": 0.90, "max_iters": 280, "atol": 1.0e-4, "accept_norm": 3.0e-1},
    {"convection_weight": 1.00, "weight": 1.00, "max_iters": 320, "atol": 1.0e-4},
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
    def __init__(
        self,
        design_x_min,
        design_x_max,
        domain_x_min,
        domain_x_max,
        y_min,
        y_max,
        tol,
        inlet_segments,
        outlet_segments,
    ):
        super().__init__()
        self._design_x_min = design_x_min
        self._design_x_max = design_x_max
        self._domain_x_min = domain_x_min
        self._domain_x_max = domain_x_max
        self._y_min = y_min
        self._y_max = y_max
        self._tol = tol
        self._inlet_segments = inlet_segments
        self._outlet_segments = outlet_segments

    def _inside_any_segment(self, y_value, segments):
        return any(between(y_value, segment, self._tol) for segment in segments)

    def inside(self, x, on_boundary):
        left_wall_outside_inlets = near(x[0], self._design_x_min, self._tol) and not self._inside_any_segment(
            x[1], self._inlet_segments
        )
        right_wall_outside_outlets = near(x[0], self._design_x_max, self._tol) and not self._inside_any_segment(
            x[1], self._outlet_segments
        )
        top_wall = near(x[1], self._y_max, self._tol)
        bottom_wall = near(x[1], self._y_min, self._tol)
        inlet_strip_caps = any(
            (near(x[1], y_edge, self._tol) and between(x[0], (self._domain_x_min, self._design_x_min), self._tol))
            for segment in self._inlet_segments
            for y_edge in segment
        )
        outlet_strip_caps = any(
            (near(x[1], y_edge, self._tol) and between(x[0], (self._design_x_max, self._domain_x_max), self._tol))
            for segment in self._outlet_segments
            for y_edge in segment
        )
        return on_boundary and (
            left_wall_outside_inlets
            or right_wall_outside_outlets
            or top_wall
            or bottom_wall
            or inlet_strip_caps
            or outlet_strip_caps
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
        DESIGN_X_MIN,
        DESIGN_X_MAX,
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


def _build_uniform_horizontal_profile(u_max):
    return Expression(("u_max", "0.0"), degree=0, u_max=u_max)


def build_velocity_profile_sets():
    inlet_profiles = [
        _build_uniform_horizontal_profile(u_max)
        for u_max in U_MAX_INLETS
    ]
    if OUTLET_BC_TYPE == "pressure":
        return inlet_profiles, []

    outlet_profiles = [
        _build_uniform_horizontal_profile(u_max)
        for u_max in U_MAX_OUTLETS
    ]
    return inlet_profiles, outlet_profiles


def build_density_bounds(mesh, density_space):
    non_design_fluid_lower = Expression(
        "(x[0] < x_min || x[0] > x_max) ? 1.0 : 0.0",
        degree=0,
        x_min=DESIGN_X_MIN,
        x_max=DESIGN_X_MAX,
    )
    return non_design_fluid_lower, 1.0


def build_volume_region(mesh, density_space):
    return Expression(
        "(x[0] >= x_min && x[0] <= x_max) ? 1.0 : 0.0",
        degree=0,
        x_min=DESIGN_X_MIN,
        x_max=DESIGN_X_MAX,
    )


def build_objective_region(mesh, density_space):
    return build_volume_region(mesh, density_space)
