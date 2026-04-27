import os
import numpy as np
from math import pi
from dolfin import DOLFIN_EPS, Expression, Function, MeshFunction, MPI, SubDomain, cells, near
from Utilities_SharedTO import load_mesh_from_xdmf


# ===================================================================
# Configuration: Borrvall Pipe Bend - Turbulent SemiFrozen (SA, Re = 2000)
#
# One inlet on the left wall and one outlet on the bottom wall.
# This config targets the SemiFrozen adjoint: the primal state remains
# monolithic in (u, p, nu_tilde), while the reciprocal wall-distance G is
# updated externally instead of entering the adjoint state.
# ===================================================================

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Mesh path for this benchmark geometry.
# Generate via: python3 Meshes/generate_borrvall_guided_meshes.py
mesh_files = {
    "MESH_DIRECTORY": os.path.join(REPO_ROOT, "Meshes/PipeBendBorrvall/mesh_guided.xdmf"),
}


def create_design_mesh():
    return load_mesh_from_xdmf(mesh_files["MESH_DIRECTORY"], MPI.comm_world)


# Geometry and reference meshing parameters.
L = 1.0
N = 120  # reference resolution used to generate the Gmsh mesh (LC = L/N)
INLET_EXTENSION_WIDTH = 0.2 * L
OUTLET_EXTENSION_HEIGHT = 0.2 * L
DESIGN_X_MIN = 0.0
DESIGN_Y_MIN = 0.0
DESIGN_X_MAX = L
DESIGN_Y_MAX = L
DOMAIN_X_MIN = DESIGN_X_MIN - INLET_EXTENSION_WIDTH
DOMAIN_Y_MIN = DESIGN_Y_MIN - OUTLET_EXTENSION_HEIGHT
DOMAIN_X_MAX = DESIGN_X_MAX
DOMAIN_Y_MAX = DESIGN_Y_MAX
TOL = DOLFIN_EPS

# Geometry parameters
INLET_HEIGHT = 0.2 * L
INLET_TOP_OFFSET = 0.2 * L
OUTLET_WIDTH = 0.2 * L
OUTLET_RIGHT_OFFSET = 0.2 * L

# Flow and Brinkman parameters for the monolithic primal solve.
MU_FLUID_VALUE = 1.0e-4
RHO_FLUID_VALUE = 1.0
U_MAX_INLET = 1.0
U_MAX_OUTLET = 1.0

# ==============================================================================================
# Reynolds number: Re = U_MAX_INLET * INLET_HEIGHT * RHO_FLUID_VALUE / MU_FLUID_VALUE = 2,000
# ==============================================================================================


# SA transport parameters for the monolithic primal state.
SA_TURBULENCE_INTENSITY = 0.075
SA_TURBULENCE_LENGTH_SCALE_RATIO = 0.05
SA_REFERENCE_LENGTH = INLET_HEIGHT
SA_SMOOTH_ABS_EPS = 1.0e-12
SA_INIT_WALL_DIST_SCALE = 0.05 * L
SA_NU_TILDE_FLOOR = 1.0e-12
SA_NU_TILDE_PENALTY_ALPHA = 1.0e3
SA_NU_TILDE_PENALTY_N = 3.0

# Relaxed wall equation parameters for the external reciprocal distance solve.
SA_WALL_SIGMA = 0.01
SA_WALL_G0 = 20.0
SA_WALL_PENALTY_ALPHA = 1.0e3
SA_WALL_PENALTY_N = 3.0
SA_WALL_G_FLOOR = 1.0e-8

# MMA objective and continuation parameters for the topology update.
VOL_FRAC = 0.08 * pi
OBJECTIVE_CONVERGENCE_TOL = 1e-5
OBJECTIVE_STREAK_TO_STOP = 5

Q_PENAL_SCHEDULE = [0.05, 0.08, 0.10, 0.15, 0.25, 0.40, 0.60, 0.80, 1.00]
MOVE_LIMIT_SCHEDULE = [0.08, 0.08, 0.07, 0.06, 0.05, 0.035, 0.025, 0.015, 0.01]
BETA_PROJ_SCHEDULE = [0.1, 0.15, 0.25, 0.4, 0.6, 0.9, 1.25, 1.75, 2.5]
MAX_INNER_ITERATIONS_SCHEDULE = [120, 120, 120, 120, 120, 120, 100, 100, 100]

# SemiFrozen primal-state solve parameters.
LINEAR_SOLVER = "mumps"
STATE_SOLVE_METHOD = "newtontr"
STATE_RTOL = 1.0e-6
STATE_ATOL = 1.0e-8
STATE_MAX_ITERS = 120
STATE_INITIAL_SA_SWEEPS = 4
# Ramp the turbulent-viscosity feedback into the momentum equations instead
# of forcing the first monolithic Newton solve to handle the full coupling at
# once from a Stokes/SA warm start.
STATE_TURBULENCE_COUPLING_SCHEDULE = [
    {"weight": 0.0, "max_iters": 220, "atol": 8.0e-4},
    {"weight": 0.35, "max_iters": 180, "atol": 5.0e-4},
    {"weight": 0.70, "max_iters": 180, "atol": 2.0e-4},
    {"weight": 1.0},
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
FILTER_RADIUS_IN_CELLS = 3.0

OUTLET_BC_TYPE = "pressure"
OUTLET_PRESSURE_VALUE = 0.0

ENABLE_PRESSURE_PIN = False
RESULTS_ROOT_NAME = "Results_SemiFrozen/Results_PipeBendBorrvall_TurbulentTO_SemiFrozen"

MARK = {"generic": 0, "walls": 1, "inlet": 2, "outlet": 3}


def between(value, limits, eps=DOLFIN_EPS):
    return (limits[0] - eps <= value) and (value <= limits[1] + eps)


def compute_port_extents(l_box, inlet_top_offset, inlet_height, outlet_right_offset, outlet_width):
    inlet_y_max = l_box - inlet_top_offset
    inlet_y_min = inlet_y_max - inlet_height
    outlet_x_max = l_box - outlet_right_offset
    outlet_x_min = outlet_x_max - outlet_width
    return inlet_y_min, inlet_y_max, outlet_x_min, outlet_x_max


class InletBoundary(SubDomain):
    def __init__(self, x_location, tol, inlet_y_min, inlet_y_max):
        super().__init__()
        self._x_location = x_location
        self._tol = tol
        self._inlet_y_min = inlet_y_min
        self._inlet_y_max = inlet_y_max

    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], self._x_location, self._tol) and between(
            x[1], (self._inlet_y_min, self._inlet_y_max), self._tol
        )


class OutletBoundary(SubDomain):
    def __init__(self, y_location, tol, outlet_x_min, outlet_x_max):
        super().__init__()
        self._y_location = y_location
        self._tol = tol
        self._outlet_x_min = outlet_x_min
        self._outlet_x_max = outlet_x_max

    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], self._y_location, self._tol) and between(
            x[0], (self._outlet_x_min, self._outlet_x_max), self._tol
        )


class WallsBoundary(SubDomain):
    def __init__(
        self,
        design_x_min,
        design_y_min,
        design_x_max,
        design_y_max,
        domain_x_min,
        domain_y_min,
        tol,
        inlet_y_min,
        inlet_y_max,
        outlet_x_min,
        outlet_x_max,
    ):
        super().__init__()
        self._design_x_min = design_x_min
        self._design_y_min = design_y_min
        self._design_x_max = design_x_max
        self._design_y_max = design_y_max
        self._domain_x_min = domain_x_min
        self._domain_y_min = domain_y_min
        self._tol = tol
        self._inlet_y_min = inlet_y_min
        self._inlet_y_max = inlet_y_max
        self._outlet_x_min = outlet_x_min
        self._outlet_x_max = outlet_x_max

    def inside(self, x, on_boundary):
        left_wall_outside_inlet = near(x[0], self._design_x_min, self._tol) and not between(
            x[1], (self._inlet_y_min, self._inlet_y_max), self._tol
        )
        right_wall = near(x[0], self._design_x_max, self._tol)
        top_wall = near(x[1], self._design_y_max, self._tol)
        bottom_wall_outside_outlet = near(x[1], self._design_y_min, self._tol) and not between(
            x[0], (self._outlet_x_min, self._outlet_x_max), self._tol
        )
        inlet_strip_caps = (
            (near(x[1], self._inlet_y_min, self._tol) or near(x[1], self._inlet_y_max, self._tol))
            and between(x[0], (self._domain_x_min, self._design_x_min), self._tol)
        )
        outlet_strip_caps = (
            (near(x[0], self._outlet_x_min, self._tol) or near(x[0], self._outlet_x_max, self._tol))
            and between(x[1], (self._domain_y_min, self._design_y_min), self._tol)
        )
        return on_boundary and (
            left_wall_outside_inlet
            or right_wall
            or top_wall
            or bottom_wall_outside_outlet
            or inlet_strip_caps
            or outlet_strip_caps
        )


def mark_boundaries(mesh):
    inlet_y_min, inlet_y_max, outlet_x_min, outlet_x_max = compute_port_extents(
        L, INLET_TOP_OFFSET, INLET_HEIGHT, OUTLET_RIGHT_OFFSET, OUTLET_WIDTH
    )
    boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    boundaries.set_all(MARK["generic"])
    WallsBoundary(
        DESIGN_X_MIN,
        DESIGN_Y_MIN,
        DESIGN_X_MAX,
        DESIGN_Y_MAX,
        DOMAIN_X_MIN,
        DOMAIN_Y_MIN,
        TOL,
        inlet_y_min,
        inlet_y_max,
        outlet_x_min,
        outlet_x_max,
    ).mark(boundaries, MARK["walls"])
    InletBoundary(DOMAIN_X_MIN, TOL, inlet_y_min, inlet_y_max).mark(boundaries, MARK["inlet"])
    OutletBoundary(DOMAIN_Y_MIN, TOL, outlet_x_min, outlet_x_max).mark(boundaries, MARK["outlet"])
    return boundaries


def build_velocity_profile_sets():
    inlet_y_min, inlet_y_max, outlet_x_min, outlet_x_max = compute_port_extents(
        L, INLET_TOP_OFFSET, INLET_HEIGHT, OUTLET_RIGHT_OFFSET, OUTLET_WIDTH
    )
    u_inlet = Expression(
        ("u_max", "0.0"),
        degree=0,
        u_max=U_MAX_INLET,
    )
    if OUTLET_BC_TYPE == "pressure":
        return [u_inlet], []

    u_outlet = Expression(
        ("0.0", "-u_max"),
        degree=0,
        u_max=U_MAX_OUTLET,
    )
    return [u_inlet], [u_outlet]


def build_density_bounds(mesh, density_space):
    non_design_fluid_lower = Expression(
        "(x[0] < x_min || x[1] < y_min || x[0] > x_max || x[1] > y_max) ? 1.0 : 0.0",
        degree=0,
        x_min=DESIGN_X_MIN,
        x_max=DESIGN_X_MAX,
        y_min=DESIGN_Y_MIN,
        y_max=DESIGN_Y_MAX,
    )
    return non_design_fluid_lower, 1.0


def build_volume_region(mesh, density_space):
    return Expression(
        "(x[0] >= x_min && x[0] <= x_max && x[1] >= y_min && x[1] <= y_max) ? 1.0 : 0.0",
        degree=0,
        x_min=DESIGN_X_MIN,
        x_max=DESIGN_X_MAX,
        y_min=DESIGN_Y_MIN,
        y_max=DESIGN_Y_MAX,
    )


def build_objective_region(mesh, density_space):
    return build_volume_region(mesh, density_space)
