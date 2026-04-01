import os
from math import pi
from dolfin import DOLFIN_EPS, Expression, Mesh, MeshFunction, MPI, SubDomain, XDMFFile, near
from Utilities_LaminarTO import load_mesh_from_xdmf


# ===================================================================
# Configuration: Borrvall Pipe Bend - Turbulent Full (SA, Re = 2000)
#
# One inlet on the left wall and one outlet on the bottom wall.
# This config targets the monolithic full-state solver (u, p, nu_tilde),
# while still keeping the wall-distance field external to the state.
# ===================================================================

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)

# Mesh files
# Generate via: cd Meshes/PipeBendBorrvall && python3 pipe_bend_gmsh.py && python3 gmsh_to_xdmf.py
mesh_files = {
    "MESH_DIRECTORY": os.path.join(REPO_ROOT, "Meshes/PipeBendBorrvall/mesh.xdmf"),
}


def create_design_mesh():
    return load_mesh_from_xdmf(mesh_files["MESH_DIRECTORY"], MPI.comm_world)


# Domain and mesh
L = 1.0
N = 120  # reference resolution used to generate the Gmsh mesh (LC = L/N)
TOL = DOLFIN_EPS

# Geometry parameters (Borrvall 2003 pipe bend case)
INLET_WIDTH = 0.2
INLET_TOP_OFFSET = 0.2
OUTLET_WIDTH = 0.2
OUTLET_RIGHT_OFFSET = 0.2

# Flow settings
MU_FLUID_VALUE = 1.0e-4
RHO_FLUID_VALUE = 1.0
U_MAX_INLET = 1.0
U_MAX_OUTLET = 1.0

# ----------------------------------------------------------------------------------------------
# Reynolds number: Re = U_MAX_INLET * INLET_WIDTH * RHO_FLUID_VALUE / MU_FLUID_VALUE = 2,000
# ----------------------------------------------------------------------------------------------

# Spalart-Allmaras settings
SA_MUT_RATIO = 5.0
SA_DISTANCE_RELAXATION = 0.01
SA_SMOOTH_ABS_EPS = 1.0e-12
SA_INIT_WALL_DIST_SCALE = 0.05 * L
SA_NU_TILDE_FLOOR = 1.0e-12
SA_NU_TILDE_PENALTY_ALPHA = 1.0e3
SA_NU_TILDE_PENALTY_N = 3.0

# Penalized reciprocal wall-distance equation (Yoon 2016 Eq. 25)
SA_USE_PENALIZED_WALL_DISTANCE = True
SA_WALL_SIGMA = SA_DISTANCE_RELAXATION
SA_WALL_G0 = 20.0
SA_WALL_PENALTY_ALPHA = 1.0e3
SA_WALL_PENALTY_N = 3.0
SA_WALL_G_FLOOR = 1.0e-8

# Topology optimization settings
VOL_FRAC = 0.08 * pi
MAX_INNER_ITERATIONS_SCHEDULE = [120, 120, 120, 120, 120, 120, 100, 100, 100]
OBJECTIVE_CONVERGENCE_TOL = 5e-5
OBJECTIVE_STREAK_TO_STOP = 5

Q_PENAL_SCHEDULE = [0.05, 0.08, 0.10, 0.15, 0.25, 0.40, 0.60, 0.80, 1.00]
MOVE_LIMIT_SCHEDULE = [0.08, 0.08, 0.07, 0.06, 0.05, 0.035, 0.025, 0.015, 0.01]
BETA_PROJ_SCHEDULE = [0.1, 0.15, 0.25, 0.4, 0.6, 0.9, 1.25, 1.75, 2.5]

# Monolithic full-state solver settings
SNES_LINEAR_SOLVER = "mumps"
FULL_STATE_LINEAR_SOLVER = "mumps"
FULL_STATE_SNES_METHOD = "newtontr"
FULL_STATE_SNES_LINE_SEARCH = "bt"
FULL_STATE_SNES_RTOL = 1.0e-6
FULL_STATE_SNES_ATOL = 1.0e-8
FULL_STATE_SNES_MAX_ITERS = 50
FULL_STATE_RESTART_WITH_STOKES = True

BETA_PROJ_VALUE = BETA_PROJ_SCHEDULE[0]
ETA_I = 0.50
QUADRATURE_DEGREE = 6
FILTER_RADIUS_IN_CELLS = 3.0

OUTLET_BC_TYPE = "pressure"
OUTLET_PRESSURE_VALUE = 0.0

ENABLE_PRESSURE_PIN = False
PRESSURE_PIN_POINT = (0.0, 0.0)
RESULTS_ROOT_NAME_FULL = "Results_Full/Results_PipeBendBorrvall_TurbulentTO_Full"

MARK = {"generic": 0, "walls": 1, "inlet": 2, "outlet": 3}


def between(value, limits, eps=DOLFIN_EPS):
    return (limits[0] - eps <= value) and (value <= limits[1] + eps)


def compute_port_extents(l_box, inlet_top_offset, inlet_width, outlet_right_offset, outlet_width):
    inlet_y_max = l_box - inlet_top_offset
    inlet_y_min = inlet_y_max - inlet_width
    outlet_x_max = l_box - outlet_right_offset
    outlet_x_min = outlet_x_max - outlet_width
    return inlet_y_min, inlet_y_max, outlet_x_min, outlet_x_max


class InletBoundary(SubDomain):
    def __init__(self, tol, inlet_y_min, inlet_y_max):
        super().__init__()
        self._tol = tol
        self._inlet_y_min = inlet_y_min
        self._inlet_y_max = inlet_y_max

    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 0.0, self._tol) and between(
            x[1], (self._inlet_y_min, self._inlet_y_max), self._tol
        )


class OutletBoundary(SubDomain):
    def __init__(self, tol, outlet_x_min, outlet_x_max):
        super().__init__()
        self._tol = tol
        self._outlet_x_min = outlet_x_min
        self._outlet_x_max = outlet_x_max

    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], 0.0, self._tol) and between(
            x[0], (self._outlet_x_min, self._outlet_x_max), self._tol
        )


class WallsBoundary(SubDomain):
    def __init__(self, l_box, tol, inlet_y_min, inlet_y_max, outlet_x_min, outlet_x_max):
        super().__init__()
        self._l_box = l_box
        self._tol = tol
        self._inlet_y_min = inlet_y_min
        self._inlet_y_max = inlet_y_max
        self._outlet_x_min = outlet_x_min
        self._outlet_x_max = outlet_x_max

    def inside(self, x, on_boundary):
        left_wall_outside_inlet = near(x[0], 0.0, self._tol) and not between(
            x[1], (self._inlet_y_min, self._inlet_y_max), self._tol
        )
        bottom_wall_outside_outlet = near(x[1], 0.0, self._tol) and not between(
            x[0], (self._outlet_x_min, self._outlet_x_max), self._tol
        )
        right_wall = near(x[0], self._l_box, self._tol)
        top_wall = near(x[1], self._l_box, self._tol)
        return on_boundary and (
            left_wall_outside_inlet or bottom_wall_outside_outlet or right_wall or top_wall
        )


def mark_boundaries(mesh):
    inlet_y_min, inlet_y_max, outlet_x_min, outlet_x_max = compute_port_extents(
        L, INLET_TOP_OFFSET, INLET_WIDTH, OUTLET_RIGHT_OFFSET, OUTLET_WIDTH
    )
    boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    boundaries.set_all(MARK["generic"])
    WallsBoundary(L, TOL, inlet_y_min, inlet_y_max, outlet_x_min, outlet_x_max).mark(boundaries, MARK["walls"])
    InletBoundary(TOL, inlet_y_min, inlet_y_max).mark(boundaries, MARK["inlet"])
    OutletBoundary(TOL, outlet_x_min, outlet_x_max).mark(boundaries, MARK["outlet"])
    return boundaries


def build_velocity_profile_sets():
    inlet_y_min, inlet_y_max, outlet_x_min, outlet_x_max = compute_port_extents(
        L, INLET_TOP_OFFSET, INLET_WIDTH, OUTLET_RIGHT_OFFSET, OUTLET_WIDTH
    )
    y_inlet_center = 0.5 * (inlet_y_min + inlet_y_max)
    u_inlet = Expression(
        ("u_max * (1 - pow(2.0 * (x[1] - y_c) / width, 2))", "0.0"),
        degree=2,
        u_max=U_MAX_INLET,
        y_c=y_inlet_center,
        width=INLET_WIDTH,
    )
    if OUTLET_BC_TYPE == "pressure":
        return [u_inlet], []

    x_outlet_center = 0.5 * (outlet_x_min + outlet_x_max)
    u_outlet = Expression(
        ("0.0", "-u_max * (1 - pow(2.0 * (x[0] - x_c) / width, 2))"),
        degree=2,
        u_max=U_MAX_OUTLET,
        x_c=x_outlet_center,
        width=OUTLET_WIDTH,
    )
    return [u_inlet], [u_outlet]
