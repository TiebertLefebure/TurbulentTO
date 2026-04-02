import os
import numpy as np
from dolfin import (
    DOLFIN_EPS,
    Expression,
    Function,
    MPI,
    MeshFunction,
    SubDomain,
    cells,
    near,
)
from Utilities_LaminarTO import load_mesh_from_xdmf


# ===================================================================
# Configuration: Dilgen 2018 2D flow manifold - Turbulent Full
#
# Paper data referenced by the corresponding frozen-SA case:
#   Re = 3500, Ub = 2.0 m/s, H = 0.1 m, nu = 5.7e-5 m^2/s
#   lambda = 2e3 s^-1, q = 0.1, filter radius = 0.028, beta = 1.5 -> 14
#   fluid volume fraction in design domain f = 0.43
#   outlet flow fractions = [0.3, 0.4, 0.3] * Fin
#
# Implementation choice for this repo:
#   keep the current SA-based manifold setup from the frozen case,
#   including geometry, passive fluid/solid regions, pressure outlets,
#   and outlet mass-flow constraints, but target the monolithic full
#   state solver (u, p, nu_tilde) while keeping the wall-distance field
#   external to the state.
# ===================================================================

# -------------------------------------------------------------------
# Geometry from Fig. 15 of Dilgen et al. 2018
# -------------------------------------------------------------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)

H = 0.1

# Main 10H x 10H design square
DESIGN_LENGTH = 10.0 * H
DESIGN_X_MIN = 0.0
DESIGN_Y_MIN = 0.0
DESIGN_X_MAX = DESIGN_X_MIN + DESIGN_LENGTH
DESIGN_Y_MAX = DESIGN_Y_MIN + DESIGN_LENGTH

# Left inlet block: 2H x 4H = H-solid / 2H-fluid / H-solid
INLET_BLOCK_X_MIN = -2.0 * H
INLET_BLOCK_X_MAX = DESIGN_X_MIN
INLET_BLOCK_Y_MIN = 6.0 * H
INLET_BLOCK_Y_MAX = 10.0 * H
INLET_HEIGHT = 2.0 * H
INLET_Y_MIN = 7.0 * H
INLET_Y_MAX = 9.0 * H

# Top outlet block: 4H x 10H = H-solid / 2H-fluid / H-solid
TOP_BLOCK_X_MIN = 6.0 * H
TOP_BLOCK_X_MAX = 10.0 * H
TOP_BLOCK_Y_MIN = DESIGN_Y_MAX
TOP_BLOCK_Y_MAX = 20.0 * H
TOP_OUTLET_X_MIN = 7.0 * H
TOP_OUTLET_X_MAX = 9.0 * H

# Right outlet block: 10H x 4H = H-solid / 2H-fluid / H-solid
RIGHT_BLOCK_X_MIN = DESIGN_X_MAX
RIGHT_BLOCK_X_MAX = 20.0 * H
RIGHT_BLOCK_Y_MIN = 4.0 * H
RIGHT_BLOCK_Y_MAX = 8.0 * H
RIGHT_OUTLET_Y_MIN = 5.0 * H
RIGHT_OUTLET_Y_MAX = 7.0 * H

# Bottom outlet block: 4H x 10H = H-solid / 2H-fluid / H-solid
BOTTOM_BLOCK_X_MIN = 0.0 * H
BOTTOM_BLOCK_X_MAX = 4.0 * H
BOTTOM_BLOCK_Y_MIN = -10.0 * H
BOTTOM_BLOCK_Y_MAX = DESIGN_Y_MIN
BOTTOM_OUTLET_X_MIN = 1.0 * H
BOTTOM_OUTLET_X_MAX = 3.0 * H

# Bounding box of the full non-rectangular computational domain.
DOMAIN_X_MIN = INLET_BLOCK_X_MIN
DOMAIN_Y_MIN = BOTTOM_BLOCK_Y_MIN
DOMAIN_X_MAX = RIGHT_BLOCK_X_MAX
DOMAIN_Y_MAX = TOP_BLOCK_Y_MAX

BOUNDARY_TOL = 1.0e-6
TOL = BOUNDARY_TOL

# Reference resolution used to generate the Gmsh mesh.
NX = 100
NY = 100

# Mesh files
# Generate via: cd Meshes/ManifoldDilgen && python3 manifold_dilgen_gmsh.py && python3 gmsh_to_xdmf.py
mesh_files = {
    'MESH_DIRECTORY': os.path.join(REPO_ROOT, 'Meshes/ManifoldDilgen/mesh.xdmf'),
}


def create_design_mesh():
    return load_mesh_from_xdmf(mesh_files['MESH_DIRECTORY'], MPI.comm_world)


# -------------------------------------------------------------------
# Flow settings
# -------------------------------------------------------------------
RHO_FLUID_VALUE = 1.0
MU_FLUID_VALUE = 5.7e-4
U_BULK_INLET = 2.0
U_MAX_INLET = 1.5 * U_BULK_INLET
ALPHA_SOLID = 2.0e3

# ----------------------------------------------------------------------------------------------
# Reynolds number: Re = U_BULK_INLET * H * RHO_FLUID_VALUE / MU_FLUID_VALUE = 350
# Defined as in Dilgen 2018
# ----------------------------------------------------------------------------------------------


# -------------------------------------------------------------------
# Spalart-Allmaras turbulence model settings
# -------------------------------------------------------------------
SA_MUT_RATIO = 5.0
SA_DISTANCE_RELAXATION = 0.01
SA_SMOOTH_ABS_EPS = 1.0e-12
SA_INIT_WALL_DIST_SCALE = 0.05 * DESIGN_LENGTH
SA_NU_TILDE_FLOOR = 1.0e-12
SA_NU_TILDE_PENALTY_ALPHA = 2.0e3
SA_NU_TILDE_PENALTY_N = 3.0

SA_USE_PENALIZED_WALL_DISTANCE = False
SA_WALL_SIGMA = SA_DISTANCE_RELAXATION
SA_WALL_G0 = 20.0
SA_WALL_PENALTY_ALPHA = 2.0e3
SA_WALL_PENALTY_N = 3.0
SA_WALL_G_FLOOR = 1.0e-8

# -------------------------------------------------------------------
# Topology optimization settings
# -------------------------------------------------------------------
VOL_FRAC = 0.43
INITIAL_DENSITY_VALUE = 0.43
OBJECTIVE_CONVERGENCE_TOL = 5.0e-5
OBJECTIVE_STREAK_TO_STOP = 5

# Start gently enough to avoid early state-solve failures, then raise the
# Brinkman interpolation and projection sharply so outlet constraints are less
# likely to be satisfied through broad gray leakage.
Q_PENAL_SCHEDULE = [0.1, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6]
MOVE_LIMIT_SCHEDULE = [0.05, 0.05, 0.04, 0.03, 0.025, 0.02, 0.015, 0.01]
BETA_PROJ_SCHEDULE = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 14.0]
MAX_INNER_ITERATIONS_SCHEDULE = [60, 60, 60, 60, 60, 60, 60, 60]

MASS_FLOW_CONSTRAINT_MARKERS = [3, 4, 5]  # top, right, bottom
MASS_FLOW_TARGET_FRACTIONS = [0.3, 0.4, 0.3]
MASS_FLOW_CONSTRAINT_MODE = 'equality'
MASS_FLOW_CONSTRAINT_TOLERANCE = 1.0e-4

# -------------------------------------------------------------------
# Monolithic full-state solver settings
# -------------------------------------------------------------------
SNES_LINEAR_SOLVER = 'mumps'
FULL_STATE_LINEAR_SOLVER = 'mumps'
FULL_STATE_SNES_METHOD = 'newtontr'
FULL_STATE_SNES_LINE_SEARCH = 'bt'
FULL_STATE_SNES_RTOL = 1.0e-4
FULL_STATE_SNES_ATOL = 3.0e-5
FULL_STATE_SNES_MAX_ITERS = 150
FULL_STATE_RESTART_WITH_STOKES = False
FULL_STATE_SNES_FALLBACK_METHOD = 'newtonls'
FULL_STATE_SNES_FALLBACK_LINE_SEARCH = 'bt'
FULL_STATE_SNES_FALLBACK_MAX_ITERS = 250

# -------------------------------------------------------------------
# Projection, filter, and output
# -------------------------------------------------------------------
BETA_PROJ_VALUE = BETA_PROJ_SCHEDULE[0]
ETA_I = 0.50
QUADRATURE_DEGREE = 6

# Match the paper's filter-radius parameter directly.
FILTER_BASE_LENGTH = 0.028
FILTER_RADIUS_IN_CELLS = 1.0

OUTLET_BC_TYPE = 'pressure'
OUTLET_PRESSURE_VALUE = 0.0

ENABLE_PRESSURE_PIN = False
PRESSURE_PIN_POINT = (DOMAIN_X_MIN, DOMAIN_Y_MIN)
RESULTS_ROOT_NAME_FULL = 'Results_Full/Results_ManifoldDilgen_TurbulentTO_Full'

MARK = {'generic': 0, 'walls': 1, 'inlet': 2, 'outlet': (3, 4, 5)}


def between(value, limits, eps=BOUNDARY_TOL):
    return (limits[0] - eps <= value) and (value <= limits[1] + eps)


def in_interval(value, min_value, max_value):
    return between(value, (min_value, max_value), BOUNDARY_TOL)


def in_rectangle(x_value, y_value, x_min, x_max, y_min, y_max):
    return in_interval(x_value, x_min, x_max) and in_interval(y_value, y_min, y_max)


def in_inlet_segment(y_value):
    return between(y_value, (INLET_Y_MIN, INLET_Y_MAX), BOUNDARY_TOL)


def in_top_outlet_segment(x_value):
    return in_interval(x_value, TOP_OUTLET_X_MIN, TOP_OUTLET_X_MAX)


def in_right_outlet_segment(y_value):
    return in_interval(y_value, RIGHT_OUTLET_Y_MIN, RIGHT_OUTLET_Y_MAX)


def in_bottom_outlet_segment(x_value):
    return in_interval(x_value, BOTTOM_OUTLET_X_MIN, BOTTOM_OUTLET_X_MAX)


class VerticalSegmentBoundary(SubDomain):
    def __init__(self, x_location, y_min, y_max):
        super().__init__()
        self._x_location = x_location
        self._y_min = y_min
        self._y_max = y_max

    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], self._x_location, BOUNDARY_TOL) and between(
            x[1], (self._y_min, self._y_max), BOUNDARY_TOL
        )


class HorizontalSegmentBoundary(SubDomain):
    def __init__(self, y_location, x_min, x_max):
        super().__init__()
        self._y_location = y_location
        self._x_min = x_min
        self._x_max = x_max

    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], self._y_location, BOUNDARY_TOL) and between(
            x[0], (self._x_min, self._x_max), BOUNDARY_TOL
        )


class WallsBoundary(SubDomain):
    def inside(self, x, on_boundary):
        if not on_boundary:
            return False

        is_inlet = near(x[0], INLET_BLOCK_X_MIN, BOUNDARY_TOL) and in_inlet_segment(x[1])
        is_top_outlet = near(x[1], TOP_BLOCK_Y_MAX, BOUNDARY_TOL) and in_top_outlet_segment(x[0])
        is_right_outlet = near(x[0], RIGHT_BLOCK_X_MAX, BOUNDARY_TOL) and in_right_outlet_segment(x[1])
        is_bottom_outlet = near(x[1], BOTTOM_BLOCK_Y_MIN, BOUNDARY_TOL) and in_bottom_outlet_segment(x[0])
        return not (is_inlet or is_top_outlet or is_right_outlet or is_bottom_outlet)


def mark_boundaries(mesh):
    boundaries = MeshFunction('size_t', mesh, mesh.topology().dim() - 1)
    boundaries.set_all(MARK['generic'])

    WallsBoundary().mark(boundaries, MARK['walls'])
    VerticalSegmentBoundary(INLET_BLOCK_X_MIN, INLET_Y_MIN, INLET_Y_MAX).mark(boundaries, MARK['inlet'])
    HorizontalSegmentBoundary(TOP_BLOCK_Y_MAX, TOP_OUTLET_X_MIN, TOP_OUTLET_X_MAX).mark(
        boundaries, MARK['outlet'][0]
    )
    VerticalSegmentBoundary(RIGHT_BLOCK_X_MAX, RIGHT_OUTLET_Y_MIN, RIGHT_OUTLET_Y_MAX).mark(
        boundaries, MARK['outlet'][1]
    )
    HorizontalSegmentBoundary(BOTTOM_BLOCK_Y_MIN, BOTTOM_OUTLET_X_MIN, BOTTOM_OUTLET_X_MAX).mark(
        boundaries, MARK['outlet'][2]
    )

    return boundaries


def build_velocity_profile_sets():
    inlet_center = 0.5 * (INLET_Y_MIN + INLET_Y_MAX)
    inlet_width = INLET_Y_MAX - INLET_Y_MIN
    u_inlet = Expression(
        ('u_max * (1 - pow(2.0 * (x[1] - y_c) / width, 2))', '0.0'),
        degree=2,
        u_max=U_MAX_INLET,
        y_c=inlet_center,
        width=inlet_width,
    )
    return [u_inlet], []


def build_density_bounds(mesh, density_space):
    lower = Function(density_space)
    upper = Function(density_space)

    lower_values = np.zeros(density_space.dim())
    upper_values = np.ones(density_space.dim())
    dofmap = density_space.dofmap()

    for cell in cells(mesh):
        dof = dofmap.cell_dofs(cell.index())[0]
        midpoint = cell.midpoint()
        x_coord = midpoint.x()
        y_coord = midpoint.y()

        if in_rectangle(
            x_coord, y_coord,
            INLET_BLOCK_X_MIN, INLET_BLOCK_X_MAX,
            INLET_BLOCK_Y_MIN, INLET_BLOCK_Y_MAX,
        ):
            if in_inlet_segment(y_coord):
                lower_values[dof] = 1.0
                upper_values[dof] = 1.0
            else:
                lower_values[dof] = 0.0
                upper_values[dof] = 0.0
        elif in_rectangle(
            x_coord, y_coord,
            TOP_BLOCK_X_MIN, TOP_BLOCK_X_MAX,
            TOP_BLOCK_Y_MIN, TOP_BLOCK_Y_MAX,
        ):
            if in_top_outlet_segment(x_coord):
                lower_values[dof] = 1.0
                upper_values[dof] = 1.0
            else:
                lower_values[dof] = 0.0
                upper_values[dof] = 0.0
        elif in_rectangle(
            x_coord, y_coord,
            RIGHT_BLOCK_X_MIN, RIGHT_BLOCK_X_MAX,
            RIGHT_BLOCK_Y_MIN, RIGHT_BLOCK_Y_MAX,
        ):
            if in_right_outlet_segment(y_coord):
                lower_values[dof] = 1.0
                upper_values[dof] = 1.0
            else:
                lower_values[dof] = 0.0
                upper_values[dof] = 0.0
        elif in_rectangle(
            x_coord, y_coord,
            BOTTOM_BLOCK_X_MIN, BOTTOM_BLOCK_X_MAX,
            BOTTOM_BLOCK_Y_MIN, BOTTOM_BLOCK_Y_MAX,
        ):
            if in_bottom_outlet_segment(x_coord):
                lower_values[dof] = 1.0
                upper_values[dof] = 1.0
            else:
                lower_values[dof] = 0.0
                upper_values[dof] = 0.0

    lower.vector().set_local(lower_values)
    lower.vector().apply('insert')
    upper.vector().set_local(upper_values)
    upper.vector().apply('insert')
    return lower, upper


def build_volume_region(mesh, density_space):
    volume_region = Function(density_space)
    region_values = np.zeros(density_space.dim())
    dofmap = density_space.dofmap()

    for cell in cells(mesh):
        dof = dofmap.cell_dofs(cell.index())[0]
        x_coord = cell.midpoint().x()
        y_coord = cell.midpoint().y()
        if (
            DESIGN_X_MIN - DOLFIN_EPS <= x_coord <= DESIGN_X_MAX + DOLFIN_EPS
            and DESIGN_Y_MIN - DOLFIN_EPS <= y_coord <= DESIGN_Y_MAX + DOLFIN_EPS
        ):
            region_values[dof] = 1.0

    volume_region.vector().set_local(region_values)
    volume_region.vector().apply('insert')
    return volume_region
