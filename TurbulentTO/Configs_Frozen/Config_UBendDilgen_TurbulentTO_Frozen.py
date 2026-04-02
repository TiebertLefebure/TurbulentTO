import os
import numpy as np
from dolfin import DOLFIN_EPS, Expression, Function, MPI, MeshFunction, SubDomain, cells, near
from Utilities_LaminarTO import load_mesh_from_xdmf


# ===================================================================
# Configuration: Dilgen 2018 U-bend - Turbulent (SA, Re = 5000)
#
# Paper data used here:
#   Ub = 2.0 m/s, H = 0.1 m, nu = 4.0e-5 m^2/s, Re = Ub * H / nu = 5000
#
# Implementation choice:
#   Keep the existing Borrvall-style turbulent workflow (IPCS + frozen-SA adjoint)
#   and match the Dilgen case at the geometry and BC-type level:
#   one velocity inlet, one pressure outlet, passive inlet/outlet pads,
#   and the internal U-bend baffle.
# ===================================================================

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)

# Mesh files
# Generate via: cd Meshes/UBendDilgen && python3 ubend_dilgen_gmsh.py && python3 gmsh_to_xdmf.py
mesh_files = {
    'MESH_DIRECTORY': os.path.join(REPO_ROOT, 'Meshes/UBendDilgen/mesh.xdmf'),
}


def create_design_mesh():
    return load_mesh_from_xdmf(mesh_files['MESH_DIRECTORY'], MPI.comm_world)


# -------------------------------------------------------------------
# Geometry (scaled exactly as in Fig. 7 of Dilgen et al. 2018)
# -------------------------------------------------------------------
H = 0.1
DESIGN_X_MIN = 0.0
DESIGN_Y_MIN = 0.0
DESIGN_X_MAX = 10.0 * H
DESIGN_Y_MAX = 10.0 * H
DESIGN_LENGTH = DESIGN_X_MAX - DESIGN_X_MIN
L = DESIGN_LENGTH
N = 120

# Fixed left block: 2H wide with 2.5H-solid / 2H-fluid / H-solid / 2H-fluid / 2.5H-solid
LEFT_BLOCK_X_MIN = -2.0 * H
LEFT_BLOCK_X_MAX = DESIGN_X_MIN

CORNER_SOLID_HEIGHT = 2.5 * H
PORT_HEIGHT = 2.0 * H
BAFFLE_THICKNESS = 1.0 * H

OUTLET_Y_MIN = CORNER_SOLID_HEIGHT
OUTLET_Y_MAX = OUTLET_Y_MIN + PORT_HEIGHT
BAFFLE_Y_MIN = OUTLET_Y_MAX
BAFFLE_Y_MAX = BAFFLE_Y_MIN + BAFFLE_THICKNESS
INLET_Y_MIN = BAFFLE_Y_MAX
INLET_Y_MAX = INLET_Y_MIN + PORT_HEIGHT

BAFFLE_LENGTH = 5.0 * H
BAFFLE_X_MIN = DESIGN_X_MIN
BAFFLE_X_MAX = BAFFLE_X_MIN + BAFFLE_LENGTH
BAFFLE_TIP_RADIUS = 0.5 * BAFFLE_THICKNESS
BAFFLE_TIP_CENTER_X = BAFFLE_X_MAX - BAFFLE_TIP_RADIUS
BAFFLE_TIP_CENTER_Y = 0.5 * (BAFFLE_Y_MIN + BAFFLE_Y_MAX)
BAFFLE_STRAIGHT_X_MAX = BAFFLE_TIP_CENTER_X
PASSIVE_BAFFLE_SAMPLES = 5

DOMAIN_X_MIN = LEFT_BLOCK_X_MIN
DOMAIN_Y_MIN = DESIGN_Y_MIN
DOMAIN_X_MAX = DESIGN_X_MAX
DOMAIN_Y_MAX = DESIGN_Y_MAX

BOUNDARY_TOL = 1.0e-6
TOL = BOUNDARY_TOL

# -------------------------------------------------------------------
# Flow settings
# -------------------------------------------------------------------
# Keep the U-bend inlet aligned with the other benchmark cases for now:
# use the standard parabolic velocity profile, but choose its peak so the
# cross-section-averaged inlet velocity still matches the paper value Ub.
# For a 2D parabola, U_bulk = (2/3) * U_max.
RHO_FLUID_VALUE = 1.0
MU_FLUID_VALUE = 4.0e-3
U_BULK_INLET = 2.0
U_MAX_INLET = 1.5 * U_BULK_INLET

# ---------------------------------------------------------------------------------------------------
# Reynolds number: Re_H = U_BULK_INLET * 0.5*PORT_HEIGHT * RHO_FLUID_VALUE / MU_FLUID_VALUE = 50
# ---------------------------------------------------------------------------------------------------

# Treat the passive U-bend baffle as a strong imposed solid region without
# making the segregated solve excessively stiff.
ALPHA_SOLID = 1.0e3


# -------------------------------------------------------------------
# Spalart-Allmaras turbulence model settings
# -------------------------------------------------------------------
# Use the shared ratio-based inlet treatment from the other turbulent cases.
SA_MUT_RATIO = 5.0
SA_DISTANCE_RELAXATION = 0.01
SA_SMOOTH_ABS_EPS = 1.0e-12
SA_INIT_WALL_DIST_SCALE = 0.05 * L
SA_NU_TILDE_FLOOR = 1.0e-12
SA_NU_TILDE_PENALTY_ALPHA = 1.0e3
SA_NU_TILDE_PENALTY_N = 3.0

# Wall-distance model selector:
#   "direct_y"    -> solve a direct distance/eikonal wall-distance PDE
#   "penalized_g" -> solve the penalized reciprocal-distance model
#   "geometric"   -> use the plain geometric distance field only
SA_WALL_MODEL = "penalized_g"
SA_WALL_DENSITY_SOURCE = "design"
SA_WALL_SIGMA = 0.10
SA_WALL_G0 = 20.0
SA_WALL_PENALTY_ALPHA = 1.0e2
SA_WALL_PENALTY_N = 3.0
SA_WALL_SOLID_THRESHOLD = 0.10
SA_WALL_G_FLOOR = 1.0e-8
SA_WALL_NEWTON_MAX_ITERS = 300
SA_WALL_NEWTON_RELAXATION = 0.1
SA_WALL_PENALTY_HOMOTOPY = [0.0, 0.05, 0.15, 0.35, 0.65, 1.0]
SA_WALL_INITIAL_SOLID_GUESS = 1.0
SA_WALL_NEWTON_RELAXATION_CANDIDATES = [0.1, 0.05, 0.02, 0.01]
SA_WALL_PREFER_PSEUDO_TIME = True

# -------------------------------------------------------------------
# Topology optimization settings
# -------------------------------------------------------------------
VOL_FRAC = 0.30
MAX_INNER_ITERATIONS_SCHEDULE = [60, 60, 60, 60, 60, 60, 60, 60]
OBJECTIVE_CONVERGENCE_TOL = 5.0e-5
OBJECTIVE_STREAK_TO_STOP = 5

# Keep the early stages gray-friendly, then raise q moderately so the
# optimizer does not keep broad semi-fluid regions near the pressure outlet.
Q_PENAL_SCHEDULE = [0.1, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.5]
MOVE_LIMIT_SCHEDULE = [0.10, 0.08, 0.06, 0.04, 0.03, 0.02, 0.015, 0.01]
BETA_PROJ_SCHEDULE = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 14.0]

# -------------------------------------------------------------------
# Solver settings
# -------------------------------------------------------------------
SNES_LINEAR_SOLVER = "mumps"
FROZEN_PICARD_STEPS = 3
NUT_RELAXATION_FACTOR = 0.35

FORWARD_IPCS_DT = 2.0e-4
FORWARD_IPCS_MAX_ITERS = 400
FORWARD_IPCS_RTOL = 2.0e-4
FORWARD_IPCS_PRESSURE_RTOL = 2.0e-3
FORWARD_IPCS_U_RELAXATION = 0.20
FORWARD_IPCS_P_RELAXATION = 0.10
FORWARD_IPCS_VEL_SOLVER = "mumps"
FORWARD_IPCS_P_SOLVER = "mumps"
FORWARD_IPCS_LOG_EVERY = 50
FORWARD_IPCS_MAX_RESTARTS = 3
FORWARD_IPCS_DT_REDUCTION_FACTOR = 0.5
FORWARD_IPCS_RELAXATION_REDUCTION_FACTOR = 0.7
FORWARD_IPCS_ERROR_ON_NONCONVERGENCE = True
FORWARD_IPCS_RESTART_WITH_STOKES = False

# -------------------------------------------------------------------
# Projection, filter, and output
# -------------------------------------------------------------------
BETA_PROJ_VALUE = BETA_PROJ_SCHEDULE[0]
ETA_I = 0.50
QUADRATURE_DEGREE = 6

# Use a real multi-cell filter radius to suppress speckle and disconnected
# artefacts during the early UBend iterations.
FILTER_RADIUS_IN_CELLS = 2.0

OUTLET_BC_TYPE = "pressure"
OUTLET_PRESSURE_VALUE = 0.0
SAVE_IPCS_RESIDUAL_PLOTS = True

ENABLE_PRESSURE_PIN = False
PRESSURE_PIN_POINT = (LEFT_BLOCK_X_MIN, OUTLET_Y_MIN)
RESULTS_ROOT_NAME = "Results_Frozen/Results_UBendDilgen_TurbulentTO_Frozen"

MARK = {"generic": 0, "walls": 1, "inlet": 2, "outlet": 3}


def between(value, limits, eps=BOUNDARY_TOL):
    return (limits[0] - eps <= value) and (value <= limits[1] + eps)


def in_interval(value, min_value, max_value):
    return between(value, (min_value, max_value), BOUNDARY_TOL)


def in_rectangle(x_value, y_value, x_min, x_max, y_min, y_max):
    return in_interval(x_value, x_min, x_max) and in_interval(y_value, y_min, y_max)


def in_rounded_baffle(x_value, y_value):
    # Figure 7 labels the obstacle thickness H and total in-design length 5H.
    # A semicircular nose with radius H/2 is the natural geometric interpretation.
    in_straight_section = in_rectangle(
        x_value, y_value,
        BAFFLE_X_MIN, BAFFLE_STRAIGHT_X_MAX,
        BAFFLE_Y_MIN, BAFFLE_Y_MAX,
    )
    dx_tip = x_value - BAFFLE_TIP_CENTER_X
    dy_tip = y_value - BAFFLE_TIP_CENTER_Y
    in_tip = dx_tip * dx_tip + dy_tip * dy_tip <= (BAFFLE_TIP_RADIUS + BOUNDARY_TOL) ** 2
    return in_straight_section or in_tip


def cell_near_rounded_baffle(cell):
    coords = np.asarray(cell.get_vertex_coordinates(), dtype=float).reshape((-1, 2))
    x_min = np.min(coords[:, 0])
    x_max = np.max(coords[:, 0])
    y_min = np.min(coords[:, 1])
    y_max = np.max(coords[:, 1])
    padding = cell.h()
    return not (
        x_max < BAFFLE_X_MIN - padding
        or x_min > BAFFLE_X_MAX + padding
        or y_max < BAFFLE_Y_MIN - padding
        or y_min > BAFFLE_Y_MAX + padding
    )


def cell_near_baffle_tip(cell):
    coords = np.asarray(cell.get_vertex_coordinates(), dtype=float).reshape((-1, 2))
    x_min = np.min(coords[:, 0])
    x_max = np.max(coords[:, 0])
    y_min = np.min(coords[:, 1])
    y_max = np.max(coords[:, 1])
    padding = cell.h()
    return not (
        x_max < BAFFLE_STRAIGHT_X_MAX - padding
        or x_min > BAFFLE_X_MAX + padding
        or y_max < BAFFLE_Y_MIN - padding
        or y_min > BAFFLE_Y_MAX + padding
    )


def estimate_baffle_fluid_fraction(cell, samples_per_edge=PASSIVE_BAFFLE_SAMPLES):
    coords = np.asarray(cell.get_vertex_coordinates(), dtype=float).reshape((-1, 2))
    samples_per_edge = max(1, int(samples_per_edge))
    fluid_samples = 0
    total_samples = 0

    for i_idx in range(samples_per_edge + 1):
        for j_idx in range(samples_per_edge + 1 - i_idx):
            xi_1 = float(i_idx) / float(samples_per_edge)
            xi_2 = float(j_idx) / float(samples_per_edge)
            xi_0 = 1.0 - xi_1 - xi_2
            point = xi_0 * coords[0] + xi_1 * coords[1] + xi_2 * coords[2]
            if not in_rounded_baffle(point[0], point[1]):
                fluid_samples += 1
            total_samples += 1

    return float(fluid_samples) / float(max(total_samples, 1))


class InletBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], LEFT_BLOCK_X_MIN, BOUNDARY_TOL) and between(
            x[1], (INLET_Y_MIN, INLET_Y_MAX), BOUNDARY_TOL
        )


class OutletBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], LEFT_BLOCK_X_MIN, BOUNDARY_TOL) and between(
            x[1], (OUTLET_Y_MIN, OUTLET_Y_MAX), BOUNDARY_TOL
        )


class WallsBoundary(SubDomain):
    def inside(self, x, on_boundary):
        if not on_boundary:
            return False

        is_inlet = near(x[0], LEFT_BLOCK_X_MIN, BOUNDARY_TOL) and between(
            x[1], (INLET_Y_MIN, INLET_Y_MAX), BOUNDARY_TOL
        )
        is_outlet = near(x[0], LEFT_BLOCK_X_MIN, BOUNDARY_TOL) and between(
            x[1], (OUTLET_Y_MIN, OUTLET_Y_MAX), BOUNDARY_TOL
        )
        return not (is_inlet or is_outlet)


def mark_boundaries(mesh):
    boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    boundaries.set_all(MARK["generic"])
    WallsBoundary().mark(boundaries, MARK["walls"])
    InletBoundary().mark(boundaries, MARK["inlet"])
    OutletBoundary().mark(boundaries, MARK["outlet"])
    return boundaries


def _distance_to_vertical_segment(x_value, y_value, x_segment, y_min, y_max):
    dy = 0.0
    if y_value < y_min:
        dy = y_min - y_value
    elif y_value > y_max:
        dy = y_value - y_max
    return float(np.hypot(x_value - x_segment, dy))


def _distance_to_horizontal_segment(x_value, y_value, x_min, x_max, y_segment):
    dx = 0.0
    if x_value < x_min:
        dx = x_min - x_value
    elif x_value > x_max:
        dx = x_value - x_max
    return float(np.hypot(dx, y_value - y_segment))


def _distance_to_baffle_boundary(x_value, y_value):
    distances = [
        _distance_to_horizontal_segment(
            x_value, y_value,
            BAFFLE_X_MIN, BAFFLE_STRAIGHT_X_MAX,
            BAFFLE_Y_MIN,
        ),
        _distance_to_horizontal_segment(
            x_value, y_value,
            BAFFLE_X_MIN, BAFFLE_STRAIGHT_X_MAX,
            BAFFLE_Y_MAX,
        ),
    ]

    if x_value >= BAFFLE_TIP_CENTER_X - BOUNDARY_TOL:
        distances.append(abs(
            np.hypot(x_value - BAFFLE_TIP_CENTER_X, y_value - BAFFLE_TIP_CENTER_Y)
            - BAFFLE_TIP_RADIUS
        ))

    return min(float(distance) for distance in distances)


def build_wall_distance_field(space, mesh, boundaries, wall_markers, custom_dx):
    wall_distance = Function(space)
    coords = np.asarray(space.tabulate_dof_coordinates(), dtype=float).reshape((-1, mesh.geometry().dim()))
    values = np.zeros(coords.shape[0], dtype=float)

    left_wall_segments = [
        (DOMAIN_Y_MIN, OUTLET_Y_MIN),
        (OUTLET_Y_MAX, INLET_Y_MIN),
        (INLET_Y_MAX, DOMAIN_Y_MAX),
    ]

    for idx, (x_value, y_value) in enumerate(coords[:, :2]):
        distances = [
            _distance_to_horizontal_segment(x_value, y_value, DOMAIN_X_MIN, DOMAIN_X_MAX, DOMAIN_Y_MIN),
            _distance_to_horizontal_segment(x_value, y_value, DOMAIN_X_MIN, DOMAIN_X_MAX, DOMAIN_Y_MAX),
            _distance_to_vertical_segment(x_value, y_value, DOMAIN_X_MAX, DOMAIN_Y_MIN, DOMAIN_Y_MAX),
            _distance_to_baffle_boundary(x_value, y_value),
        ]
        for y_min, y_max in left_wall_segments:
            distances.append(_distance_to_vertical_segment(x_value, y_value, LEFT_BLOCK_X_MIN, y_min, y_max))
        values[idx] = max(min(distances), 0.0)

    wall_distance.vector().set_local(values)
    wall_distance.vector().apply("insert")
    return wall_distance


def build_velocity_profile_sets():
    inlet_center = 0.5 * (INLET_Y_MIN + INLET_Y_MAX)
    inlet_width = INLET_Y_MAX - INLET_Y_MIN
    u_inlet = Expression(
        ("u_max * (1 - pow(2.0 * (x[1] - y_c) / width, 2))", "0.0"),
        degree=2,
        u_max=U_MAX_INLET,
        y_c=inlet_center,
        width=inlet_width,
    )

    # Dilgen 2018 prescribes a pressure outlet.  Keep the Borrvall-style
    # workflow by leaving the outlet velocity unconstrained.
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
            LEFT_BLOCK_X_MIN, LEFT_BLOCK_X_MAX,
            DOMAIN_Y_MIN, DOMAIN_Y_MAX,
        ):
            if (
                in_interval(y_coord, INLET_Y_MIN, INLET_Y_MAX)
                or in_interval(y_coord, OUTLET_Y_MIN, OUTLET_Y_MAX)
            ):
                lower_values[dof] = 1.0
                upper_values[dof] = 1.0
            else:
                lower_values[dof] = 0.0
                upper_values[dof] = 0.0
        elif in_rectangle(
            x_coord, y_coord,
            BAFFLE_X_MIN, BAFFLE_STRAIGHT_X_MAX,
            BAFFLE_Y_MIN, BAFFLE_Y_MAX,
        ):
            lower_values[dof] = 0.0
            upper_values[dof] = 0.0
        elif cell_near_baffle_tip(cell):
            fluid_fraction = estimate_baffle_fluid_fraction(cell)
            if fluid_fraction <= 0.5 + 1.0e-12:
                lower_values[dof] = fluid_fraction
                upper_values[dof] = fluid_fraction

    lower.vector().set_local(lower_values)
    lower.vector().apply("insert")
    upper.vector().set_local(upper_values)
    upper.vector().apply("insert")
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
    volume_region.vector().apply("insert")
    return volume_region
