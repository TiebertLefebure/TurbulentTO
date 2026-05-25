import os

from dolfin import DOLFIN_EPS, Expression, MeshFunction, MPI, SubDomain, near

from Utilities_SharedTO import build_cell_tag_restriction_functions, load_mesh_from_xdmf


# -------------------------------------------------------------------------
# Configuration: Yoon 2016 Double Pipe - Laminar reference (Re = 1)
#
# Uses the same geometry and parabolic inlet-profile shape as the turbulent
# Yoon double-pipe case. The laminar reference uses a normalized inlet
# magnitude and viscosity so rho * U_max * H2 / mu = 1 without inflating the
# dimensional residuals and objective.
# -------------------------------------------------------------------------

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

mesh_files = {
    "MESH_DIRECTORY": os.path.join(REPO_ROOT, "Meshes/DoublePipeYoon/mesh_yoon_yplus1.xdmf"),
    "CELL_DIRECTORY": os.path.join(REPO_ROOT, "Meshes/DoublePipeYoon/cell_yoon_yplus1.xdmf"),
    "FACET_DIRECTORY": os.path.join(REPO_ROOT, "Meshes/DoublePipeYoon/facet_yoon_yplus1.xdmf"),
}

DESIGN_DOMAIN_TAG = 1


def create_design_mesh():
    return load_mesh_from_xdmf(mesh_files["MESH_DIRECTORY"], MPI.comm_world)


build_density_bounds, build_volume_region, build_objective_region = build_cell_tag_restriction_functions(
    mesh_files["CELL_DIRECTORY"],
    design_tags=(DESIGN_DOMAIN_TAG,),
)


L1 = 3.0
L2 = 2.0
H1 = 5.0
H2 = 1.0
H3 = 1.0
H_MAX = 1.0 / 50.0
DESIGN_X_MIN = 0.0
DESIGN_X_MAX = L1
DOMAIN_X_MIN = -L2
DOMAIN_X_MAX = L1 + L2
DOMAIN_Y_MIN = 0.0
DOMAIN_Y_MAX = H1

BOTTOM_INLET_Y_MIN = 1.0
BOTTOM_INLET_Y_MAX = BOTTOM_INLET_Y_MIN + H2
TOP_INLET_Y_MIN = 3.0
TOP_INLET_Y_MAX = TOP_INLET_Y_MIN + H2
BOTTOM_OUTLET_Y_MIN = 1.0
BOTTOM_OUTLET_Y_MAX = BOTTOM_OUTLET_Y_MIN + H3
TOP_OUTLET_Y_MIN = 3.0
TOP_OUTLET_Y_MAX = TOP_OUTLET_Y_MIN + H3

INLET_SEGMENTS = [
    (TOP_INLET_Y_MIN, TOP_INLET_Y_MAX),
    (BOTTOM_INLET_Y_MIN, BOTTOM_INLET_Y_MAX),
]
OUTLET_SEGMENTS = [
    (TOP_OUTLET_Y_MIN, TOP_OUTLET_Y_MAX),
    (BOTTOM_OUTLET_Y_MIN, BOTTOM_OUTLET_Y_MAX),
]
TOL = DOLFIN_EPS

REYNOLDS_NUMBER = 1.0
RHO_FLUID_VALUE = 1.0
U_MAX_INLETS = [1.0, 1.0]
MU_FLUID_VALUE = U_MAX_INLETS[0] * H2 * RHO_FLUID_VALUE / REYNOLDS_NUMBER
ALPHA_FLUID = 0.0
ALPHA_SOLID = 1.0e5

# Re = rho * U_MAX_INLET * H2 / mu = 1 for both 1 m high inlet ports.

VOL_FRAC = 0.40
INITIAL_DENSITY_VALUE = VOL_FRAC
INITIAL_DENSITY_MATCH_FILTERED_VOLUME = True
OBJECTIVE_TYPE = "dissipation"
OBJECTIVE_CONVERGENCE_TOL = 1e-7
OBJECTIVE_STREAK_TO_STOP = 10

Q_PENAL_SCHEDULE = [0.01, 0.02, 0.04, 0.08, 0.12, 0.20, 0.35, 0.50, 0.75, 1.00]
BETA_PROJ_SCHEDULE = [1.0, 2.0, 4.0, 8.0, 12.0, 16.0, 32.0, 64.0, 96.0, 128.0]
MOVE_LIMIT_SCHEDULE = [0.05, 0.04, 0.03, 0.02, 0.015, 0.01, 0.0075, 0.005, 0.003, 0.002]
MAX_INNER_ITERATIONS_SCHEDULE = [50, 60, 80, 100, 100, 120, 150, 180, 120, 120]

SNES_LINEAR_SOLVER = "mumps"
FILTER_BASE_LENGTH = H_MAX
FILTER_RADIUS_IN_CELLS = 4.0
FILTER_DENOMINATOR_FLOOR = 1.0e-12
QUADRATURE_DEGREE = 6
FORWARD_SNES_RTOL = 1.0e-6
FORWARD_SNES_ATOL = 1.0e-8
FORWARD_SNES_MAX_ITERS = 220
ADJOINT_SNES_RTOL = 1.0e-6
ADJOINT_SNES_ATOL = 1.0e-9
SNES_MAX_ITERS = 200

BETA_PROJ_VALUE = BETA_PROJ_SCHEDULE[0]
ETA_I = 0.50

OUTLET_BC_TYPE = "pressure"
OUTLET_PRESSURE_VALUE = 0.0
PRESSURE_OUTLET_COMPONENT_BCS = [
    {"marker": "outlet", "component": 1, "value": 0.0},
]
ENABLE_PRESSURE_PIN = False
SAVE_DF0DX_VECTOR = True
LOG_DF0DX_STATS = True
SAVE_DF0DX_CENTERED_FIELD = True
RESULTS_ROOT_NAME = "Results_Laminar/Results_DoublePipeYoon_LaminarTO"

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
        left_step_wall = near(x[0], self._design_x_min, self._tol) and not self._inside_any_segment(
            x[1], self._inlet_segments
        )
        right_step_wall = near(x[0], self._design_x_max, self._tol) and not self._inside_any_segment(
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
            left_step_wall
            or right_step_wall
            or top_wall
            or bottom_wall
            or inlet_strip_caps
            or outlet_strip_caps
        )


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


def _parabolic_x_profile(y_min, y_max, u_max):
    height = y_max - y_min
    return Expression(
        ("u_max*4.0*(x[1] - y_min)*(y_max - x[1])/(height*height)", "0.0"),
        degree=2,
        u_max=u_max,
        y_min=y_min,
        y_max=y_max,
        height=height,
    )


def build_velocity_profile_sets():
    inlet_profiles = [
        _parabolic_x_profile(segment[0], segment[1], u_max)
        for segment, u_max in zip(INLET_SEGMENTS, U_MAX_INLETS)
    ]
    return inlet_profiles, []
