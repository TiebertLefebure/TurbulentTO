import os

from dolfin import DOLFIN_EPS, Expression, MeshFunction, MPI, SubDomain, near

from Utilities_SharedTO import build_cell_tag_restriction_functions, load_mesh_from_xdmf


# ===================================================================
# Configuration: Alexandersen 2026 U-Bend - Turbulent SemiFrozen (SA)
#
# Geometry, mesh size, and boundary conditions follow Figure 11 and
# Table 4 of Bayat, Li, and Alexandersen (2026), while the turbulence
# closure remains the in-house SemiFrozen Spalart-Allmaras implementation.
# ===================================================================

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

mesh_files = {
    "MESH_DIRECTORY": os.path.join(REPO_ROOT, "Meshes/UBendAlexandersen/mesh_guided.xdmf"),
    "CELL_DIRECTORY": os.path.join(REPO_ROOT, "Meshes/UBendAlexandersen/cell_guided.xdmf"),
}

DESIGN_DOMAIN_TAG = 1
NON_DESIGN_FLUID_TAG = 2


def create_design_mesh():
    return load_mesh_from_xdmf(mesh_files["MESH_DIRECTORY"], MPI.comm_world)


build_density_bounds, build_volume_region, build_objective_region = build_cell_tag_restriction_functions(
    mesh_files["CELL_DIRECTORY"],
    design_tags=(DESIGN_DOMAIN_TAG,),
    non_design_fluid_tags=(NON_DESIGN_FLUID_TAG,),
)


L = 1.0
H_MAX = 0.007
LEAD_LENGTH = 0.2 * L
PORT_HEIGHT = 0.2 * L
TOP_PORT_Y_MIN = 0.55 * L
TOP_PORT_Y_MAX = TOP_PORT_Y_MIN + PORT_HEIGHT
BOTTOM_PORT_Y_MIN = 0.25 * L
BOTTOM_PORT_Y_MAX = BOTTOM_PORT_Y_MIN + PORT_HEIGHT

BAR_THICKNESS = 0.10 * L
BAR_RADIUS = 0.5 * BAR_THICKNESS
BAR_TOTAL_LENGTH = 0.70 * L
BAR_X_START = -LEAD_LENGTH
BAR_TIP_X = BAR_X_START + BAR_TOTAL_LENGTH
BAR_RECT_X_MAX = BAR_TIP_X - BAR_RADIUS
BAR_Y_MIN = 0.5 * (L - BAR_THICKNESS)
BAR_Y_MAX = BAR_Y_MIN + BAR_THICKNESS

DESIGN_X_MIN = 0.0
DESIGN_Y_MIN = 0.0
DESIGN_X_MAX = L
DESIGN_Y_MAX = L
DOMAIN_X_MIN = DESIGN_X_MIN - LEAD_LENGTH
DOMAIN_Y_MIN = DESIGN_Y_MIN
DOMAIN_X_MAX = DESIGN_X_MAX
DOMAIN_Y_MAX = DESIGN_Y_MAX
TOL = DOLFIN_EPS

REYNOLDS_NUMBER = 5000.0
RHO_FLUID_VALUE = 1.0
U_MAX_INLET = 1.0
MU_FLUID_VALUE = U_MAX_INLET * PORT_HEIGHT * RHO_FLUID_VALUE / REYNOLDS_NUMBER

SA_TURBULENCE_INTENSITY = 0.075
SA_TURBULENCE_LENGTH_SCALE_RATIO = 0.05
SA_REFERENCE_LENGTH = PORT_HEIGHT
SA_REYNOLDS_NUMBER = REYNOLDS_NUMBER
SA_SMOOTH_ABS_EPS = 1.0e-12
SA_INIT_WALL_DIST_SCALE = 0.05 * L
SA_NU_TILDE_FLOOR = 1.0e-12
SA_NU_TILDE_PENALTY_ALPHA = 1.0e3
SA_NU_TILDE_PENALTY_N = 3.0
SA_NU_TILDE_PENALTY_INTERPOLATION = "power"

SA_WALL_DISTANCE_MODE = "reciprocal_penalized"
SA_WALL_SIGMA = 0.01
SA_WALL_G0 = 20.0
SA_WALL_PENALTY_ALPHA = 1.0e3
SA_WALL_PENALTY_N = 3.0
SA_WALL_PENALTY_INTERPOLATION = "power"
SA_WALL_G_FLOOR = 1.0e-8

VOL_FRAC = 0.27
OBJECTIVE_CONVERGENCE_TOL = 1.0e-5
OBJECTIVE_STREAK_TO_STOP = 5

Q_PENAL_SCHEDULE = [0.05, 0.10, 0.20, 0.40, 0.80, 1.50, 2.50, 3.00, 3.00]
MOVE_LIMIT_SCHEDULE = [0.08, 0.07, 0.06, 0.045, 0.03, 0.02, 0.01, 0.005, 0.002]
BETA_PROJ_SCHEDULE = [0.10, 0.25, 0.50, 1.00, 2.00, 4.00, 8.00, 16.00, 24.00]
MAX_INNER_ITERATIONS_SCHEDULE = [60, 70, 80, 90, 90, 90, 100, 100, 100]

LINEAR_SOLVER = "mumps"
STATE_SOLVE_METHOD = "newtontr"
STATE_RTOL = 1.0e-6
STATE_ATOL = 1.0e-8
STATE_MAX_ITERS = 160
STATE_INITIAL_SA_SWEEPS = 6
STATE_INITIAL_SA_RELAXATION = 0.35
STATE_TURBULENCE_COUPLING_SCHEDULE = [
    {"convection_weight": 0.00, "weight": 0.00, "max_iters": 200, "atol": 8.0e-4, "accept_norm": 1.0e-1},
    {"convection_weight": 0.20, "weight": 0.00, "max_iters": 220, "atol": 5.0e-4, "accept_norm": 8.0e-2},
    {"convection_weight": 0.45, "weight": 0.20, "max_iters": 240, "atol": 3.0e-4, "accept_norm": 4.0e-2},
    {"convection_weight": 0.70, "weight": 0.50, "max_iters": 260, "atol": 2.0e-4, "accept_norm": 1.5e-2},
    {"convection_weight": 0.90, "weight": 0.75, "max_iters": 280, "atol": 1.5e-4, "accept_norm": 5.0e-3},
    {"convection_weight": 1.00, "weight": 1.00, "max_iters": 320},
]
STATE_RECOVERY_ATTEMPTS = [
    {
        "label": "current-iterate line-search retry",
        "method": "newtonls",
        "line_search": "bt",
        "max_iters": 260,
        "restart_with_stokes": False,
    },
    {
        "label": "Stokes rebuild line-search retry",
        "method": "newtonls",
        "line_search": "bt",
        "max_iters": 320,
        "restart_with_stokes": True,
    },
]

BETA_PROJ_VALUE = BETA_PROJ_SCHEDULE[0]
ETA_I = 0.50
QUADRATURE_DEGREE = 6
FILTER_BASE_LENGTH = H_MAX
FILTER_RADIUS_IN_CELLS = 3.0

OUTLET_BC_TYPE = "pressure"
OUTLET_PRESSURE_VALUE = 0.0
PRESSURE_OUTLET_COMPONENT_BCS = [
    {"marker": "outlet", "component": 1, "value": 0.0},
]

ENABLE_PRESSURE_PIN = False
RESULTS_ROOT_NAME = "Results_SemiFrozen/Results_UBendAlexandersen_TurbulentTO_SemiFrozen"

MARK = {"generic": 0, "walls": 1, "inlet": 2, "outlet": 3}


def between(value, limits, eps=DOLFIN_EPS):
    return (limits[0] - eps <= value) and (value <= limits[1] + eps)


def is_inlet_point(x):
    return near(x[0], DOMAIN_X_MIN, TOL) and between(x[1], (TOP_PORT_Y_MIN, TOP_PORT_Y_MAX), TOL)


def is_outlet_point(x):
    return near(x[0], DOMAIN_X_MIN, TOL) and between(x[1], (BOTTOM_PORT_Y_MIN, BOTTOM_PORT_Y_MAX), TOL)


class InletBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and is_inlet_point(x)


class OutletBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and is_outlet_point(x)


class WallsBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and not is_inlet_point(x) and not is_outlet_point(x)


def mark_boundaries(mesh):
    boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    boundaries.set_all(MARK["generic"])
    WallsBoundary().mark(boundaries, MARK["walls"])
    InletBoundary().mark(boundaries, MARK["inlet"])
    OutletBoundary().mark(boundaries, MARK["outlet"])
    return boundaries


def build_velocity_profile_sets():
    u_inlet = Expression(("u_max", "0.0"), degree=0, u_max=U_MAX_INLET)
    return [u_inlet], []
