import os

from dolfin import DOLFIN_EPS, Expression, MeshFunction, MPI, SubDomain, near

from Utilities_SharedTO import build_cell_tag_restriction_functions, load_mesh_from_xdmf


# ===================================================================
# Configuration: Alexandersen 2026 U-Bend - Turbulent Frozen (SA)
#
# Geometry, mesh size, and boundary conditions follow Figure 11 and
# Table 4 of Bayat, Li, and Alexandersen (2026), while the turbulence
# closure remains the in-house frozen Spalart-Allmaras implementation.
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

# Fixed non-design solid bar that forces the 180-degree turn.
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

# ==============================================================================================
# Reynolds number: Re = U_MAX_INLET * PORT_HEIGHT * RHO_FLUID_VALUE / MU_FLUID_VALUE = 5,000
# ==============================================================================================

SA_TURBULENCE_INTENSITY = 0.075
SA_TURBULENCE_LENGTH_SCALE_RATIO = 0.05
SA_REFERENCE_LENGTH = PORT_HEIGHT
SA_REYNOLDS_NUMBER = REYNOLDS_NUMBER
SA_SMOOTH_ABS_EPS = 1.0e-12
SA_INIT_WALL_DIST_SCALE = 0.05 * L
SA_NU_TILDE_FLOOR = 1.0e-12
SA_WALL_DENSITY_SOURCE = "passive" #'design'
SA_NU_TILDE_PENALTY_ALPHA = 30.0 #1.0e3
SA_NU_TILDE_PENALTY_N = 3.0

SA_WALL_SIGMA = 0.01
SA_WALL_G0 = 20.0
SA_WALL_PENALTY_ALPHA = 1.0e3
SA_WALL_PENALTY_N = 3.0
SA_WALL_G_FLOOR = 1.0e-8

VOL_FRAC = 0.27
OBJECTIVE_CONVERGENCE_TOL = 1.0e-5
OBJECTIVE_STREAK_TO_STOP = 5

Q_PENAL_SCHEDULE = [0.05, 0.20, 0.50, 1.00, 1.50, 2.50, 3.00, 3.00]
BETA_PROJ_SCHEDULE = [0.10, 0.50, 1.00, 2.00, 4.00, 8.00, 16.00, 24.00]
MOVE_LIMIT_SCHEDULE = [0.08, 0.06, 0.045, 0.03, 0.02, 0.01, 0.005, 0.002]
MAX_INNER_ITERATIONS_SCHEDULE = [12, 20, 35, 50, 70, 80, 90, 90] 

LINEAR_SOLVER = "mumps"

# FORWARD_FLOW_SOLVER = "snes" or FORWARD_FLOW_SOLVER = "ipcs".
# Keep IPCS as the default for continuity with existing U-bend runs; switch this
# to "snes" to use the monolithic frozen-viscosity Navier-Stokes solve.
FORWARD_FLOW_SOLVER = "ipcs"

# ============================================================================ #
# SNES forward solver parameters, used when FORWARD_FLOW_SOLVER = "snes".
FORWARD_SNES_METHOD = "newtonls"
FORWARD_SNES_LINE_SEARCH = "bt"
FORWARD_SNES_LINEAR_SOLVER = "mumps"
FORWARD_SNES_RTOL = 1.0e-6
FORWARD_SNES_ATOL = 1.0e-8
FORWARD_SNES_MAX_ITERS = 260
FORWARD_SNES_ERROR_ON_NONCONVERGENCE = False
FORWARD_SNES_ACCEPT_NONCONVERGED_WITH_ACCEPT_NORM = True
FORWARD_SNES_ACCEPTED_RESIDUAL_FACTOR = 1.0
FORWARD_SNES_STOP_AT_ACCEPT_NORM = True
FORWARD_SNES_ACCEPT_NORM_SOLVE_FACTOR = 0.05
FORWARD_SNES_MAX_ACCEPTED_ABSOLUTE_RESIDUAL = 1.0e-2

FORWARD_SNES_ADAPTIVE_CONVECTION = True
FORWARD_SNES_MIN_CONVECTION_STEP = 0.01
FORWARD_SNES_MAX_ADAPTIVE_CONVECTION_STEPS = 24
FORWARD_SNES_STARTUP_CONVECTION_SCHEDULE = [
    {"convection_weight": 0.00, "max_iters": 180, "atol": 2.0e-7, "accept_norm": 5.0e-3, "accept_nonconverged": True},
    {"convection_weight": 0.15, "max_iters": 220, "atol": 1.5e-7, "accept_norm": 4.0e-3, "accept_nonconverged": True},
    {"convection_weight": 0.35, "max_iters": 240, "atol": 1.0e-7, "accept_norm": 3.0e-3, "accept_nonconverged": True},
    {"convection_weight": 0.60, "max_iters": 260, "atol": 8.0e-8, "accept_norm": 2.0e-3, "accept_nonconverged": True},
    {"convection_weight": 0.85, "max_iters": 280, "atol": 5.0e-8, "accept_norm": 1.0e-3, "accept_nonconverged": True},
    {"convection_weight": 1.00, "max_iters": 320, "atol": 1.0e-8, "accept_norm": 5.0e-4, "accept_nonconverged": True},
]
FORWARD_SNES_CONVECTION_SCHEDULE = [
    {"convection_weight": 0.00, "max_iters": 120, "atol": 1.0e-7, "accept_norm": 2.0e-3, "accept_nonconverged": True},
    {"convection_weight": 0.30, "max_iters": 180, "atol": 8.0e-8, "accept_norm": 1.5e-3, "accept_nonconverged": True},
    {"convection_weight": 0.60, "max_iters": 220, "atol": 5.0e-8, "accept_norm": 1.0e-3, "accept_nonconverged": True},
    {"convection_weight": 0.85, "max_iters": 260, "atol": 2.5e-8, "accept_norm": 7.5e-4, "accept_nonconverged": True},
    {"convection_weight": 1.00, "max_iters": 300, "atol": 1.0e-8, "accept_norm": 5.0e-4, "accept_nonconverged": True},
]
FORWARD_SNES_RECOVERY_ATTEMPTS = [
    {
        "label": "current-iterate l2 line-search retry",
        "line_search": "l2",
        "max_iters": 320,
        "restart_with_stokes": False,
        "accept_norm": 7.5e-4,
        "accept_nonconverged": True,
    },
    {
        "label": "current-iterate trust-region retry",
        "method": "newtontr",
        "max_iters": 340,
        "restart_with_stokes": False,
        "accept_norm": 7.5e-4,
        "accept_nonconverged": True,
    },
    {
        "label": "Stokes rebuild backtracking retry",
        "line_search": "bt",
        "max_iters": 360,
        "restart_with_stokes": True,
        "accept_norm": 5.0e-4,
        "accept_nonconverged": True,
    },
    {
        "label": "Stokes rebuild l2 line-search retry",
        "line_search": "l2",
        "max_iters": 380,
        "restart_with_stokes": True,
        "accept_norm": 5.0e-4,
        "accept_nonconverged": True,
    },
]
# ============================================================================ #

PICARD_STEPS = 3
TURBULENCE_RELAXATION = 0.20
SA_EDDY_VISCOSITY_RATIO_CEILING = 50.0

# =========================================================================== #
# IPCS forward solver parameters, used when FORWARD_FLOW_SOLVER = "ipcs".
FORWARD_IPCS_DT = 2.5e-6
FORWARD_IPCS_MAX_ITERS = 300
FORWARD_IPCS_VELOCITY_RTOL = 1.0e-4
FORWARD_IPCS_PRESSURE_RTOL = 2.0e-3
FORWARD_IPCS_VEL_RELAXATION = 0.07
FORWARD_IPCS_P_RELAXATION = 0.02
FORWARD_IPCS_VEL_SOLVER = "bicgstab"
FORWARD_IPCS_VEL_PRECONDITIONER = "ilu"
FORWARD_IPCS_P_SOLVER = "bicgstab"
FORWARD_IPCS_P_PRECONDITIONER = "ilu"
FORWARD_IPCS_LOG_EVERY = 50
FORWARD_IPCS_MAX_RESTARTS = 2
FORWARD_IPCS_DT_REDUCTION_FACTOR = 0.5
FORWARD_IPCS_RELAXATION_REDUCTION_FACTOR = 0.7

FORWARD_IPCS_ACCEPT_BEST_SCORE = 1.0
FORWARD_IPCS_FINAL_ACCEPT_BEST_SCORE = 1.0
FORWARD_IPCS_ERROR_ON_NONCONVERGENCE = True
# =========================================================================== #

BETA_PROJ_VALUE = BETA_PROJ_SCHEDULE[0]
ETA_I = 0.50
QUADRATURE_DEGREE = 6
FILTER_RADIUS_IN_CELLS = 3.0

OUTLET_BC_TYPE = "pressure"
OUTLET_PRESSURE_VALUE = 0.0
PRESSURE_OUTLET_COMPONENT_BCS = [
    {"marker": "outlet", "component": 1, "value": 0.0},
]

ENABLE_PRESSURE_PIN = False
RESULTS_ROOT_NAME = "Results_Frozen/Results_UBendAlexandersen_TurbulentTO_Frozen"

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
