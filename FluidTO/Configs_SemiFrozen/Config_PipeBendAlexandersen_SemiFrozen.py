import os

from dolfin import DOLFIN_EPS, Expression, MeshFunction, MPI, SubDomain, near

from Utilities_SharedTO import build_cell_tag_restriction_functions, load_mesh_from_xdmf


# ===================================================================
# Configuration: Alexandersen 2026 Pipe-Bend - Turbulent SemiFrozen (SA)
#
# Geometry, mesh size, and boundary conditions follow Figure 10 and
# Table 3 of Bayat, Li, and Alexandersen (2026), while the turbulence
# closure remains the in-house SemiFrozen Spalart-Allmaras implementation.
# ===================================================================

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

mesh_files = {
    "MESH_DIRECTORY": os.path.join(REPO_ROOT, "Meshes/PipeBendAlexandersen/mesh_yplus1.xdmf"),
    "CELL_DIRECTORY": os.path.join(REPO_ROOT, "Meshes/PipeBendAlexandersen/cell_yplus1.xdmf"),
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
INLET_HEIGHT = 0.2 * L
OUTLET_WIDTH = 0.2 * L
INLET_Y_MIN = 0.7 * L
INLET_Y_MAX = INLET_Y_MIN + INLET_HEIGHT
OUTLET_X_MIN = 0.7 * L
OUTLET_X_MAX = OUTLET_X_MIN + OUTLET_WIDTH

DESIGN_X_MIN = 0.0
DESIGN_Y_MIN = 0.0
DESIGN_X_MAX = L
DESIGN_Y_MAX = L
DOMAIN_X_MIN = DESIGN_X_MIN - LEAD_LENGTH
DOMAIN_Y_MIN = DESIGN_Y_MIN - LEAD_LENGTH
DOMAIN_X_MAX = DESIGN_X_MAX
DOMAIN_Y_MAX = DESIGN_Y_MAX
TOL = DOLFIN_EPS

REYNOLDS_NUMBER = 5000.0
RHO_FLUID_VALUE = 1.0
U_MAX_INLET = 1.0
MU_FLUID_VALUE = U_MAX_INLET * INLET_HEIGHT * RHO_FLUID_VALUE / REYNOLDS_NUMBER

# ==============================================================================================
# Reynolds number: Re = U_MAX_INLET * INLET_HEIGHT * RHO_FLUID_VALUE / MU_FLUID_VALUE = 5,000
# ==============================================================================================

SA_TURBULENCE_INTENSITY = 0.075
SA_TURBULENCE_LENGTH_SCALE_RATIO = 0.05
SA_REFERENCE_VELOCITY = U_MAX_INLET
SA_REFERENCE_LENGTH = INLET_HEIGHT
SA_REYNOLDS_NUMBER = REYNOLDS_NUMBER
SA_SMOOTH_ABS_EPS = 1.0e-12
SA_INIT_WALL_DIST_SCALE = 0.05 * L
SA_NU_TILDE_FLOOR = 1.0e-12
SA_NU_TILDE_PENALTY_ALPHA = 1.0e3
SA_NU_TILDE_PENALTY_N = 3.0
SA_NU_TILDE_PENALTY_INTERPOLATION = "power"

# Relaxed wall equation parameters for the external reciprocal distance solve.
SA_WALL_DISTANCE_MODE = "reciprocal_penalized"
SA_WALL_SIGMA = 0.01
SA_WALL_G0 = 20.0
SA_WALL_PENALTY_ALPHA = 1.0e3
SA_WALL_PENALTY_N = 3.0
SA_WALL_PENALTY_INTERPOLATION = "power"
SA_WALL_G_FLOOR = 1.0e-8

SA_WALL_DENSITY_SOURCE = "design"  # with "passive", the reciprocal wall-distance solver uses density_upper_bound
SA_WALL_SOLID_THRESHOLD = 0.10
SA_WALL_DISTANCE_FLOOR = 0.25 * H_MAX
STATE_INITIAL_SA_RELAXATION = 0.25
STATE_INITIAL_SA_SWEEPS = 4

VOL_FRAC = 0.25
OBJECTIVE_TYPE = "average_inlet_pressure"
OBJECTIVE_CONVERGENCE_TOL = 1.0e-5
OBJECTIVE_STREAK_TO_STOP = 5
SAVE_SEMIFROZEN_DIAGNOSTICS = True

# Use a shorter pipe-bend continuation for exploratory optimization runs.
# Reserve beta > 32 and additional q=3 holds for final polishing only.
Q_PENAL_SCHEDULE = [0.05, 0.20, 0.50, 1.00, 1.50, 2.50, 3.00, 3.00]
MOVE_LIMIT_SCHEDULE = [0.04, 0.035, 0.03, 0.025, 0.02, 0.01, 0.005, 0.002]
BETA_PROJ_SCHEDULE = [0.10, 0.50, 1.00, 2.00, 4.00, 8.00, 16.00, 32.00]
MAX_INNER_ITERATIONS_SCHEDULE = [8, 10, 15, 20, 25, 30, 40, 50]

LINEAR_SOLVER = "mumps"
STATE_SOLVE_METHOD = "newtontr"
STATE_LINE_SEARCH = "bt"
STATE_RTOL = 1.0e-6
STATE_ATOL = 1.0e-8
STATE_MAX_ITERS = 120

STATE_ACCEPTED_RESIDUAL_FACTOR = 1.5
STATE_ACCEPT_NONCONVERGED_WITH_ACCEPT_NORM = True
STATE_ADAPTIVE_COUPLING = True
STATE_MIN_COUPLING_STEP = 0.01
STATE_MAX_ADAPTIVE_COUPLING_STEPS = 16

STATE_TURBULENCE_COUPLING_SCHEDULE = [
    {"convection_weight": 0.00, "weight": 0.00, "max_iters": 80,  "atol": 1.5e-3, "accept_norm": 1.5e-3},
    {"convection_weight": 0.25, "weight": 0.00, "max_iters": 120, "atol": 1.2e-3, "accept_norm": 1.2e-3},
    {"convection_weight": 0.50, "weight": 0.00, "max_iters": 160, "atol": 1.0e-3, "accept_norm": 1.5e-3},
    {"convection_weight": 0.65, "weight": 0.00, "max_iters": 180, "atol": 1.0e-3, "accept_norm": 1.8e-3},
    {"convection_weight": 0.80, "weight": 0.00, "max_iters": 200, "atol": 1.0e-3, "accept_norm": 2.0e-3},
    {"convection_weight": 1.00, "weight": 0.00, "max_iters": 220, "atol": 1.0e-3, "accept_norm": 2.0e-3},

    {"convection_weight": 1.00, "weight": 0.10, "max_iters": 180, "atol": 9.0e-4, "accept_norm": 2.0e-3},
    {"convection_weight": 1.00, "weight": 0.25, "max_iters": 200, "atol": 8.0e-4, "accept_norm": 1.8e-3},
    {"convection_weight": 1.00, "weight": 0.50, "max_iters": 240, "atol": 7.5e-4, "accept_norm": 1.5e-3},
    {"convection_weight": 1.00, "weight": 0.75, "max_iters": 280, "atol": 7.5e-4, "accept_norm": 1.2e-3},
    {"convection_weight": 1.00, "weight": 1.00, "max_iters": 320},
]

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
        "max_iters": 260,
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
    {"marker": "outlet", "component": 0, "value": 0.0},
]

ENABLE_PRESSURE_PIN = False
RESULTS_ROOT_NAME = "Results_SemiFrozen/Results_PipeBendAlexandersen_SemiFrozen"

MARK = {"generic": 0, "walls": 1, "inlet": 2, "outlet": 3}


def between(value, limits, eps=DOLFIN_EPS):
    return (limits[0] - eps <= value) and (value <= limits[1] + eps)


def is_inlet_point(x):
    return near(x[0], DOMAIN_X_MIN, TOL) and between(x[1], (INLET_Y_MIN, INLET_Y_MAX), TOL)


def is_outlet_point(x):
    return near(x[1], DOMAIN_Y_MIN, TOL) and between(x[0], (OUTLET_X_MIN, OUTLET_X_MAX), TOL)


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
