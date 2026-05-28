import os

from dolfin import DOLFIN_EPS, Expression, MeshFunction, MPI, SubDomain, near

from Utilities_SharedTO import (
    build_cell_tag_restriction_functions,
    eddy_viscosity_ratio_from_turbulence_intensity,
    load_mesh_from_xdmf,
)
from Utilities_TurbulentTO_Frozen import nu_tilde_from_viscosity_ratio


# ===================================================================
# Configuration: Alexandersen 2026 Pipe-Bend - Turbulent Frozen (SA)
#
# Geometry, mesh size, and boundary conditions follow Figure 10 and
# Table 3 of Alexandersen (2026). The turbulence closure deliberately
# remains the in-house frozen Spalart-Allmaras implementation for the
# thesis comparison.
# ===================================================================

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

mesh_files = {
    "MESH_DIRECTORY": os.path.join(REPO_ROOT, "Meshes/PipeBendAlexandersen/mesh_yplus1.xdmf"),
    "CELL_DIRECTORY": os.path.join(REPO_ROOT, "Meshes/PipeBendAlexandersen/cell_yplus1.xdmf"),
}

DESIGN_DOMAIN_TAG = 1
NON_DESIGN_FLUID_TAG = 2


RESUME_OPTIMIZATION = False


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
ALPHA_FLUID = 0.0
ALPHA_SOLID = 1.0e5

# ==============================================================================================
# Reynolds number: Re = U_MAX_INLET * INLET_HEIGHT * RHO_FLUID_VALUE / MU_FLUID_VALUE = 5,000
# ==============================================================================================

# SA inlet data are modelling choices for this solver, not values from Alexandersen's k-epsilon paper.
SA_TURBULENCE_INTENSITY = 0.075
SA_TURBULENCE_LENGTH_SCALE_RATIO = 0.05
SA_REFERENCE_VELOCITY = U_MAX_INLET
SA_REFERENCE_LENGTH = INLET_HEIGHT
SA_REYNOLDS_NUMBER = REYNOLDS_NUMBER
SA_SMOOTH_ABS_EPS = 1.0e-12
SA_INIT_WALL_DIST_SCALE = 0.05 * L
SA_NU_TILDE_FLOOR = 1.0e-12
SA_LAMINAR_KINEMATIC_VISCOSITY = MU_FLUID_VALUE / RHO_FLUID_VALUE
SA_INLET_EDDY_VISCOSITY_RATIO = eddy_viscosity_ratio_from_turbulence_intensity(
    SA_TURBULENCE_INTENSITY,
    SA_LAMINAR_KINEMATIC_VISCOSITY,
    length_scale_ratio=SA_TURBULENCE_LENGTH_SCALE_RATIO,
    reynolds_number=SA_REYNOLDS_NUMBER,
)
SA_NU_TILDE_INITIAL = nu_tilde_from_viscosity_ratio(
    SA_INLET_EDDY_VISCOSITY_RATIO,
    SA_LAMINAR_KINEMATIC_VISCOSITY,
)
SA_NU_TILDE_CEILING = None
SA_EDDY_VISCOSITY_RATIO_CEILING = 50.0

SAVE_SA_CLIPPING_DIAGNOSTICS = False

# Topology-created solids act as walls for the reciprocal wall-distance solve.
# The paper's implicit k-epsilon wall-function parameters psi_max=1000 and
# P_con=4 are used here as the closest SA analogues: wall penalty amplitude
# and solid-indicator exponent.
SA_NU_TILDE_PENALTY_ALPHA = 1.0e5
SA_NU_TILDE_PENALTY_ALPHA_SCHEDULE = [1.0e3, 3.0e3, 1.0e4, 3.0e4, 1.0e5, 1.0e5, 1.0e5]
SA_NU_TILDE_PENALTY_N = 3.0
SA_NU_TILDE_PENALTY_INTERPOLATION = "power"

SA_WALL_DISTANCE_MODE = "reciprocal_penalized"
SA_WALL_SIGMA = 0.1
SA_WALL_G0 = 20.0
SA_WALL_PENALTY_ALPHA = 1.0e5
SA_WALL_PENALTY_ALPHA_SCHEDULE = [1.0e3, 3.0e3, 1.0e4, 3.0e4, 1.0e5, 1.0e5, 1.0e5]
SA_WALL_PENALTY_N = 3.0
SA_WALL_PENALTY_INTERPOLATION = "power"
SA_WALL_G_FLOOR = 1.0e-8

SA_WALL_DENSITY_SOURCE = "design"
SA_WALL_SOLID_THRESHOLD = 0.50
SA_WALL_DISTANCE_FLOOR = 0.25 * H_MAX

# ================================================================== #
# MMA Objective and Continuation Parameters
VOL_FRAC = 0.25
OBJECTIVE_TYPE = "dissipation"
OBJECTIVE_CONVERGENCE_TOL = 1.0e-6
OBJECTIVE_STREAK_TO_STOP = 10
INITIAL_DENSITY_VALUE = VOL_FRAC
INITIAL_DENSITY_MATCH_FILTERED_VOLUME = True

# Paper: alpha(phi) = alpha_max * (1 - phi) / (1 + q_a * phi).
# Code:  alpha(rho) = alpha_solid * (1 - rho) / (1 + rho / q_penal)
# when ALPHA_FLUID = 0, so q_penal = 1 / q_a.

#PAPER_Q_ALPHA_SCHEDULE = [150.0, 75.0, 30.0, 15.0]
#Q_PENAL_SCHEDULE = [1.0 / q_alpha for q_alpha in PAPER_Q_ALPHA_SCHEDULE]
#BETA_PROJ_SCHEDULE = [4.0, 6.0, 9.0, 13.0]
#MAX_INNER_ITERATIONS_SCHEDULE = [25, 25, 25, 25]
#MOVE_LIMIT_SCHEDULE = [0.05, 0.04, 0.03, 0.02]

# Stabilized pressure-drop continuation for frozen-SA bend optimization.
# The late hold stage and smaller move limits damp MMA oscillations after
# projection and Brinkman penalization become stiff.
Q_PENAL_SCHEDULE = [0.01, 0.02, 0.04, 0.08, 0.08, 0.12, 0.20]
BETA_PROJ_SCHEDULE = [1.0, 2.0, 4.0, 8.0, 8.0, 12.0, 16.0]
MOVE_LIMIT_SCHEDULE = [0.015, 0.012, 0.010, 0.006, 0.004, 0.003, 0.002]
MAX_INNER_ITERATIONS_SCHEDULE = [25, 45, 80, 130, 180, 220, 260]

MAX_INNER_ITERATIONS = MAX_INNER_ITERATIONS_SCHEDULE[0]

# Keep the production pipe-bend run focused on topology optimization. The
# one-shot sensitivity verification lives in
# Config_PipeBendAlexandersen_Frozen_Sensitivity.py.
RUN_FINITE_DIFFERENCE_CHECKS = False
RUN_TAYLOR_SENSITIVITY_CHECKS = False
STOP_AFTER_FINITE_DIFFERENCE_CHECKS = False
FINITE_DIFFERENCE_CHECK_FLOW_SOLVER = "same"
FINITE_DIFFERENCE_CHECK_PICARD_FLOW_SOLVER = "same"
FINITE_DIFFERENCE_CHECK_ITERATIONS = (0,)
FINITE_DIFFERENCE_CHECK_ACTIVE_POSITIONS = None
FINITE_DIFFERENCE_CHECK_DOF_INDICES = None
FINITE_DIFFERENCE_CHECK_STEP = 1.0e-3
FINITE_DIFFERENCE_CHECK_SAMPLES = 4
FINITE_DIFFERENCE_CHECK_SEED = 13
FINITE_DIFFERENCE_CHECK_CLIP_TO_BOUNDS = True
FINITE_DIFFERENCE_CHECK_UPDATED_TURBULENCE = False
FINITE_DIFFERENCE_CHECK_PICARD_STEPS = 1
TAYLOR_CHECK_STEPS = (1.0e-3, 3.0e-4, 1.0e-4, 3.0e-5)
TAYLOR_CHECK_SEED = 29
SENSITIVITY_VERIFICATION_MODE_NAME = "frozen"
SENSITIVITY_VERIFICATION_DERIVATIVE_COLUMN = "AdjointDerivative"
DILGEN_FROZEN_RESULTS_ROOT_NAME = None
DILGEN_SEMIFROZEN_RESULTS_ROOT_NAME = None
# ================================================================== #

LINEAR_SOLVER = "mumps"

# FORWARD_FLOW_SOLVER = "snes" or FORWARD_FLOW_SOLVER = "ipcs".
# Use IPCS for frozen-SA Picard updates, then use SNES for the final flow state
# that feeds the adjoint. This avoids high-Re Newton failures during Picard.

FORWARD_FLOW_SOLVER = "snes"        # Solver for [Final Flow]
FORWARD_PICARD_FLOW_SOLVER = "ipcs" # Solver for [Picard] flow
FORWARD_SNES_WARM_START_WITH_IPCS = True
FORWARD_SNES_IPCS_WARM_START_MODE = "final" # "initial" or "final"


FORWARD_SNES_STRICT_FINAL_SOLVE = True


# =========================================================================== #
# SNES forward solver parameters, used when FORWARD_FLOW_SOLVER = "snes"
FORWARD_SNES_METHOD = "newtonls"
FORWARD_SNES_LINE_SEARCH = "bt"
FORWARD_SNES_LINEAR_SOLVER = "mumps"
FORWARD_SNES_RTOL = 1.0e-6
FORWARD_SNES_ATOL = 1.0e-8
FORWARD_SNES_MAX_ITERS = 220

FORWARD_SNES_ADAPTIVE_CONVECTION = True
#FORWARD_SNES_MIN_CONVECTION_STEP = 0.03
#FORWARD_SNES_MAX_ADAPTIVE_CONVECTION_STEPS = 16
FORWARD_SNES_MIN_CONVECTION_STEP = 0.005
FORWARD_SNES_MAX_ADAPTIVE_CONVECTION_STEPS = 40


FORWARD_SNES_STOP_AT_ACCEPT_NORM = False


FORWARD_SNES_STARTUP_CONVECTION_SCHEDULE = [
    {"convection_weight": 0.00, "max_iters": 80, "atol": 2.0e-7, "accept_norm": 2.0e-4, "accept_nonconverged": True},
    {"convection_weight": 0.20, "max_iters": 100, "atol": 1.5e-7, "accept_norm": 1.5e-4, "accept_nonconverged": True},
    {"convection_weight": 0.40, "max_iters": 120, "atol": 1.0e-7, "accept_norm": 1.0e-4, "accept_nonconverged": True},
    {"convection_weight": 0.60, "max_iters": 140, "atol": 8.0e-8, "accept_norm": 8.0e-5, "accept_nonconverged": True},
    {"convection_weight": 0.80, "max_iters": 160, "atol": 5.0e-8, "accept_norm": 6.0e-5, "accept_nonconverged": True},
    {"convection_weight": 1.00, "max_iters": 180, "atol": 1.0e-8, "accept_norm": 5.0e-5, "accept_nonconverged": True},
]
FORWARD_SNES_CONVECTION_SCHEDULE = [
    {"convection_weight": 0.00, "max_iters": 60, "atol": 1.0e-7, "accept_norm": 1.5e-4, "accept_nonconverged": True},
    {"convection_weight": 0.25, "max_iters": 80, "atol": 8.0e-8, "accept_norm": 1.0e-4, "accept_nonconverged": True},
    {"convection_weight": 0.50, "max_iters": 100, "atol": 5.0e-8, "accept_norm": 8.0e-5, "accept_nonconverged": True},
    {"convection_weight": 0.75, "max_iters": 120, "atol": 2.5e-8, "accept_norm": 6.0e-5, "accept_nonconverged": True},
    {"convection_weight": 1.00, "max_iters": 140, "atol": 1.0e-8, "accept_norm": 5.0e-5, "accept_nonconverged": True},
]
FORWARD_SNES_RECOVERY_ATTEMPTS = [
    {
        "label": "current-iterate l2 line-search retry",
        "line_search": "l2",
        "max_iters": 180,
        "restart_with_stokes": False,
        "accept_norm": 7.5e-5,
        "accept_nonconverged": True,
    },
    {
        "label": "current-iterate trust-region retry",
        "method": "newtontr",
        "max_iters": 220,
        "restart_with_stokes": False,
        "accept_norm": 7.5e-5,
        "accept_nonconverged": True,
    },
    {
        "label": "Stokes rebuild backtracking retry",
        "line_search": "bt",
        "max_iters": 240,
        "restart_with_stokes": True,
        "accept_norm": 5.0e-5,
        "accept_nonconverged": True,
    },
    {
        "label": "Stokes rebuild l2 line-search retry",
        "line_search": "l2",
        "max_iters": 260,
        "restart_with_stokes": True,
        "accept_norm": 5.0e-5,
        "accept_nonconverged": True,
    },
]

# =========================================================================== #

PICARD_STEPS = 5
TURBULENCE_RELAXATION = 0.05

# =========================================================================== #
# IPCS forward solver parameters, used when FORWARD_FLOW_SOLVER = "ipcs"
FORWARD_IPCS_DT = 5.0e-6
FORWARD_IPCS_MAX_ITERS = 600
FORWARD_IPCS_VELOCITY_RTOL = 2.0e-4
FORWARD_IPCS_PRESSURE_RTOL = 1.0e-3
FORWARD_IPCS_VEL_RELAXATION = 0.10
FORWARD_IPCS_P_RELAXATION = 0.03
FORWARD_IPCS_VEL_SOLVER = "bicgstab"
FORWARD_IPCS_VEL_PRECONDITIONER = "ilu"
FORWARD_IPCS_P_SOLVER = "bicgstab"
FORWARD_IPCS_P_PRECONDITIONER = "ilu"
FORWARD_IPCS_LOG_EVERY = 50
FORWARD_IPCS_MAX_RESTARTS = 3
FORWARD_IPCS_DT_REDUCTION_FACTOR = 0.5
FORWARD_IPCS_RELAXATION_REDUCTION_FACTOR = 0.7
FORWARD_IPCS_ACCEPT_BEST_SCORE = 1.0

FORWARD_IPCS_PICARD_VELOCITY_RTOL = 1.0e-3
FORWARD_IPCS_PICARD_ACCEPT_BEST_SCORE = 1.5
# =========================================================================== #


BETA_PROJ_VALUE = BETA_PROJ_SCHEDULE[0]
ETA_I = 0.50
QUADRATURE_DEGREE = 6
FILTER_BASE_LENGTH = H_MAX
FILTER_RADIUS_IN_CELLS = 4.0
FILTER_DENOMINATOR_FLOOR = 1.0e-12

OUTLET_BC_TYPE = "pressure"
OUTLET_PRESSURE_VALUE = 0.0
PRESSURE_OUTLET_COMPONENT_BCS = [
    {"marker": "outlet", "component": 0, "value": 0.0},
]

ENABLE_PRESSURE_PIN = False
SAVE_DILGEN_PAPER_DATA = False
LOG_DILGEN_FIG8_COLUMNS = False
SAVE_DF0DX_VECTOR = True
SAVE_IPCS_RESIDUAL_PLOTS = False
SAVE_IPCS_RESIDUAL_SVGS = False
RESULTS_ROOT_BASE_NAME = "Results_Frozen/Results_PipeBendAlexandersen_Frozen"
if OBJECTIVE_TYPE == "dissipation":
    RESULTS_ROOT_NAME = RESULTS_ROOT_BASE_NAME + "_JD"
elif OBJECTIVE_TYPE == "average_inlet_pressure":
    RESULTS_ROOT_NAME = RESULTS_ROOT_BASE_NAME + "_Jp"
else:
    RESULTS_ROOT_NAME = RESULTS_ROOT_BASE_NAME

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
