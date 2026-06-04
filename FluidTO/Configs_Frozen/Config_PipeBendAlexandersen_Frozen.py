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


# Resume from the reconstructed iteration-70 checkpoint by default.
RESUME_OPTIMIZATION = True


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

# Switch this between "dissipation" (J_D) and "average_inlet_pressure" (J_p).
# Objective-specific settings below keep both continuation paths in one config.
OBJECTIVE_TYPE = "average_inlet_pressure"
_OBJECTIVE_TYPE_NORMALIZED = OBJECTIVE_TYPE.strip().lower()
_USE_PRESSURE_OBJECTIVE = _OBJECTIVE_TYPE_NORMALIZED in (
    "average_inlet_pressure",
    "inlet_pressure",
    "mean_inlet_pressure",
)
_USE_DISSIPATION_OBJECTIVE = _OBJECTIVE_TYPE_NORMALIZED in (
    "dissipation",
    "power_dissipation",
    "volume_dissipation",
)
if not (_USE_PRESSURE_OBJECTIVE or _USE_DISSIPATION_OBJECTIVE):
    raise ValueError(
        "OBJECTIVE_TYPE must be 'dissipation' or 'average_inlet_pressure', got {!r}.".format(
            OBJECTIVE_TYPE
        )
    )

ALPHA_SOLID_JD = 1.0e5
ALPHA_SOLID_JP = 1.0e3
ALPHA_SOLID = ALPHA_SOLID_JP if _USE_PRESSURE_OBJECTIVE else ALPHA_SOLID_JD

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

# Stabilize the advection-dominated SA working-variable transport in bend TO
# cases. Keep pseudo-time damping disabled unless nu_tilde still spikes.
SA_SUPG_STABILIZATION = True
SA_SUPG_TAU_SCALE = 1.0
SA_PSEUDO_TIME_STABILIZATION = False
SA_PSEUDO_DT = 1.0e-4
SA_PSEUDO_TIME_STEPS = 3

SAVE_SA_CLIPPING_DIAGNOSTICS_JD = False
SAVE_SA_CLIPPING_DIAGNOSTICS_JP = False
SAVE_SA_CLIPPING_DIAGNOSTICS = (
    SAVE_SA_CLIPPING_DIAGNOSTICS_JP
    if _USE_PRESSURE_OBJECTIVE else SAVE_SA_CLIPPING_DIAGNOSTICS_JD
)

# Topology-created solids act as walls for the reciprocal wall-distance solve.
SA_NU_TILDE_PENALTY_ALPHA = 1.0e3
SA_NU_TILDE_PENALTY_ALPHA_SCHEDULE_JD = [1e1, 3e1, 1e2, 1e3, 1e4, 3e4, 1e5]

SA_NU_TILDE_PENALTY_ALPHA_SCHEDULE_JP = [1.0e3]

SA_NU_TILDE_PENALTY_ALPHA_SCHEDULE = (
    SA_NU_TILDE_PENALTY_ALPHA_SCHEDULE_JP
    if _USE_PRESSURE_OBJECTIVE else SA_NU_TILDE_PENALTY_ALPHA_SCHEDULE_JD
)
SA_NU_TILDE_PENALTY_N = 4.0
SA_NU_TILDE_PENALTY_INTERPOLATION = "power"

SA_WALL_DISTANCE_MODE = "reciprocal_penalized"
SA_WALL_SIGMA = 0.01
SA_WALL_G0 = 20.0
SA_WALL_PENALTY_ALPHA = 1.0e3
SA_WALL_PENALTY_ALPHA_SCHEDULE_JD = [1e1, 3e1, 1e2, 1e3, 1e4, 3e4, 1e5]

SA_WALL_PENALTY_ALPHA_SCHEDULE_JP = [1.0e3]

SA_WALL_PENALTY_ALPHA_SCHEDULE = (
    SA_WALL_PENALTY_ALPHA_SCHEDULE_JP
    if _USE_PRESSURE_OBJECTIVE else SA_WALL_PENALTY_ALPHA_SCHEDULE_JD
)
SA_WALL_PENALTY_N = 4.0
SA_WALL_PENALTY_INTERPOLATION = "power"
SA_WALL_G_FLOOR = 1.0e-8

SA_WALL_DENSITY_SOURCE = "design"
SA_WALL_SOLID_THRESHOLD = 0.50
SA_WALL_DISTANCE_FLOOR = 0.25 * H_MAX

# ================================================================== #
# MMA Objective and Continuation Parameters
VOL_FRAC = 0.25
OBJECTIVE_CONVERGENCE_TOL_JD = 1.0e-6
OBJECTIVE_CONVERGENCE_TOL_JP = 1.0e-5
OBJECTIVE_STREAK_TO_STOP_JD = 10
OBJECTIVE_STREAK_TO_STOP_JP = 14
INITIAL_DENSITY_VALUE = VOL_FRAC
INITIAL_DENSITY_MATCH_FILTERED_VOLUME = True


# Stabilized continuation for frozen-SA bend optimization. J_D keeps the
# current robust path. J_p records the smooth-continuation settings used for
# the pressure-objective pipe-bend results.
Q_PENAL_SCHEDULE_JD = [0.01, 0.02, 0.04, 0.08, 0.08, 0.12, 0.20]
BETA_PROJ_SCHEDULE_JD = [1.0, 2.0, 4.0, 8.0, 8.0, 12.0, 16.0]
MOVE_LIMIT_SCHEDULE_JD = [0.010, 0.008, 0.006, 0.004, 0.003, 0.002, 0.0015]
MAX_INNER_ITERATIONS_SCHEDULE_JD = [60, 80, 100, 120, 160, 180, 220]

CONTINUATION_VARIANT = "smooth_continuation"

PAPER_Q_ALPHA_LEGACY = [150.0, 75.0, 30.0, 15.0]
Q_PENAL_LEGACY = [1.0 / q_alpha for q_alpha in PAPER_Q_ALPHA_LEGACY]
BETA_PROJ_LEGACY = [4.0, 6.0, 9.0, 13.0]
MOVE_LIMIT_LEGACY = [0.05, 0.04, 0.03, 0.02]
MAX_INNER_ITERATIONS_LEGACY = [25, 25, 25, 25]

if CONTINUATION_VARIANT == "single_stage_500":
    PAPER_Q_ALPHA_SCHEDULE_JP = [150.0]
    Q_PENAL_SCHEDULE_JP = [1.0 / PAPER_Q_ALPHA_SCHEDULE_JP[0]]
    BETA_PROJ_SCHEDULE_JP = [4.0]
    MOVE_LIMIT_SCHEDULE_JP = [0.010]
    MAX_INNER_ITERATIONS_SCHEDULE_JP = [500]
    OBJECTIVE_CONVERGENCE_TOL_JP = 0.0
    OBJECTIVE_STREAK_TO_STOP_JP = 500
elif CONTINUATION_VARIANT == "smooth_continuation":
    # Stage 1 matches the completed q=1/150, beta=4 segment and is capped at
    # 70 iterations, so a checkpoint at inner_count=70 advances directly into
    # the damped continuation stages.
    PAPER_Q_ALPHA_SCHEDULE_JP = [150.0, 120.0, 100.0, 75.0, 50.0, 30.0, 15.0]
    Q_PENAL_SCHEDULE_JP = [
        1.0 / q_alpha for q_alpha in PAPER_Q_ALPHA_SCHEDULE_JP
    ]
    BETA_PROJ_SCHEDULE_JP = [4.0, 4.5, 4.75, 5.25, 6.0, 7.0, 8.0]
    MOVE_LIMIT_SCHEDULE_JP = [0.010, 0.0015, 0.00125, 0.0010, 0.0008, 0.0007, 0.0006]
    MAX_INNER_ITERATIONS_SCHEDULE_JP = [70, 140, 160, 180, 200, 220, 260]
    OBJECTIVE_CONVERGENCE_TOL_JP = 1.0e-5
    OBJECTIVE_STREAK_TO_STOP_JP = 14
elif CONTINUATION_VARIANT == "legacy_8fedc94":
    PAPER_Q_ALPHA_SCHEDULE_JP = PAPER_Q_ALPHA_LEGACY
    Q_PENAL_SCHEDULE_JP = Q_PENAL_LEGACY
    BETA_PROJ_SCHEDULE_JP = BETA_PROJ_LEGACY
    MOVE_LIMIT_SCHEDULE_JP = MOVE_LIMIT_LEGACY
    MAX_INNER_ITERATIONS_SCHEDULE_JP = MAX_INNER_ITERATIONS_LEGACY
    OBJECTIVE_CONVERGENCE_TOL_JP = 1.0e-5
    OBJECTIVE_STREAK_TO_STOP_JP = 5
else:
    raise ValueError(
        "CONTINUATION_VARIANT must be 'single_stage_500', "
        "'smooth_continuation', or 'legacy_8fedc94'."
    )

OBJECTIVE_CONVERGENCE_TOL = (
    OBJECTIVE_CONVERGENCE_TOL_JP
    if _USE_PRESSURE_OBJECTIVE else OBJECTIVE_CONVERGENCE_TOL_JD
)
OBJECTIVE_STREAK_TO_STOP = (
    OBJECTIVE_STREAK_TO_STOP_JP
    if _USE_PRESSURE_OBJECTIVE else OBJECTIVE_STREAK_TO_STOP_JD
)

Q_PENAL_SCHEDULE = (
    Q_PENAL_SCHEDULE_JP if _USE_PRESSURE_OBJECTIVE else Q_PENAL_SCHEDULE_JD
)
BETA_PROJ_SCHEDULE = (
    BETA_PROJ_SCHEDULE_JP if _USE_PRESSURE_OBJECTIVE else BETA_PROJ_SCHEDULE_JD
)
MOVE_LIMIT_SCHEDULE = (
    MOVE_LIMIT_SCHEDULE_JP if _USE_PRESSURE_OBJECTIVE else MOVE_LIMIT_SCHEDULE_JD
)
MAX_INNER_ITERATIONS_SCHEDULE = (
    MAX_INNER_ITERATIONS_SCHEDULE_JP
    if _USE_PRESSURE_OBJECTIVE else MAX_INNER_ITERATIONS_SCHEDULE_JD
)

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

FORWARD_FLOW_SOLVER = "snes"
FORWARD_PICARD_FLOW_SOLVER = "ipcs"
FORWARD_SNES_WARM_START_WITH_IPCS = True
FORWARD_SNES_IPCS_WARM_START_MODE = "final"


FORWARD_SNES_STRICT_FINAL_SOLVE = False


# =========================================================================== #
# SNES forward solver parameters, used when FORWARD_FLOW_SOLVER = "snes"
FORWARD_SNES_METHOD = "newtonls"
FORWARD_SNES_LINE_SEARCH = "bt"
FORWARD_SNES_LINEAR_SOLVER = "mumps"
FORWARD_SNES_RTOL = 1.0e-6
FORWARD_SNES_ATOL = 1.0e-8
FORWARD_SNES_MAX_ITERS = 260

FORWARD_SNES_ADAPTIVE_CONVECTION = True
FORWARD_SNES_MIN_CONVECTION_STEP = 0.0025
FORWARD_SNES_MAX_ADAPTIVE_CONVECTION_STEPS = 80


FORWARD_SNES_STOP_AT_ACCEPT_NORM = True
FORWARD_SNES_ACCEPT_NORM_SOLVE_FACTOR = 0.5

# If one MMA update creates a design whose forward solve cannot be continued,
# retry smaller fractions of that last design step before giving up.
FORWARD_RETRY_BACKTRACK_DESIGN = True
FORWARD_RETRY_BACKTRACK_FACTORS = [0.5, 0.25, 0.10, 0.05]


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

PICARD_STEPS_JD = 5
PICARD_STEPS_JP = 3
TURBULENCE_RELAXATION_JD = 0.05
TURBULENCE_RELAXATION_JP = 0.04
PICARD_STEPS = PICARD_STEPS_JP if _USE_PRESSURE_OBJECTIVE else PICARD_STEPS_JD
TURBULENCE_RELAXATION = (
    TURBULENCE_RELAXATION_JP
    if _USE_PRESSURE_OBJECTIVE else TURBULENCE_RELAXATION_JD
)

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

FORWARD_IPCS_PICARD_VELOCITY_RTOL = FORWARD_IPCS_VELOCITY_RTOL
FORWARD_IPCS_PICARD_PRESSURE_RTOL = FORWARD_IPCS_PRESSURE_RTOL
FORWARD_IPCS_PICARD_ACCEPT_BEST_SCORE = FORWARD_IPCS_ACCEPT_BEST_SCORE
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
SAVE_DF0DX_VECTOR = False
SAVE_IPCS_RESIDUAL_PLOTS = False
SAVE_IPCS_RESIDUAL_SVGS = False

RESULTS_ROOT_BASE_NAME = "Results_Frozen/Results_PipeBendAlexandersen_Frozen"
if _USE_DISSIPATION_OBJECTIVE:
    RESULTS_ROOT_NAME = RESULTS_ROOT_BASE_NAME + "_JD"
elif _USE_PRESSURE_OBJECTIVE:
    RESULTS_ROOT_SUFFIX_BY_VARIANT = {
        "single_stage_500": "_Jp_Old",
        "smooth_continuation": "_Jp_SmoothContinuation",
        "legacy_8fedc94": "_Jp_Legacy8fedc94",
    }
    RESULTS_ROOT_NAME = RESULTS_ROOT_BASE_NAME + RESULTS_ROOT_SUFFIX_BY_VARIANT[
        CONTINUATION_VARIANT
    ]
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
