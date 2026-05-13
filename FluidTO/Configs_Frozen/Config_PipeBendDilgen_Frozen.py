import os

import numpy as np

from dolfin import Constant, DOLFIN_EPS, Expression, MeshFunction, MPI, SubDomain, UserExpression, near

from Utilities_SharedTO import load_mesh_from_xdmf
from Utilities_TurbulentTO_Frozen import nu_tilde_from_viscosity_ratio

try:
    _trapezoid = np.trapezoid
except AttributeError:
    _trapezoid = np.trapz


# ===================================================================
# Configuration: Dilgen 2018 Channel/Pipe-Bend - Turbulent Frozen (SA)
#
# Dilgen's 90 degree bend verification case uses the same square bend
# layout, no inlet/outlet non-design extensions, a fully developed
# turbulent inlet profile, Ub = 5 m/s, nu = 5e-5 m^2/s, and p = 0 at
# the outlet. The inlet velocity and turbulence profiles in the paper are
# obtained from an a-priori fully developed precursor simulation. This config
# reads the repository-local SA precursor profiles generated in
# PrecursorProfiles/DilgenPipeBendSA.
# ===================================================================

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

mesh_files = {
    "MESH_DIRECTORY": os.path.join(REPO_ROOT, "Meshes/PipeBendDilgen/mesh_yplus1.xdmf"),
}

INLET_PROFILE_CSV = os.environ.get(
    "DILGEN_INLET_PROFILE_CSV",
    os.path.join(
        REPO_ROOT,
        "PrecursorProfiles/DilgenPipeBendSA/dilgen_sa_channel_profile.csv",
    ),
)
USE_PRECURSOR_INLET_PROFILES = True


def create_design_mesh():
    return load_mesh_from_xdmf(mesh_files["MESH_DIRECTORY"], MPI.comm_world)


L = 1.0
H_MAX = 1.0 / 120.0
INLET_HEIGHT = 0.2 * L
INLET_HALF_HEIGHT = 0.5 * INLET_HEIGHT
OUTLET_WIDTH = 0.2 * L
INLET_Y_MIN = 0.7 * L
INLET_Y_MAX = INLET_Y_MIN + INLET_HEIGHT
INLET_Y_CENTER = 0.5 * (INLET_Y_MIN + INLET_Y_MAX)
OUTLET_X_MIN = 0.7 * L
OUTLET_X_MAX = OUTLET_X_MIN + OUTLET_WIDTH

DOMAIN_X_MIN = 0.0
DOMAIN_Y_MIN = 0.0
DOMAIN_X_MAX = L
DOMAIN_Y_MAX = L
TOL = DOLFIN_EPS

RHO_FLUID_VALUE = 1.0
U_BULK_INLET = 5.0
KINEMATIC_VISCOSITY = 5.0e-5
MU_FLUID_VALUE = RHO_FLUID_VALUE * KINEMATIC_VISCOSITY
REYNOLDS_NUMBER = U_BULK_INLET * INLET_HALF_HEIGHT / KINEMATIC_VISCOSITY

# Fallback empirical turbulent channel profile, only used when
# USE_PRECURSOR_INLET_PROFILES = False.
TURBULENT_PROFILE_EXPONENT = 7.0
U_MAX_INLET = U_BULK_INLET * (TURBULENT_PROFILE_EXPONENT + 1.0) / TURBULENT_PROFILE_EXPONENT

# Fallback SA inlet estimate, only used when no tabulated precursor profile is
# requested. The normal Dilgen path uses build_turbulence_inlet_profile_sets().
SA_TURBULENCE_INTENSITY = 0.05
SA_TURBULENCE_LENGTH_SCALE_RATIO = 0.07
SA_REFERENCE_VELOCITY = U_BULK_INLET
SA_REFERENCE_LENGTH = INLET_HALF_HEIGHT
SA_REYNOLDS_NUMBER = REYNOLDS_NUMBER
SA_SMOOTH_ABS_EPS = 1.0e-12
SA_INIT_WALL_DIST_SCALE = 0.05 * L
SA_NU_TILDE_FLOOR = 1.0e-12
SA_EDDY_VISCOSITY_RATIO_CEILING = 100.0

SA_SUPG_STABILIZATION = False
SA_SUPG_TAU_SCALE = 1.0
SA_PSEUDO_TIME_STABILIZATION = False
SA_PSEUDO_DT = 1.0e-4
SA_PSEUDO_TIME_STEPS = 3

SAVE_SA_CLIPPING_DIAGNOSTICS = False

SA_NU_TILDE_PENALTY_ALPHA = 1.0e3
SA_NU_TILDE_PENALTY_ALPHA_SCHEDULE = [1.0e3]
# Dilgen Eq. (12) damps SA nu_tilde with the same Brinkman/RAMP indicator
# chi(gamma) = q(1 - gamma)/(q + gamma) used in the momentum equation.
SA_NU_TILDE_PENALTY_INTERPOLATION = "brinkman"
# Ignored by the Brinkman interpolation mode; kept for configs that use
# the driver's legacy power-law nu_tilde penalty.
SA_NU_TILDE_PENALTY_N = 3.0

SA_WALL_SIGMA = 0.01
SA_WALL_G0 = 20.0
SA_WALL_PENALTY_ALPHA = 1.0e3
SA_WALL_PENALTY_ALPHA_SCHEDULE = [1.0e3]
SA_WALL_PENALTY_INTERPOLATION = "brinkman"
SA_WALL_PENALTY_N = 3.0
SA_WALL_G_FLOOR = 1.0e-8
# Dilgen uses a Poisson-like wall-distance method for SA, not the Yoon
# reciprocal-distance G equation used by the topology-created-wall variants.
SA_WALL_DISTANCE_MODE = "poisson"
# Dilgen Sec. 5.1 verifies sensitivities on an all-fluid bend, so the SA wall
# distance is the geometric wall distance and is not topology-penalized there.
SA_WALL_DENSITY_SOURCE = "passive"
SA_WALL_SOLID_THRESHOLD = 0.10
SA_WALL_DISTANCE_FLOOR = 0.25 * H_MAX

# ================================================================== #
# Sensitivity-verification objective and continuation parameters
#
# The Re = 10,000 bend in Dilgen Sec. 5.1 is a sensitivity-verification
# case, not an optimization case. Dilgen evaluates an all-fluid design with
# q = 0.1, lambda = 1e3 s^-1, and central finite differences with
# Delta h = 1e-6. The local normalized FEniCS filter clips filtered densities
# to [0, 1], so the finite-difference check uses a near-fluid interior point
# gamma = 0.99 to keep the +/- perturbations symmetric.
# ================================================================== #
VOL_FRAC = 0.99
INITIAL_DENSITY_VALUE = 0.99
INITIAL_DENSITY_MATCH_FILTERED_VOLUME = False

LOG_DILGEN_FIG8_COLUMNS = False
DILGEN_OBJECTIVE_REFERENCE_LENGTH = INLET_HALF_HEIGHT
USE_HEAVISIDE_PROJECTION = False
OBJECTIVE_TYPE = "dissipation"
OBJECTIVE_CONVERGENCE_TOL = 0.0
OBJECTIVE_STREAK_TO_STOP = 1000000000

ALPHA_FLUID = 0.0
ALPHA_SOLID = 1.0e3

Q_PENAL_SCHEDULE = [0.1]
BETA_PROJ_SCHEDULE = [1.0]
MOVE_LIMIT_SCHEDULE = [1.0e-12]
MAX_INNER_ITERATIONS_SCHEDULE = [1]
RUN_FINITE_DIFFERENCE_CHECKS = True
FINITE_DIFFERENCE_CHECK_ITERATIONS = (0,)
FINITE_DIFFERENCE_CHECK_STEP = 1.0e-6
FINITE_DIFFERENCE_CHECK_SAMPLES = 5
FINITE_DIFFERENCE_CHECK_FLOW_SOLVER = "ipcs_snes_polish"
FINITE_DIFFERENCE_CHECK_PICARD_FLOW_SOLVER = "ipcs"
# The paper reports CV1-CV4 but does not publish their coordinates. By default
# these are deterministic samples; set FINITE_DIFFERENCE_CHECK_DOF_INDICES to
# force specific cells on a fixed mesh.
FINITE_DIFFERENCE_CHECK_SEED = 13
FINITE_DIFFERENCE_CHECK_CLIP_TO_BOUNDS = False
FINITE_DIFFERENCE_CHECK_UPDATED_TURBULENCE = False
RUN_TAYLOR_SENSITIVITY_CHECKS = True
TAYLOR_CHECK_STEPS = (1.0e-3, 3.0e-4, 1.0e-4, 3.0e-5)
TAYLOR_CHECK_SEED = 29
SENSITIVITY_VERIFICATION_MODE_NAME = "frozen-turbulence"
SENSITIVITY_VERIFICATION_DERIVATIVE_COLUMN = "FrozenTurbulence"
# ================================================================== #

LINEAR_SOLVER = "mumps"             # Solver for [Picard] SA transport and [Adjoint] linear system

FORWARD_FLOW_SOLVER = "ipcs_snes_polish"        # Solver for [Final flow] 
FORWARD_PICARD_FLOW_SOLVER = "ipcs"             # Solver for [Picard] flow 

FORWARD_SNES_WARM_START_WITH_IPCS = True
FORWARD_SNES_IPCS_WARM_START_MODE = "initial"
FORWARD_SNES_STRICT_FINAL_SOLVE = True

FORWARD_SNES_METHOD = "newtonls"
FORWARD_SNES_LINE_SEARCH = "bt"
FORWARD_SNES_LINEAR_SOLVER = "mumps"
FORWARD_SNES_RTOL = 1.0e-7
FORWARD_SNES_ATOL = 1.0e-9
FORWARD_SNES_MAX_ITERS = 220
FORWARD_SNES_ACCEPTED_RESIDUAL_FACTOR = 1.0

FORWARD_SNES_STARTUP_CONVECTION_SCHEDULE = [
    {"convection_weight": 0.00, "max_iters": 80, "atol": 2.0e-7, "accept_norm": 2.0e-4, "accept_nonconverged": True},
    {"convection_weight": 0.20, "max_iters": 100, "atol": 1.5e-7, "accept_norm": 1.5e-4, "accept_nonconverged": True},
    {"convection_weight": 0.40, "max_iters": 120, "atol": 1.0e-7, "accept_norm": 1.0e-4, "accept_nonconverged": True},
    {"convection_weight": 0.60, "max_iters": 140, "atol": 8.0e-8, "accept_norm": 8.0e-5, "accept_nonconverged": True},
    {"convection_weight": 0.80, "max_iters": 160, "atol": 5.0e-8, "accept_norm": 6.0e-5, "accept_nonconverged": True},
    {"convection_weight": 1.00, "max_iters": 180, "atol": 1.0e-8, "accept_norm": 5.0e-5, "accept_nonconverged": True},
]
FORWARD_SNES_CONVECTION_SCHEDULE = [
    {"convection_weight": 0.00, "max_iters": 70,  "atol": 1.0e-8, "accept_norm": 1.0e-5, "accept_nonconverged": True},
    {"convection_weight": 0.50, "max_iters": 100, "atol": 5.0e-9, "accept_norm": 5.0e-6, "accept_nonconverged": True},
    {"convection_weight": 1.00, "max_iters": 220, "atol": 1.0e-9, "rtol": 1.0e-7, "accept_norm": None, "accept_nonconverged": False},
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
]

PICARD_STEPS = 8
TURBULENCE_RELAXATION = 0.35             

FORWARD_IPCS_DT = 5.0e-7
FORWARD_IPCS_MAX_ITERS = 600
FORWARD_IPCS_VELOCITY_RTOL = 1.0e-4
FORWARD_IPCS_PRESSURE_RTOL = 2.0e-3
FORWARD_IPCS_VEL_RELAXATION = 0.05
FORWARD_IPCS_P_RELAXATION = 0.02
FORWARD_IPCS_VEL_SOLVER = "bicgstab"     # Solver for IPCS Tentative Velocity Step + Velocity Update Step
FORWARD_IPCS_VEL_PRECONDITIONER = "ilu"  # Preconditioner for IPCS Tentative Velocity Step + Velocity Update Step
FORWARD_IPCS_P_SOLVER = "bicgstab"       # Solver for IPCS Pressure Correction Step
FORWARD_IPCS_P_PRECONDITIONER = "ilu"    # Preconditioner for IPCS Pressure Correction Step
FORWARD_IPCS_LOG_EVERY = 50
FORWARD_IPCS_MAX_RESTARTS = 2
FORWARD_IPCS_DT_REDUCTION_FACTOR = 0.5
FORWARD_IPCS_RELAXATION_REDUCTION_FACTOR = 0.7
# IPCS is used only for the frozen-SA Picard updates in this config. Accept a
# near-steady Picard iterate and let the final monolithic SNES solve own the
# strict flow state used by the objective and adjoint.
FORWARD_IPCS_PICARD_ACCEPT_BEST_SCORE = 3.0

BETA_PROJ_VALUE = BETA_PROJ_SCHEDULE[0]
ETA_I = 0.50
QUADRATURE_DEGREE = 6
FILTER_BASE_LENGTH = 1.0
FILTER_RADIUS_IN_CELLS = 0.0

OUTLET_BC_TYPE = "pressure"
OUTLET_PRESSURE_VALUE = 0.0
ENABLE_PRESSURE_PIN = False
DILGEN_FROZEN_RESULTS_ROOT_NAME = "Results_Frozen/Results_PipeBendDilgen_Frozen"
DILGEN_SEMIFROZEN_RESULTS_ROOT_NAME = "Results_SemiFrozen/Results_PipeBendDilgen_SemiFrozen_Verification"
RESULTS_ROOT_NAME = DILGEN_FROZEN_RESULTS_ROOT_NAME
SAVE_DILGEN_PAPER_DATA = True
DILGEN_PAPER_GRID_POINTS = 201
DILGEN_PAPER_LINE_POINTS = 401
DILGEN_PAPER_HORIZONTAL_LINE_Y = 0.5 * L

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


_PROFILE_CACHE = None


def _load_precursor_profile():
    global _PROFILE_CACHE
    if _PROFILE_CACHE is not None:
        return _PROFILE_CACHE
    if not os.path.isfile(INLET_PROFILE_CSV):
        raise FileNotFoundError(
            "Missing Dilgen inlet precursor profile '{}'. Generate it with:\n"
            "  python3 PrecursorProfiles/DilgenPipeBendSA/generate_dilgen_sa_channel_profiles.py".format(
                INLET_PROFILE_CSV
            )
        )

    data = np.genfromtxt(INLET_PROFILE_CSV, delimiter=",", names=True, comments="#")
    required_columns = ("y", "u_x", "nu_tilde")
    missing_columns = [name for name in required_columns if name not in data.dtype.names]
    if missing_columns:
        raise ValueError(
            "Dilgen inlet profile '{}' is missing columns: {}".format(
                INLET_PROFILE_CSV,
                ", ".join(missing_columns),
            )
        )

    y_local = np.asarray(data["y"], dtype=float)
    u_x = np.asarray(data["u_x"], dtype=float)
    nu_tilde = np.asarray(data["nu_tilde"], dtype=float)

    order = np.argsort(y_local)
    y_local = y_local[order]
    u_x = u_x[order]
    nu_tilde = nu_tilde[order]

    if y_local[0] > DOLFIN_EPS or abs(y_local[-1] - INLET_HEIGHT) > 1.0e-8:
        raise ValueError(
            "Dilgen inlet profile y-range must be [0, {:.6g}], got [{:.6g}, {:.6g}].".format(
                INLET_HEIGHT,
                float(y_local[0]),
                float(y_local[-1]),
            )
        )

    profile_bulk = float(_trapezoid(u_x, y_local) / INLET_HEIGHT)
    if profile_bulk <= 0.0:
        raise ValueError("Dilgen inlet profile has non-positive bulk velocity.")

    _PROFILE_CACHE = {
        "y": y_local,
        "u_x": u_x * (U_BULK_INLET / profile_bulk),
        "nu_tilde": nu_tilde,
        "bulk_velocity_raw": profile_bulk,
    }
    return _PROFILE_CACHE


class TabulatedDilgenVelocity(UserExpression):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        profile = _load_precursor_profile()
        self.y = profile["y"]
        self.u_x = profile["u_x"]

    def eval(self, values, x):
        y_local = min(max(x[1] - INLET_Y_MIN, 0.0), INLET_HEIGHT)
        values[0] = float(np.interp(y_local, self.y, self.u_x))
        values[1] = 0.0

    def value_shape(self):
        return (2,)


class TabulatedDilgenNuTilde(UserExpression):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        profile = _load_precursor_profile()
        self.y = profile["y"]
        self.nu_tilde = profile["nu_tilde"]

    def eval(self, values, x):
        y_local = min(max(x[1] - INLET_Y_MIN, 0.0), INLET_HEIGHT)
        values[0] = float(np.interp(y_local, self.y, self.nu_tilde))

    def value_shape(self):
        return ()


def _build_power_law_velocity_profile():
    return Expression(
        (
            "fabs((x[1] - y_center) / half_height) <= 1.0 ? "
            "u_centerline * pow(1.0 - fabs((x[1] - y_center) / half_height), 1.0 / exponent) : 0.0",
            "0.0",
        ),
        degree=4,
        u_centerline=U_MAX_INLET,
        y_center=INLET_Y_CENTER,
        half_height=INLET_HALF_HEIGHT,
        exponent=TURBULENT_PROFILE_EXPONENT,
    )


def build_velocity_profile_sets():
    if USE_PRECURSOR_INLET_PROFILES:
        return [TabulatedDilgenVelocity(degree=1)], []
    u_inlet = _build_power_law_velocity_profile()
    return [u_inlet], []


def build_turbulence_inlet_profile_sets():
    if not USE_PRECURSOR_INLET_PROFILES:
        ratio = (
            (0.09 ** 0.25)
            * (1.5 ** 0.5)
            * SA_TURBULENCE_INTENSITY
            * SA_REYNOLDS_NUMBER
            * SA_TURBULENCE_LENGTH_SCALE_RATIO
        )
        return [Constant(nu_tilde_from_viscosity_ratio(ratio, KINEMATIC_VISCOSITY))]
    return [TabulatedDilgenNuTilde(degree=1)]
