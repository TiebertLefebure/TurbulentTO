import os

import numpy as np

from dolfin import Constant, DOLFIN_EPS, Expression, MeshFunction, MPI, SubDomain, UserExpression, near

from Utilities_SharedTO import build_cell_tag_restriction_functions, load_mesh_from_xdmf
from Utilities_TurbulentTO_Frozen import nu_tilde_from_viscosity_ratio

try:
    _trapezoid = np.trapezoid
except AttributeError:
    _trapezoid = np.trapz


# ===================================================================
# Configuration: Dilgen 2018 2D U-bend - Turbulent Frozen (SA)
#
# Dilgen's Sec. 6.1 U-bend is a topology-optimization case, not the
# all-fluid sensitivity-verification bend from Sec. 5.1. The geometry is
# scaled with the half inlet channel height H = 0.1 m: a 10H x 10H design
# box, 2H inlet/outlet extensions, 2H port heights, 2H x 2.5H fixed
# solid lead blocks, a 1H-thick fixed solid separator, and a 5H separator
# reach into the design domain. The inlet uses fully developed turbulent
# channel profiles.
# ===================================================================

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

mesh_files = {
    "MESH_DIRECTORY": os.path.join(REPO_ROOT, "Meshes/UBendDilgen/mesh_yplus1.xdmf"),
    "CELL_DIRECTORY": os.path.join(REPO_ROOT, "Meshes/UBendDilgen/cell_yplus1.xdmf"),
}

INLET_PROFILE_CSV = os.environ.get(
    "DILGEN_UBEND_INLET_PROFILE_CSV",
    os.path.join(
        REPO_ROOT,
        "PrecursorProfiles/DilgenUBendSA/dilgen_sa_channel_profile.csv",
    ),
)
USE_PRECURSOR_INLET_PROFILES = True

DESIGN_DOMAIN_TAG = 1
NON_DESIGN_FLUID_TAG = 2
NON_DESIGN_SOLID_TAG = 3


def create_design_mesh():
    return load_mesh_from_xdmf(mesh_files["MESH_DIRECTORY"], MPI.comm_world)


build_density_bounds, build_volume_region, build_objective_region = build_cell_tag_restriction_functions(
    mesh_files["CELL_DIRECTORY"],
    design_tags=(DESIGN_DOMAIN_TAG,),
    non_design_fluid_tags=(NON_DESIGN_FLUID_TAG,),
    non_design_solid_tags=(NON_DESIGN_SOLID_TAG,),
)


H = 0.1
L = 10.0 * H
H_MAX = 1.0 / 120.0
LEAD_LENGTH = 2.0 * H
PORT_HEIGHT = 2.0 * H
INLET_HALF_HEIGHT = H

TOP_PORT_Y_MIN = 5.5 * H
TOP_PORT_Y_MAX = TOP_PORT_Y_MIN + PORT_HEIGHT
BOTTOM_PORT_Y_MIN = 2.5 * H
BOTTOM_PORT_Y_MAX = BOTTOM_PORT_Y_MIN + PORT_HEIGHT

BAR_THICKNESS = H
BAR_RADIUS = 0.5 * BAR_THICKNESS
BAR_X_START = -LEAD_LENGTH
BAR_TIP_X = 5.0 * H
BAR_TOTAL_LENGTH = BAR_TIP_X - BAR_X_START
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

RHO_FLUID_VALUE = 1.0
U_BULK_INLET = 2.0
KINEMATIC_VISCOSITY = 4.0e-5
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

SA_NU_TILDE_PENALTY_ALPHA = 2.0e3
SA_NU_TILDE_PENALTY_ALPHA_SCHEDULE = [2.0e3] * 10
# Dilgen Eq. (12) damps SA nu_tilde with the same Brinkman/RAMP indicator
# chi(gamma) = q(1 - gamma)/(q + gamma) used in the momentum equation.
SA_NU_TILDE_PENALTY_INTERPOLATION = "brinkman"
# Ignored by the Brinkman interpolation; retained for the legacy
# power-law mode.
SA_NU_TILDE_PENALTY_N = 3.0

SA_WALL_SIGMA = 0.01
SA_WALL_G0 = 20.0
SA_WALL_PENALTY_ALPHA = 2.0e3
SA_WALL_PENALTY_ALPHA_SCHEDULE = [2.0e3] * 10
SA_WALL_PENALTY_INTERPOLATION = "brinkman"
# Ignored by the Brinkman Poisson wall-distance mode; retained for the
# legacy reciprocal-distance mode.
SA_WALL_PENALTY_N = 3.0
SA_WALL_G_FLOOR = 1.0e-8
# Dilgen uses a Poisson-like wall-distance equation, penalized in porous
# material with the same Brinkman chi(gamma) form as the SA nu_tilde equation.
SA_WALL_DISTANCE_MODE = "poisson_penalized"
SA_WALL_DENSITY_SOURCE = "design"
SA_WALL_DISTANCE_FLOOR = 0.25 * H_MAX
SA_WALL_DISTANCE_SOLVE_FLOOR = 0.0

# ================================================================== #
# MMA Objective and Continuation Parameters
#
# Dilgen Sec. 6.1 uses Re = 5,000, lambda = 2e3 s^-1, q = 0.1,
# filter radius r = 0.01, beta = 1.5 increased by a factor 1.5 every
# 50 cycles with beta capped at 14, 500 total cycles, and f = 0.30.
# ================================================================== #
VOL_FRAC = 0.30
INITIAL_DENSITY_VALUE = 1.0 # VOL_FRAC
INITIAL_DENSITY_MATCH_FILTERED_VOLUME = False #True

LOG_DILGEN_FIG8_COLUMNS = True
DILGEN_OBJECTIVE_REFERENCE_LENGTH = H
OBJECTIVE_TYPE = "dissipation"
OBJECTIVE_CONVERGENCE_TOL = 0.0
OBJECTIVE_STREAK_TO_STOP = 1000000000

ALPHA_FLUID = 0.0
ALPHA_SOLID = 2.0e3

Q_PENAL_SCHEDULE = [0.1] * 10
BETA_PROJ_SCHEDULE = [1.5, 2.25, 3.375, 5.0625, 7.59375, 11.390625, 14.0, 14.0, 14.0, 14.0]
# Dilgen does not report the MMA move limit.
MOVE_LIMIT_SCHEDULE = [0.05] * 10
MAX_INNER_ITERATIONS_SCHEDULE = [50] * 10
RUN_FINITE_DIFFERENCE_CHECKS = False
# ================================================================== #

LINEAR_SOLVER = "mumps"

# Final flow solver: "snes", "ipcs", or "ipcs_snes_polish".
# Picard flow solves may use the cheaper "ipcs" path independently.
FORWARD_FLOW_SOLVER = "ipcs"
FORWARD_PICARD_FLOW_SOLVER = "ipcs"

FORWARD_SNES_WARM_START_WITH_IPCS = True
FORWARD_SNES_IPCS_WARM_START_MODE = "final"
FORWARD_SNES_STRICT_FINAL_SOLVE = False

FORWARD_SNES_METHOD = "newtonls"
FORWARD_SNES_LINE_SEARCH = "bt"
FORWARD_SNES_LINEAR_SOLVER = "mumps"
FORWARD_SNES_RTOL = 1.0e-6
FORWARD_SNES_ATOL = 1.0e-8
FORWARD_SNES_MAX_ITERS = 260

FORWARD_SNES_ACCEPTED_RESIDUAL_FACTOR = 1.0

FORWARD_SNES_ADAPTIVE_CONVECTION = True
FORWARD_SNES_MIN_CONVECTION_STEP = 0.03
FORWARD_SNES_MAX_ADAPTIVE_CONVECTION_STEPS = 16

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

PICARD_STEPS = 3
TURBULENCE_RELAXATION = 0.20

FORWARD_IPCS_DT = 2.5e-6
FORWARD_IPCS_MAX_ITERS = 300
FORWARD_IPCS_VELOCITY_RTOL = 2.0e-4 #1.0e-4
FORWARD_IPCS_PRESSURE_RTOL = 2.0e-3
FORWARD_IPCS_VEL_RELAXATION = 0.07
FORWARD_IPCS_P_RELAXATION = 0.02
FORWARD_IPCS_VEL_SOLVER = "bicgstab"
FORWARD_IPCS_VEL_PRECONDITIONER = "ilu"
FORWARD_IPCS_P_SOLVER = "bicgstab"
FORWARD_IPCS_P_PRECONDITIONER = "ilu"
FORWARD_IPCS_LOG_EVERY = 50
FORWARD_IPCS_MAX_RESTARTS = 3
FORWARD_IPCS_DT_REDUCTION_FACTOR = 0.5
FORWARD_IPCS_RELAXATION_REDUCTION_FACTOR = 0.7

FORWARD_IPCS_PICARD_ACCEPT_BEST_SCORE = 3.0
FORWARD_IPCS_ERROR_ON_NONCONVERGENCE = True

BETA_PROJ_VALUE = BETA_PROJ_SCHEDULE[0]
ETA_I = 0.50
QUADRATURE_DEGREE = 6
FILTER_BASE_LENGTH = 1.0
FILTER_RADIUS_IN_CELLS = 0.01

OUTLET_BC_TYPE = "pressure"
OUTLET_PRESSURE_VALUE = 0.0
ENABLE_PRESSURE_PIN = False
RESULTS_ROOT_NAME = "Results_Frozen/Results_UBendDilgen_Frozen"

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


_PROFILE_CACHE = None


def _load_precursor_profile():
    global _PROFILE_CACHE
    if _PROFILE_CACHE is not None:
        return _PROFILE_CACHE
    if not os.path.isfile(INLET_PROFILE_CSV):
        raise FileNotFoundError(
            "Missing Dilgen U-bend inlet precursor profile '{}'. Generate it with:\n"
            "  python3 PrecursorProfiles/DilgenPipeBendSA/generate_dilgen_sa_channel_profiles.py "
            "--bulk-velocity 2.0 --half-height 0.1 --kinematic-viscosity 4.0e-5 "
            "--output PrecursorProfiles/DilgenUBendSA/dilgen_sa_channel_profile.csv".format(
                INLET_PROFILE_CSV
            )
        )

    data = np.genfromtxt(INLET_PROFILE_CSV, delimiter=",", names=True, comments="#")
    required_columns = ("y", "u_x", "nu_tilde")
    missing_columns = [name for name in required_columns if name not in data.dtype.names]
    if missing_columns:
        raise ValueError(
            "Dilgen U-bend inlet profile '{}' is missing columns: {}".format(
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

    if y_local[0] > DOLFIN_EPS or abs(y_local[-1] - PORT_HEIGHT) > 1.0e-8:
        raise ValueError(
            "Dilgen U-bend inlet profile y-range must be [0, {:.6g}], got [{:.6g}, {:.6g}].".format(
                PORT_HEIGHT,
                float(y_local[0]),
                float(y_local[-1]),
            )
        )

    profile_bulk = float(_trapezoid(u_x, y_local) / PORT_HEIGHT)
    if profile_bulk <= 0.0:
        raise ValueError("Dilgen U-bend inlet profile has non-positive bulk velocity.")

    _PROFILE_CACHE = {
        "y": y_local,
        "u_x": u_x * (U_BULK_INLET / profile_bulk),
        "nu_tilde": nu_tilde,
        "bulk_velocity_raw": profile_bulk,
    }
    return _PROFILE_CACHE


class TabulatedDilgenUBendVelocity(UserExpression):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        profile = _load_precursor_profile()
        self.y = profile["y"]
        self.u_x = profile["u_x"]

    def eval(self, values, x):
        y_local = min(max(x[1] - TOP_PORT_Y_MIN, 0.0), PORT_HEIGHT)
        values[0] = float(np.interp(y_local, self.y, self.u_x))
        values[1] = 0.0

    def value_shape(self):
        return (2,)


class TabulatedDilgenUBendNuTilde(UserExpression):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        profile = _load_precursor_profile()
        self.y = profile["y"]
        self.nu_tilde = profile["nu_tilde"]

    def eval(self, values, x):
        y_local = min(max(x[1] - TOP_PORT_Y_MIN, 0.0), PORT_HEIGHT)
        values[0] = float(np.interp(y_local, self.y, self.nu_tilde))

    def value_shape(self):
        return ()


def _build_power_law_velocity_profile():
    y_center = 0.5 * (TOP_PORT_Y_MIN + TOP_PORT_Y_MAX)
    return Expression(
        (
            "fabs((x[1] - y_center) / half_height) <= 1.0 ? "
            "u_centerline * pow(1.0 - fabs((x[1] - y_center) / half_height), 1.0 / exponent) : 0.0",
            "0.0",
        ),
        degree=4,
        u_centerline=U_MAX_INLET,
        y_center=y_center,
        half_height=INLET_HALF_HEIGHT,
        exponent=TURBULENT_PROFILE_EXPONENT,
    )


def build_velocity_profile_sets():
    if USE_PRECURSOR_INLET_PROFILES:
        return [TabulatedDilgenUBendVelocity(degree=1)], []
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
    return [TabulatedDilgenUBendNuTilde(degree=1)]
