import os
from dolfin import DOLFIN_EPS, Expression, MeshFunction, MPI, SubDomain, near
from Utilities_SharedTO import build_cell_tag_restriction_functions, load_mesh_from_xdmf


# ===================================================================
# Configuration: Yoon 2016 Diffuser - Turbulent Frozen (SA, Re = 3,000)
#
# This follows Yoon's Fig. 20-22 diffuser setup, but uses the frozen
# adjoint approximation: the adjoint state is only (u*, p*), while the
# converged reciprocal wall distance G and SA working variable nu_tilde
# are held fixed during the adjoint solve.
# ===================================================================

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Generate via: python3 Meshes/DiffuserYoon/generate_diffuser_yoon_yplus1.py
mesh_files = {
    "MESH_DIRECTORY": os.path.join(REPO_ROOT, "Meshes/DiffuserYoon/mesh_yoon_yplus1.xdmf"),
    "CELL_DIRECTORY": os.path.join(REPO_ROOT, "Meshes/DiffuserYoon/cell_yoon_yplus1.xdmf"),
    "FACET_DIRECTORY": os.path.join(REPO_ROOT, "Meshes/DiffuserYoon/facet_yoon_yplus1.xdmf"),
}

DESIGN_DOMAIN_TAG = 1


RESUME_OPTIMIZATION = False


def create_design_mesh():
    return load_mesh_from_xdmf(mesh_files["MESH_DIRECTORY"], MPI.comm_world)


build_density_bounds, build_volume_region, build_objective_region = build_cell_tag_restriction_functions(
    mesh_files["CELL_DIRECTORY"],
    design_tags=(DESIGN_DOMAIN_TAG,),
)


# Geometry.
L = 1.0
N = 180
DESIGN_X_MIN = 0.0
DESIGN_X_MAX = L
DOMAIN_X_MIN = 0.0
DOMAIN_Y_MIN = 0.0
DOMAIN_X_MAX = L
DOMAIN_Y_MAX = L
OUTLET_Y_MIN = L / 3.0
OUTLET_Y_MAX = 2.0 * L / 3.0
TOL = DOLFIN_EPS

# Flow and Brinkman parameters from Fig. 20.
RHO_FLUID_VALUE = 1000.0
MU_FLUID_VALUE = 1.0
U_MAX_INLET = 3.0
U_MAX_OUTLET = 3.0 * U_MAX_INLET
ALPHA_FLUID = 0.0
ALPHA_SOLID = 1.0e9

# Re = rho * U_MAX_INLET * L / mu = 3,000.
REYNOLDS_NUMBER = 3000.0

# Yoon prescribes nu_tilde = 1 at both input and output ports in Fig. 22.
SA_NU_TILDE_INLET = 1.0
SA_NU_TILDE_OUTLET = 1.0
SA_NU_TILDE_INITIAL = 1.0
SA_SMOOTH_ABS_EPS = 1.0e-12
SA_INIT_WALL_DIST_SCALE = 0.05 * L
SA_NU_TILDE_FLOOR = 1.0e-12
SA_NU_TILDE_PENALTY_ALPHA = 1.0e5
SA_NU_TILDE_PENALTY_N = 3.0
SA_NU_TILDE_PENALTY_INTERPOLATION = "power"

SAVE_SA_CLIPPING_DIAGNOSTICS = False

# Yoon relaxed reciprocal wall-distance equation, Eq. (25).
SA_WALL_DISTANCE_MODE = "reciprocal_penalized"
SA_WALL_DENSITY_SOURCE = "design"
SA_WALL_SIGMA = 0.1
SA_WALL_G0 = 20.0
SA_WALL_PENALTY_ALPHA = 1.0e5
SA_WALL_PENALTY_N = 3.0
SA_WALL_PENALTY_INTERPOLATION = "power"
SA_WALL_G_FLOOR = 1.0e-8

# Objective and constraint: Yoon Eq. (32)-(33), 70% solid mass usage.
# This code uses rho = 1 for fluid and rho = 0 for solid, so 70% solid
# corresponds to a maximum fluid volume fraction of 30%.
OBJECTIVE_TYPE = "dissipation"
VOL_FRAC = 0.30
OBJECTIVE_CONVERGENCE_TOL = 1e-6
OBJECTIVE_STREAK_TO_STOP = 10

# Final-stage reference values from the laminar Yoon diffuser TO run. These are
# reporting thresholds only; they do not drive MMA or convergence.
FINAL_DESIGN_PRESSURE_DROP_TARGET_PA = 1.1831e4
FINAL_DESIGN_VISCOUS_DISSIPATION_TARGET_W_PER_M = 3.4299e4
REPORT_FINAL_METRIC_TARGETS = True

# Yoon Fig. 20 gives n_u = 0.01-0.1 for the Brinkman interpolation.

#Q_PENAL_SCHEDULE = [0.01, 0.02, 0.05, 0.10]
#MOVE_LIMIT_SCHEDULE = [0.04, 0.03, 0.02, 0.015]
#MAX_INNER_ITERATIONS_SCHEDULE = [45, 40, 50, 70]

# Keep the first four stages identical to older runs. A checkpoint saved after
# the old stage 4 will resume into the q=0.08 polishing hold below, then proceed
# into progressively stronger Brinkman/projection continuation.
Q_PENAL_SCHEDULE = [
    0.01, 0.02, 0.04, 0.08,
    0.08, 0.12, 0.20, 0.35,
    ]
BETA_PROJ_SCHEDULE = [
    1.0, 2.0, 4.0, 8.0,
    8.0, 12.0, 16.0, 32.0,
]
MOVE_LIMIT_SCHEDULE = [
    0.05, 0.04, 0.03, 0.02,
    0.012, 0.010, 0.0075, 0.005,
]
MAX_INNER_ITERATIONS_SCHEDULE = [
    50, 60, 80, 100,
    200, 180, 220, 260,
]

# Thesis-standard solver path: use the same projection/filter continuation as
# the Alexandersen pipe-bend case, even though Yoon used direct element
# design variables without filtering.
USE_HEAVISIDE_PROJECTION = True
BETA_PROJ_VALUE = BETA_PROJ_SCHEDULE[0]
ETA_I = 0.50
FILTER_RADIUS_IN_CELLS = 4.0
QUADRATURE_DEGREE = 6

# Frozen forward/adjoint solve parameters.
LINEAR_SOLVER = "mumps"

# Use IPCS for the Picard-coupled updates and SNES for the final steady
# flow residual that is differentiated by the frozen adjoint.
FORWARD_FLOW_SOLVER = "snes"
FORWARD_PICARD_FLOW_SOLVER = "ipcs"

FORWARD_SNES_METHOD = "newtonls"
FORWARD_SNES_LINE_SEARCH = "bt"
FORWARD_SNES_RTOL = 1.0e-6
FORWARD_SNES_ATOL = 1.0e-8
FORWARD_SNES_MAX_ITERS = 120
FORWARD_SNES_ERROR_ON_NONCONVERGENCE = False
FORWARD_SNES_ACCEPT_NONCONVERGED_WITH_ACCEPT_NORM = True
FORWARD_SNES_ACCEPT_INITIAL_IF_WITHIN_ACCEPT_NORM = True
FORWARD_SNES_ACCEPTED_RESIDUAL_FACTOR = 1.0


FORWARD_SNES_STOP_AT_ACCEPT_NORM = True


FORWARD_SNES_ACCEPT_NORM_SOLVE_FACTOR = 1.0
FORWARD_SNES_MAX_ACCEPTED_ABSOLUTE_RESIDUAL = 1.0e-3
FORWARD_SNES_ADAPTIVE_CONVECTION = True
FORWARD_SNES_MIN_CONVECTION_STEP = 0.10
FORWARD_SNES_MAX_ADAPTIVE_CONVECTION_STEPS = 8
FORWARD_SNES_STARTUP_CONVECTION_SCHEDULE = [
    {"convection_weight": 0.00, "max_iters": 80, "atol": 1.0e-7, "accept_norm": 2.0e-4, "accept_nonconverged": True},
    {"convection_weight": 0.50, "max_iters": 100, "atol": 5.0e-8, "accept_norm": 1.0e-4, "accept_nonconverged": True},
    {"convection_weight": 1.00, "max_iters": 140, "atol": 1.0e-8, "accept_norm": 5.0e-5, "accept_nonconverged": True},
]
FORWARD_SNES_CONVECTION_SCHEDULE = [
    {"convection_weight": 0.00, "max_iters": 60, "atol": 1.0e-7, "accept_norm": 1.5e-4, "accept_nonconverged": True},
    {"convection_weight": 0.50, "max_iters": 80, "atol": 5.0e-8, "accept_norm": 8.0e-5, "accept_nonconverged": True},
    {"convection_weight": 1.00, "max_iters": 120, "atol": 1.0e-8, "accept_norm": 5.0e-5, "accept_nonconverged": True},
]
FORWARD_SNES_RECOVERY_ATTEMPTS = [
    {
        "label": "l2 line-search retry",
        "line_search": "l2",
        "max_iters": 160,
        "restart_with_stokes": False,
        "accept_norm": 7.5e-5,
        "accept_nonconverged": True,
    },
    {
        "label": "trust-region retry",
        "method": "newtontr",
        "max_iters": 180,
        "restart_with_stokes": False,
        "accept_norm": 7.5e-5,
        "accept_nonconverged": True,
    },
]

# ============================================

PICARD_STEPS = 3
TURBULENCE_RELAXATION = 0.15

FORWARD_IPCS_DT = 5.0e-6 
FORWARD_IPCS_MAX_ITERS = 400 
FORWARD_IPCS_VELOCITY_RTOL = 2.0e-4 
FORWARD_IPCS_PRESSURE_RTOL = 1.0e-3 
FORWARD_IPCS_VEL_RELAXATION = 0.10
FORWARD_IPCS_P_RELAXATION = 0.03
FORWARD_IPCS_MAX_RESTARTS = 3 
FORWARD_IPCS_DT_REDUCTION_FACTOR = 0.5
FORWARD_IPCS_RELAXATION_REDUCTION_FACTOR = 0.7
FORWARD_IPCS_VEL_SOLVER = "bicgstab"
FORWARD_IPCS_VEL_PRECONDITIONER = "ilu"
FORWARD_IPCS_P_SOLVER = "bicgstab"
FORWARD_IPCS_P_PRECONDITIONER = "ilu"
FORWARD_IPCS_LOG_EVERY = 50
FORWARD_IPCS_ACCEPT_BEST_SCORE = 1.00

# ============================================

OUTLET_BC_TYPE = "pressure" # "velocity" or "pressure"
ENABLE_PRESSURE_PIN = True
PRESSURE_PIN_POINT = (DOMAIN_X_MIN, DOMAIN_Y_MIN)

RESULTS_ROOT_BASE_NAME = "Results_Frozen/Results_DiffuserYoon_Frozen"
if OUTLET_BC_TYPE == "velocity":
    RESULTS_ROOT_NAME = RESULTS_ROOT_BASE_NAME + "_VelocityOutlet"
elif OUTLET_BC_TYPE == "pressure":
    RESULTS_ROOT_NAME = RESULTS_ROOT_BASE_NAME + "_PressureOutlet"
else:
    RESULTS_ROOT_NAME = RESULTS_ROOT_BASE_NAME

MARK = {"generic": 0, "walls": 1, "inlet": 2, "outlet": 3}


def between(value, limits, eps=DOLFIN_EPS):
    return (limits[0] - eps <= value) and (value <= limits[1] + eps)


class WallsBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and (
            near(x[1], DOMAIN_Y_MIN, TOL)
            or near(x[1], DOMAIN_Y_MAX, TOL)
            or (
                near(x[0], DOMAIN_X_MAX, TOL)
                and (
                    between(x[1], (DOMAIN_Y_MIN, OUTLET_Y_MIN), TOL)
                    or between(x[1], (OUTLET_Y_MAX, DOMAIN_Y_MAX), TOL)
                )
            )
        )


class InletBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], DOMAIN_X_MIN, TOL) and between(
            x[1], (DOMAIN_Y_MIN, DOMAIN_Y_MAX), TOL
        )


class OutletBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], DOMAIN_X_MAX, TOL) and between(
            x[1], (OUTLET_Y_MIN, OUTLET_Y_MAX), TOL
        )


def mark_boundaries(mesh):
    boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    boundaries.set_all(MARK["generic"])
    WallsBoundary().mark(boundaries, MARK["walls"])
    InletBoundary().mark(boundaries, MARK["inlet"])
    OutletBoundary().mark(boundaries, MARK["outlet"])
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
    u_inlet = _parabolic_x_profile(DOMAIN_Y_MIN, DOMAIN_Y_MAX, U_MAX_INLET)
    u_outlet = _parabolic_x_profile(OUTLET_Y_MIN, OUTLET_Y_MAX, U_MAX_OUTLET)
    return [u_inlet], [u_outlet]
