import os
from dolfin import DOLFIN_EPS, Expression, MeshFunction, MPI, SubDomain, near
from Utilities_SharedTO import build_cell_tag_restriction_functions, load_mesh_from_xdmf


# ===================================================================
# Configuration: Yoon 2016 Diffuser - Turbulent Full (SA, Re = 3,000)
#
# This follows Yoon's Fig. 20-22 diffuser setup: a 1 m x 1 m design
# domain, parabolic inlet and outlet velocity profiles, SA nu_tilde
# prescribed at both ports, and the coupled reciprocal wall-distance
# state G included in the adjoint system.
# ===================================================================

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Generate via: python3 Meshes/DiffuserYoon/generate_diffuser_yoon_yplus1.py
mesh_files = {
    "MESH_DIRECTORY": os.path.join(REPO_ROOT, "Meshes/DiffuserYoon/mesh_yoon_yplus1.xdmf"),
    "CELL_DIRECTORY": os.path.join(REPO_ROOT, "Meshes/DiffuserYoon/cell_yoon_yplus1.xdmf"),
    "FACET_DIRECTORY": os.path.join(REPO_ROOT, "Meshes/DiffuserYoon/facet_yoon_yplus1.xdmf"),
}

DESIGN_DOMAIN_TAG = 1


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
OBJECTIVE_TYPE = "dissipation"
VOL_FRAC = 0.30
INITIAL_DENSITY_VALUE = 1.0
INITIAL_DENSITY_MATCH_FILTERED_VOLUME = False
OBJECTIVE_CONVERGENCE_TOL = 1e-5
OBJECTIVE_STREAK_TO_STOP = 5

# Yoon Fig. 20 gives n_u = 0.01-0.1 for the Brinkman interpolation.
Q_PENAL_SCHEDULE = [0.01, 0.02, 0.05, 0.10]
MOVE_LIMIT_SCHEDULE = [0.08, 0.06, 0.04, 0.03]
MAX_INNER_ITERATIONS_SCHEDULE = [100, 120, 160, 220]

# Yoon uses direct element design variables; disable the projective sharpening.
USE_HEAVISIDE_PROJECTION = False
BETA_PROJ_SCHEDULE = [1.0, 1.0, 1.0, 1.0]
BETA_PROJ_VALUE = BETA_PROJ_SCHEDULE[0]
ETA_I = 0.50
FILTER_RADIUS_IN_CELLS = 0.0
QUADRATURE_DEGREE = 6

# Full primal-state (u, p, nu_tilde, G) solve parameters.
LINEAR_SOLVER = "mumps"
STATE_SOLVE_METHOD = "newtontr"
STATE_LINE_SEARCH = "bt"
STATE_RTOL = 1.0e-6
STATE_ATOL = 1.0e-8
STATE_MAX_ITERS = 160
STATE_ACCEPT_NONCONVERGED_WITH_ACCEPT_NORM = True
STATE_INITIAL_SA_SWEEPS = 4
STATE_INITIAL_SA_RELAXATION = 0.25
STATE_TURBULENCE_COUPLING_SCHEDULE = [
    {"convection_weight": 0.0, "weight": 0.0, "max_iters": 180, "atol": 8.0e-4, "accept_norm": 5.0e-2},
    {"convection_weight": 0.50, "weight": 0.0, "max_iters": 200, "atol": 6.0e-4, "accept_norm": 2.0e-2, "accept_growth": 1.25},
    {"convection_weight": 1.0, "weight": 0.0, "max_iters": 220, "atol": 5.0e-4, "accept_norm": 1.0e-2, "accept_growth": 1.25},
    {"convection_weight": 1.0, "weight": 0.35, "max_iters": 220, "atol": 3.0e-4, "accept_norm": 7.5e-3, "accept_growth": 1.25},
    {"convection_weight": 1.0, "weight": 0.70, "max_iters": 240, "atol": 2.0e-4, "accept_norm": 5.0e-3, "accept_growth": 1.25},
    {"convection_weight": 1.0, "weight": 1.0, "max_iters": 300},
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

OUTLET_BC_TYPE = "velocity"
ENABLE_PRESSURE_PIN = True
PRESSURE_PIN_POINT = (DOMAIN_X_MIN, DOMAIN_Y_MIN)

RESULTS_ROOT_NAME = "Results_Full/Results_DiffuserYoon_Full"

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
