import os
from dolfin import DOLFIN_EPS, Expression, MeshFunction, MPI, SubDomain, near
from Utilities_SharedTO import build_cell_tag_restriction_functions, load_mesh_from_xdmf


# ===================================================================
# Configuration: Yoon 2016 Double Pipe - Turbulent Frozen (SA, Re = 3,000)
#
# This follows Yoon's Fig. 7/Fig. 10 two-inlet/two-outlet pipe setup:
# parabolic velocity at both inlet ports, p = 0 at both outlet ports,
# and nu_tilde = 50 at both inlet and outlet ports.
# ===================================================================

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Generate via: python3 Meshes/DoublePipeYoon/generate_double_pipe_yoon_yplus1.py
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


# Geometry from Yoon Fig. 7.
L1 = 3.0
L2 = 2.0
H1 = 5.0
H2 = 1.0
H3 = 1.0
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

# Flow and Brinkman parameters from Yoon Fig. 7/Fig. 10.
RHO_FLUID_VALUE = 1.0
MU_FLUID_VALUE = 1.0
U_MAX_INLETS = [3000.0, 3000.0]
ALPHA_FLUID = 0.0
ALPHA_SOLID = 1.0e5

# Re = rho * U_MAX * h / mu = 3,000 for the 1 m high ports.
REYNOLDS_NUMBER = 3000.0

# Yoon Fig. 10 imposes nu_tilde = 50 at both input and output ports.
SA_NU_TILDE_INLETS = [50.0, 50.0]
SA_NU_TILDE_OUTLETS = [50.0, 50.0]
SA_NU_TILDE_INITIAL = 50.0
SA_SMOOTH_ABS_EPS = 1.0e-12
SA_INIT_WALL_DIST_SCALE = 0.05 * H1
SA_NU_TILDE_FLOOR = 1.0e-12
SA_NU_TILDE_PENALTY_ALPHA = 1.0e6
SA_NU_TILDE_PENALTY_N = 3.0
SA_NU_TILDE_PENALTY_INTERPOLATION = "power"

SAVE_SA_CLIPPING_DIAGNOSTICS = False

# Yoon relaxed reciprocal wall-distance equation, Eq. (25).
SA_WALL_DISTANCE_MODE = "reciprocal_penalized"
SA_WALL_DENSITY_SOURCE = "design"
SA_WALL_SIGMA = 0.1
SA_WALL_G0 = 20.0
SA_WALL_PENALTY_ALPHA = 1.0e6
SA_WALL_PENALTY_N = 3.0
SA_WALL_PENALTY_INTERPOLATION = "power"
SA_WALL_G_FLOOR = 1.0e-8

# Objective and constraint: Yoon Eq. (32)-(33), 60% solid mass usage.
# This code uses rho = 1 for fluid and rho = 0 for solid, so 60% solid
# corresponds to a maximum fluid volume fraction of 40%.
OBJECTIVE_TYPE = "dissipation"
VOL_FRAC = 0.40
OBJECTIVE_CONVERGENCE_TOL = 1.0e-5
OBJECTIVE_STREAK_TO_STOP = 5

# Yoon Fig. 7 gives n_u = 0.1 for the Brinkman interpolation.

#Q_PENAL_SCHEDULE = [0.10, 0.10, 0.10, 0.10]
#MOVE_LIMIT_SCHEDULE = [0.08, 0.06, 0.04, 0.03]
#MAX_INNER_ITERATIONS_SCHEDULE = [30, 50, 50, 50]

Q_PENAL_SCHEDULE = [0.01, 0.02, 0.04, 0.08]
BETA_PROJ_SCHEDULE = [1.0, 2.0, 4.0, 8.0]
MOVE_LIMIT_SCHEDULE = [0.05, 0.04, 0.03, 0.02]
MAX_INNER_ITERATIONS_SCHEDULE = [50, 60, 80, 100]

# Yoon uses direct element design variables; disable the projective sharpening.
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
FORWARD_SNES_MAX_ITERS = 160
FORWARD_SNES_ERROR_ON_NONCONVERGENCE = False
FORWARD_SNES_ACCEPT_NONCONVERGED_WITH_ACCEPT_NORM = True
FORWARD_SNES_ACCEPT_INITIAL_IF_WITHIN_ACCEPT_NORM = True
FORWARD_SNES_ACCEPTED_RESIDUAL_FACTOR = 1.0
FORWARD_SNES_STOP_AT_ACCEPT_NORM = True
FORWARD_SNES_ACCEPT_NORM_SOLVE_FACTOR = 1.0
FORWARD_SNES_MAX_ACCEPTED_ABSOLUTE_RESIDUAL = 2.0e-3
FORWARD_SNES_ADAPTIVE_CONVECTION = True
FORWARD_SNES_MIN_CONVECTION_STEP = 0.05
FORWARD_SNES_MAX_ADAPTIVE_CONVECTION_STEPS = 10
FORWARD_SNES_STARTUP_CONVECTION_SCHEDULE = [
    {"convection_weight": 0.00, "max_iters": 80, "atol": 1.0e-7, "accept_norm": 2.0e-4, "accept_nonconverged": True},
    {"convection_weight": 0.25, "max_iters": 100, "atol": 8.0e-8, "accept_norm": 4.0e-4, "accept_nonconverged": True},
    {"convection_weight": 0.50, "max_iters": 120, "atol": 5.0e-8, "accept_norm": 8.0e-4, "accept_nonconverged": True},
    {"convection_weight": 0.75, "max_iters": 140, "atol": 2.0e-8, "accept_norm": 1.2e-3, "accept_nonconverged": True},
    {"convection_weight": 1.00, "max_iters": 180, "atol": 1.0e-8, "accept_norm": 1.5e-3, "accept_nonconverged": True},
]
FORWARD_SNES_CONVECTION_SCHEDULE = [
    {"convection_weight": 0.00, "max_iters": 70, "atol": 1.0e-7, "accept_norm": 2.0e-4, "accept_nonconverged": True},
    {"convection_weight": 0.33, "max_iters": 100, "atol": 6.0e-8, "accept_norm": 6.0e-4, "accept_nonconverged": True},
    {"convection_weight": 0.67, "max_iters": 130, "atol": 3.0e-8, "accept_norm": 1.0e-3, "accept_nonconverged": True},
    {"convection_weight": 1.00, "max_iters": 170, "atol": 1.0e-8, "accept_norm": 1.5e-3, "accept_nonconverged": True},
]
FORWARD_SNES_RECOVERY_ATTEMPTS = [
    {
        "label": "l2 line-search retry",
        "line_search": "l2",
        "max_iters": 220,
        "restart_with_stokes": False,
        "accept_norm": 2.0e-3,
        "accept_nonconverged": True,
    },
    {
        "label": "trust-region retry",
        "method": "newtontr",
        "max_iters": 240,
        "restart_with_stokes": False,
        "accept_norm": 2.0e-3,
        "accept_nonconverged": True,
    },
]

# ======================================
PICARD_STEPS = 3
TURBULENCE_RELAXATION = 0.15
# ======================================

FORWARD_IPCS_DT = 5.0e-6
FORWARD_IPCS_MAX_ITERS = 400
FORWARD_IPCS_VELOCITY_RTOL = 1.0e-4
FORWARD_IPCS_PRESSURE_RTOL = 2.0e-3
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

OUTLET_BC_TYPE = "pressure"
OUTLET_PRESSURE_VALUE = 0.0
ENABLE_PRESSURE_PIN = False

RESULTS_ROOT_NAME = "Results_Frozen/Results_DoublePipeYoon_Re3000_NuTilde50_Frozen"

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
        DESIGN_X_MIN, DESIGN_X_MAX, DOMAIN_X_MIN, DOMAIN_X_MAX, DOMAIN_Y_MIN, DOMAIN_Y_MAX,
        TOL, INLET_SEGMENTS, OUTLET_SEGMENTS,
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
