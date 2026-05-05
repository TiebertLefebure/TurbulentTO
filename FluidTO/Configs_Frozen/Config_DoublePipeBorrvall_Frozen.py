import os
from dolfin import DOLFIN_EPS, Expression, MeshFunction, MPI, SubDomain, near
from Utilities_SharedTO import build_cell_tag_restriction_functions, load_mesh_from_xdmf


# ===================================================================
# Configuration: Borrvall Double Pipe - Turbulent Frozen
#
# Two symmetric ports on left (inlets) and right (outlets)
# ===================================================================

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Mesh path for this benchmark geometry.
# Generate via: python3 Meshes/generate_borrvall_guided_meshes.py
mesh_files = {
    'MESH_DIRECTORY': os.path.join(REPO_ROOT, 'Meshes/DoublePipeBorrvall/mesh_guided.xdmf'),
    'CELL_DIRECTORY': os.path.join(REPO_ROOT, 'Meshes/DoublePipeBorrvall/cell_guided.xdmf'),
}

DESIGN_DOMAIN_TAG = 1
NON_DESIGN_FLUID_TAG = 2


def create_design_mesh():
    return load_mesh_from_xdmf(mesh_files['MESH_DIRECTORY'], MPI.comm_world)


build_density_bounds, build_volume_region, build_objective_region = build_cell_tag_restriction_functions(
    mesh_files["CELL_DIRECTORY"],
    design_tags=(DESIGN_DOMAIN_TAG,),
    non_design_fluid_tags=(NON_DESIGN_FLUID_TAG,),
)


# Geometry and reference meshing parameters.
L = 1.0
INLET_EXTENSION_WIDTH = 0.2 * L
OUTLET_EXTENSION_WIDTH = 0.2 * L
DESIGN_X_MIN = 0.0
DESIGN_Y_MIN = 0.0
DESIGN_X_MAX = 1.5 * L
DESIGN_Y_MAX = L
DOMAIN_X_MIN = DESIGN_X_MIN - INLET_EXTENSION_WIDTH
DOMAIN_Y_MIN = DESIGN_Y_MIN
DOMAIN_X_MAX = DESIGN_X_MAX + OUTLET_EXTENSION_WIDTH
DOMAIN_Y_MAX = DESIGN_Y_MAX
NX = 150  # reference resolution used to generate the Gmsh mesh
NY = 100
TOL = DOLFIN_EPS

# Port layout on left/right boundaries
PORT_WIDTH = 1.0 / 6.0
PORT_TOP_MARGIN = 1.0 / 6.0
PORT_BOTTOM_MARGIN = 1.0 / 6.0

TOP_PORT_Y_MAX = DOMAIN_Y_MAX - PORT_TOP_MARGIN
TOP_PORT_Y_MIN = TOP_PORT_Y_MAX - PORT_WIDTH
BOTTOM_PORT_Y_MIN = PORT_BOTTOM_MARGIN
BOTTOM_PORT_Y_MAX = BOTTOM_PORT_Y_MIN + PORT_WIDTH

INLET_SEGMENTS = [
    (TOP_PORT_Y_MIN, TOP_PORT_Y_MAX),
    (BOTTOM_PORT_Y_MIN, BOTTOM_PORT_Y_MAX),
]
OUTLET_SEGMENTS = [
    (TOP_PORT_Y_MIN, TOP_PORT_Y_MAX),
    (BOTTOM_PORT_Y_MIN, BOTTOM_PORT_Y_MAX),
]

# Flow and Brinkman parameters for the frozen forward solve.
MU_FLUID_VALUE = 1.0e-4
RHO_FLUID_VALUE = 1.0
U_MAX_INLETS = [1.0, 1.0]
U_MAX_OUTLETS = [1.0, 1.0]

# ================================================================================================
# Reynolds number: Re = U_MAX_INLETS * PORT_WIDTH * RHO_FLUID_VALUE / MU_FLUID_VALUE = 1,600
# ================================================================================================

# SA transport parameters for the frozen turbulence update.
SA_TURBULENCE_INTENSITY = 0.075
SA_TURBULENCE_LENGTH_SCALE_RATIO = 0.06
SA_REFERENCE_LENGTH = PORT_WIDTH
SA_SMOOTH_ABS_EPS = 1.0e-12
SA_INIT_WALL_DIST_SCALE = 0.05 * DOMAIN_Y_MAX
SA_NU_TILDE_FLOOR = 1.0e-12
SA_EDDY_VISCOSITY_RATIO_CEILING = 50.0
SA_NU_TILDE_PENALTY_ALPHA = 1.0e3
SA_NU_TILDE_PENALTY_N = 3.0

# Relaxed wall equation parameters for the external reciprocal distance solve.
SA_WALL_SIGMA = 0.01                    # PDE regularisation length
SA_WALL_G0 = 20.0                       # reference reciprocal distance in solid
SA_WALL_PENALTY_ALPHA = 1.0e3           # penalty amplitude
SA_WALL_PENALTY_N = 3.0                 # penalty exponent
SA_WALL_G_FLOOR = 1.0e-8                # floor on reciprocal distance (avoids division by zero)

# MMA objective and continuation parameters for the topology update.
VOL_FRAC = 1.0 / 3.0          # target fluid volume fraction (two thin channels ≈ 1/3)
OBJECTIVE_CONVERGENCE_TOL = 1.0e-5
OBJECTIVE_STREAK_TO_STOP = 5

# Continuation schedules — one entry per stage, applied in order.
Q_PENAL_SCHEDULE    = [0.05, 0.10, 0.20, 0.50, 1.00, 1.50, 2.00, 3.00, 3.00, 3.00]
MOVE_LIMIT_SCHEDULE = [0.08, 0.06, 0.04, 0.025, 0.015, 0.008, 0.004, 0.002, 0.001, 0.0005]
BETA_PROJ_SCHEDULE  = [0.5,  1.0,  2.0,  4.0,  8.0,  12.0, 16.0, 24.0, 32.0, 64.0]
MAX_INNER_ITERATIONS_SCHEDULE = [50, 80, 80, 80, 100, 120, 140, 160, 180, 200]

# Frozen flow/turbulence coupling and forward solve parameters.
LINEAR_SOLVER = "mumps"

# FORWARD_FLOW_SOLVER = "snes" or FORWARD_FLOW_SOLVER = "ipcs".
# This case keeps the monolithic SNES solve, but starts the first global
# forward solve with a convection ramp so Newton does not jump directly from a
# Stokes guess to the full Re=1600 frozen-viscosity residual.
FORWARD_FLOW_SOLVER = "snes"
FORWARD_SNES_METHOD = "newtonls"
FORWARD_SNES_LINE_SEARCH = "bt"
FORWARD_SNES_RTOL = 1.0e-6
FORWARD_SNES_ATOL = 1.0e-8
FORWARD_SNES_MAX_ITERS = 120

FORWARD_SNES_ERROR_ON_NONCONVERGENCE = True
FORWARD_SNES_ACCEPT_NONCONVERGED_WITH_ACCEPT_NORM = False
FORWARD_SNES_ACCEPTED_RESIDUAL_FACTOR = 1.0
FORWARD_SNES_STOP_AT_ACCEPT_NORM = True
FORWARD_SNES_ACCEPT_NORM_SOLVE_FACTOR = 0.05
FORWARD_SNES_MAX_ACCEPTED_ABSOLUTE_RESIDUAL = 2.0e-3

FORWARD_SNES_ADAPTIVE_CONVECTION = True
FORWARD_SNES_MIN_CONVECTION_STEP = 0.01
FORWARD_SNES_MAX_ADAPTIVE_CONVECTION_STEPS = 24
FORWARD_SNES_STARTUP_CONVECTION_SCHEDULE = [
    {"convection_weight": 0.00, "max_iters": 100, "atol": 1.0e-7, "accept_norm": 1.0e-6, "accept_nonconverged": True},
    {"convection_weight": 0.10, "max_iters": 140, "atol": 8.0e-8, "accept_norm": 1.2e-3, "accept_nonconverged": True},
    {"convection_weight": 0.20, "max_iters": 160, "atol": 6.0e-8, "accept_norm": 1.2e-3, "accept_nonconverged": True},
    {"convection_weight": 0.35, "max_iters": 180, "atol": 5.0e-8, "accept_norm": 1.0e-3, "accept_nonconverged": True},
    {"convection_weight": 0.50, "max_iters": 200, "atol": 4.0e-8, "accept_norm": 9.0e-4, "accept_nonconverged": True},
    {"convection_weight": 0.70, "max_iters": 220, "atol": 3.0e-8, "accept_norm": 8.0e-4, "accept_nonconverged": True},
    {"convection_weight": 0.85, "max_iters": 240, "atol": 2.0e-8, "accept_norm": 6.0e-4, "accept_nonconverged": True},
    {"convection_weight": 1.00, "max_iters": 260, "atol": 1.0e-8, "accept_norm": 5.0e-4, "accept_nonconverged": True},
]
FORWARD_SNES_CONVECTION_SCHEDULE = [
    {"convection_weight": 0.00, "max_iters": 80, "atol": 1.0e-7, "accept_norm": 1.0e-6, "accept_nonconverged": True},
    {"convection_weight": 0.25, "max_iters": 140, "atol": 6.0e-8, "accept_norm": 1.0e-3, "accept_nonconverged": True},
    {"convection_weight": 0.50, "max_iters": 160, "atol": 4.0e-8, "accept_norm": 8.0e-4, "accept_nonconverged": True},
    {"convection_weight": 0.75, "max_iters": 180, "atol": 2.0e-8, "accept_norm": 6.0e-4, "accept_nonconverged": True},
    {"convection_weight": 1.00, "max_iters": 220, "atol": 1.0e-8, "accept_norm": 5.0e-4, "accept_nonconverged": True},
]
FORWARD_SNES_RECOVERY_ATTEMPTS = [
    {
        "label": "current-iterate l2 line-search retry",
        "line_search": "l2",
        "max_iters": 240,
        "restart_with_stokes": False,
        "accept_norm": 7.5e-4,
        "accept_nonconverged": True,
    },
    {
        "label": "current-iterate trust-region retry",
        "method": "newtontr",
        "max_iters": 260,
        "restart_with_stokes": False,
        "accept_norm": 7.5e-4,
        "accept_nonconverged": True,
    },
    {
        "label": "Stokes rebuild backtracking retry",
        "line_search": "bt",
        "max_iters": 280,
        "restart_with_stokes": True,
        "accept_norm": 5.0e-4,
        "accept_nonconverged": True,
    },
    {
        "label": "Stokes rebuild l2 line-search retry",
        "line_search": "l2",
        "max_iters": 300,
        "restart_with_stokes": True,
        "accept_norm": 5.0e-4,
        "accept_nonconverged": True,
    },
]

# Outer NS–SA coupling: solve NS → solve SA → repeat PICARD_STEPS times,
# then one final NS solve with the converged nu_tilde_frozen.
PICARD_STEPS = 3
TURBULENCE_RELAXATION = 0.35   # under-relaxation on the frozen SA update

# IPCS forward solver parameters, used when FORWARD_FLOW_SOLVER = "ipcs":
FORWARD_IPCS_DT                 = 3.13e-6
FORWARD_IPCS_MAX_ITERS          = 600
FORWARD_IPCS_VELOCITY_RTOL      = 1.0e-4
FORWARD_IPCS_PRESSURE_RTOL      = 2.0e-3
FORWARD_IPCS_LOG_EVERY          = 50
FORWARD_IPCS_VEL_RELAXATION     = 0.07
FORWARD_IPCS_P_RELAXATION       = 0.07
FORWARD_IPCS_MAX_RESTARTS       = 4
FORWARD_IPCS_VEL_SOLVER         = "bicgstab"
FORWARD_IPCS_VEL_PRECONDITIONER = "ilu"
FORWARD_IPCS_P_SOLVER           = "bicgstab"
FORWARD_IPCS_P_PRECONDITIONER   = "ilu"
FORWARD_IPCS_DT_REDUCTION_FACTOR = 0.5
FORWARD_IPCS_RELAXATION_REDUCTION_FACTOR = 0.7
# The double-pipe startup often lands in a near-steady IPCS state on stage 1
# before pressure fully meets the strict target; accept that best iterate and
# let later Picard/continuation updates settle the field instead of aborting.
FORWARD_IPCS_ACCEPT_BEST_SCORE  = 2.5

# Projection, boundary-condition, and output settings for the optimization loop.
BETA_PROJ_VALUE = BETA_PROJ_SCHEDULE[0]  # initial projection sharpness (updated per stage)
ETA_I = 0.50           # projection threshold
QUADRATURE_DEGREE = 6  # raised to handle nonlinear terms accurately
FILTER_RADIUS_IN_CELLS = 2.0  # PDE filter radius in mesh cell widths
# FILTER_RADIUS_IN_CELLS = 2.0 for NX = 150 and NY = 100

# Use pressure outlets here; switch to "velocity" only when imposing outlet profiles below.
OUTLET_BC_TYPE = "pressure"
OUTLET_PRESSURE_VALUE = 0.0

ENABLE_PRESSURE_PIN = False
RESULTS_ROOT_NAME = "Results_Frozen/Results_DoublePipeBorrvall_TurbulentTO_Frozen"

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
        left_wall_outside_inlets = near(x[0], self._design_x_min, self._tol) and not self._inside_any_segment(
            x[1], self._inlet_segments
        )
        right_wall_outside_outlets = near(x[0], self._design_x_max, self._tol) and not self._inside_any_segment(
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
            left_wall_outside_inlets
            or right_wall_outside_outlets
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


def _build_uniform_horizontal_profile(u_max):
    return Expression(("u_max", "0.0"), degree=0, u_max=u_max)


def build_velocity_profile_sets():
    inlet_profiles = [
        _build_uniform_horizontal_profile(u_max)
        for u_max in U_MAX_INLETS
    ]
    if OUTLET_BC_TYPE == "pressure":
        return inlet_profiles, []

    outlet_profiles = [
        _build_uniform_horizontal_profile(u_max)
        for u_max in U_MAX_OUTLETS
    ]
    return inlet_profiles, outlet_profiles
