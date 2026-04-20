import os
from dolfin import DOLFIN_EPS, Expression, MeshFunction, MPI, SubDomain, near
from Utilities_SharedTO import load_mesh_from_xdmf


# ===================================================================
# Configuration: Borrvall Double Pipe - Turbulent Frozen (SA, Re = 1,660)
#
# Two symmetric ports on left (inlets) and right (outlets).
# ===================================================================

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Mesh path for this benchmark geometry.
# Generate via: cd Meshes/DoublePipeBorrvall && python3 double_pipe_gmsh.py && python3 gmsh_to_xdmf.py
mesh_files = {
    'MESH_DIRECTORY': os.path.join(REPO_ROOT, 'Meshes/DoublePipeBorrvall/mesh.xdmf'),
}


def create_design_mesh():
    return load_mesh_from_xdmf(mesh_files['MESH_DIRECTORY'], MPI.comm_world)


# Geometry and reference meshing parameters for the two-port design box.
DOMAIN_X_MIN = 0.0
DOMAIN_Y_MIN = 0.0
DOMAIN_X_MAX = 1.5
DOMAIN_Y_MAX = 1.0
NX = 150  # reference resolution used to generate the Gmsh mesh
NY = 100
TOL = DOLFIN_EPS

# Port layout on left/right boundaries
PORT_WIDTH = 1.0 / 6.0
PORT_TOP_MARGIN = 1.0 / 4.0
PORT_BOTTOM_MARGIN = 1.0 / 4.0

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
MU_FLUID_VALUE = 1.0 / 6.0
RHO_FLUID_VALUE = 1.0
U_MAX_INLETS = [1.0, 1.0]
U_MAX_OUTLETS = [1.0, 1.0]

# SA transport parameters for the frozen turbulence update.
SA_MUT_RATIO = 5.0                      # nu_t / nu_lam at inlets (same for both ports)
SA_SMOOTH_ABS_EPS = 1.0e-12             # smoothing for |nu_tilde| in chi computation
SA_INIT_WALL_DIST_SCALE = 0.05 * DOMAIN_Y_MAX  # scale for initial nu_tilde ramp from walls
SA_NU_TILDE_FLOOR = 1.0e-12             # hard floor to prevent negative nu_tilde
SA_NU_TILDE_PENALTY_ALPHA = 1.0e3       # penalty strength for nu_tilde in solid regions
SA_NU_TILDE_PENALTY_N = 3.0             # penalty exponent (matches Brinkman)

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

# -------------------------------------------------------------------
# Continuation schedules — one entry per stage, applied in order.
# -------------------------------------------------------------------
Q_PENAL_SCHEDULE    = [0.05, 0.10, 0.20, 0.50, 1.00, 1.50, 2.00, 3.00, 3.00, 3.00]
MOVE_LIMIT_SCHEDULE = [0.08, 0.06, 0.04, 0.025, 0.015, 0.008, 0.004, 0.002, 0.001, 0.0005]
BETA_PROJ_SCHEDULE  = [0.5,  1.0,  2.0,  4.0,  8.0,  16.0, 32.0, 64.0, 128.0, 256.0]
MAX_INNER_ITERATIONS_SCHEDULE = [50, 80, 80, 80, 100, 120, 140, 160, 180, 200]

# Frozen flow/turbulence coupling and IPCS solve parameters.
LINEAR_SOLVER = "mumps"   # direct LU solver for the Stokes warm start and the adjoint

# Outer NS–SA coupling: solve NS → solve SA → repeat PICARD_STEPS times,
# then one final NS solve with the converged nu_tilde_frozen.
PICARD_STEPS = 1
TURBULENCE_RELAXATION = 0.35   # under-relaxation on the frozen SA update

# IPCS forward solver parameters:
#   dt                        : pseudo-time step (smaller → more stable, more iterations needed)
#   vel_relaxation/p_relaxation : under-relaxation (lower → more stable at high Re, slower convergence)
#   velocity_rtol             : ||Δu||/||u|| convergence threshold; tighter → smaller R_NS → better adjoint
# Start closer to the previously successful retry settings so the first attempt
# does not burn the full 400-step budget before adaptive backoff kicks in.
FORWARD_IPCS_DT                 = 2.5e-5
FORWARD_IPCS_MAX_ITERS          = 150
FORWARD_IPCS_VELOCITY_RTOL      = 1.0e-4
FORWARD_IPCS_PRESSURE_RTOL      = 2.0e-3
FORWARD_IPCS_LOG_EVERY          = 50
FORWARD_IPCS_VEL_RELAXATION     = 0.14
FORWARD_IPCS_P_RELAXATION       = 0.05
FORWARD_IPCS_MAX_RESTARTS       = 3
FORWARD_IPCS_VEL_SOLVER         = "bicgstab"
FORWARD_IPCS_VEL_PRECONDITIONER = "ilu"
FORWARD_IPCS_P_SOLVER           = "cg"
FORWARD_IPCS_P_PRECONDITIONER   = "ilu"
FORWARD_IPCS_DT_REDUCTION_FACTOR = 0.5
FORWARD_IPCS_RELAXATION_REDUCTION_FACTOR = 0.7

# Projection, boundary-condition, and output settings for the optimization loop.
BETA_PROJ_VALUE = BETA_PROJ_SCHEDULE[0]  # initial projection sharpness (updated per stage)
ETA_I = 0.50           # projection threshold
QUADRATURE_DEGREE = 6  # raised to handle nonlinear terms accurately
FILTER_RADIUS_IN_CELLS = 2.0  # PDE filter radius in mesh cell widths

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
    def __init__(self, x_min, x_max, y_min, y_max, tol, inlet_segments, outlet_segments):
        super().__init__()
        self._x_min = x_min
        self._x_max = x_max
        self._y_min = y_min
        self._y_max = y_max
        self._tol = tol
        self._inlet_segments = inlet_segments
        self._outlet_segments = outlet_segments

    def _inside_any_segment(self, y_value, segments):
        return any(between(y_value, segment, self._tol) for segment in segments)

    def inside(self, x, on_boundary):
        left_wall_outside_inlets = near(x[0], self._x_min, self._tol) and not self._inside_any_segment(
            x[1], self._inlet_segments
        )
        right_wall_outside_outlets = near(x[0], self._x_max, self._tol) and not self._inside_any_segment(
            x[1], self._outlet_segments
        )
        top_wall = near(x[1], self._y_max, self._tol)
        bottom_wall = near(x[1], self._y_min, self._tol)
        return on_boundary and (
            left_wall_outside_inlets or right_wall_outside_outlets or top_wall or bottom_wall
        )


def mark_boundaries(mesh):
    boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    boundaries.set_all(MARK["generic"])

    WallsBoundary(
        DOMAIN_X_MIN, DOMAIN_X_MAX, DOMAIN_Y_MIN, DOMAIN_Y_MAX,
        TOL, INLET_SEGMENTS, OUTLET_SEGMENTS,
    ).mark(boundaries, MARK["walls"])

    for marker, segment in zip(MARK["inlet"], INLET_SEGMENTS):
        VerticalPortBoundary(DOMAIN_X_MIN, TOL, segment[0], segment[1]).mark(boundaries, marker)
    for marker, segment in zip(MARK["outlet"], OUTLET_SEGMENTS):
        VerticalPortBoundary(DOMAIN_X_MAX, TOL, segment[0], segment[1]).mark(boundaries, marker)
    return boundaries


def _build_horizontal_profile(u_max, y_min, y_max):
    y_center = 0.5 * (y_min + y_max)
    width = y_max - y_min
    return Expression(
        ("u_max * (1 - pow(2.0 * (x[1] - y_c) / width, 2))", "0.0"),
        degree=2, u_max=u_max, y_c=y_center, width=width,
    )


def build_velocity_profile_sets():
    inlet_profiles = [
        _build_horizontal_profile(u_max, segment[0], segment[1])
        for u_max, segment in zip(U_MAX_INLETS, INLET_SEGMENTS)
    ]
    if OUTLET_BC_TYPE == "pressure":
        return inlet_profiles, []

    outlet_profiles = [
        _build_horizontal_profile(u_max, segment[0], segment[1])
        for u_max, segment in zip(U_MAX_OUTLETS, OUTLET_SEGMENTS)
    ]
    return inlet_profiles, outlet_profiles
