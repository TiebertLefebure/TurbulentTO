import os
from dolfin import DOLFIN_EPS, Expression, Mesh, MeshFunction, MPI, SubDomain, XDMFFile, near
from Utilities_LaminarTO import load_mesh_from_xdmf


# ===================================================================
# Configuration: Borrvall Double Pipe — Turbulent (SA, Re = 1660)
#
# Two symmetric ports on left (inlets) and right (outlets).
# Optimal topology: two straight horizontal channels (no cross-flow).
# ===================================================================

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# Mesh files
# Generate via: cd Meshes/DoublePipeBorrvall && python3 double_pipe_gmsh.py && python3 gmsh_to_xdmf.py
mesh_files = {
    'MESH_DIRECTORY': os.path.join(THIS_DIR, 'Meshes/DoublePipeBorrvall/mesh.xdmf'),
}


def create_design_mesh():
    return load_mesh_from_xdmf(mesh_files['MESH_DIRECTORY'], MPI.comm_world)


# Domain and mesh
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

# Flow settings
MU_FLUID_VALUE = 1.0e-4
RHO_FLUID_VALUE = 1.0
U_MAX_INLETS = [1.0, 1.0]
U_MAX_OUTLETS = [1.0, 1.0]

# ----------------------------------------------------------------------------------------------
# Reynolds number: Re = U_MAX_INLET * PORT_WIDTH * RHO_FLUID_VALUE / MU_FLUID_VALUE = 1,660
# ----------------------------------------------------------------------------------------------

# -------------------------------------------------------------------
# Spalart-Allmaras turbulence model settings
# -------------------------------------------------------------------
# SA_MUT_RATIO: target turbulent viscosity ratio nu_t / nu_lam at inlets.
# The main solver inverts the SA constitutive relation nu_t = nu_tilde * fv1(chi)
# to get the corresponding nu_tilde BC.  At Re=1660 with fully developed turbulent
# channel flow, a ratio of ~5-10 is physically reasonable.
SA_MUT_RATIO = 5.0                      # nu_t / nu_lam at inlets (same for both ports)
SA_DISTANCE_RELAXATION = 0.01           # Helmholtz relaxation for wall-distance PDE
SA_SMOOTH_ABS_EPS = 1.0e-12            # smoothing for |nu_tilde| in chi computation
SA_INIT_WALL_DIST_SCALE = 0.05 * DOMAIN_Y_MAX  # scale for initial nu_tilde ramp from walls
SA_NU_TILDE_FLOOR = 1.0e-12            # hard floor to prevent negative nu_tilde
SA_NU_TILDE_PENALTY_ALPHA = 1.0e3      # penalty strength for nu_tilde in solid regions
SA_NU_TILDE_PENALTY_N = 3.0            # penalty exponent (matches Brinkman)

# Penalized reciprocal wall-distance (Yoon 2016, Eq. 25)
# Modifies wall distance in solid regions so SA sees a nearby "wall" there.
SA_USE_PENALIZED_WALL_DISTANCE = True
SA_WALL_SIGMA = SA_DISTANCE_RELAXATION  # PDE regularisation length
SA_WALL_G0 = 20.0                       # reference reciprocal distance in solid
SA_WALL_PENALTY_ALPHA = 1.0e3           # penalty amplitude
SA_WALL_PENALTY_N = 3.0                 # penalty exponent
SA_WALL_G_FLOOR = 1.0e-8               # floor on reciprocal distance (avoids division by zero)

# -------------------------------------------------------------------
# Topology optimization settings
# -------------------------------------------------------------------
VOL_FRAC = 1.0 / 3.0          # target fluid volume fraction (two thin channels ≈ 1/3)
INITIAL_DENSITY_VALUE = 1.0   # start from connected fluid and let MMA remove material
MAX_INNER_ITERATIONS_SCHEDULE = [35, 80, 100, 120, 120, 120, 100, 100]
OBJECTIVE_CONVERGENCE_TOL = 5e-5
OBJECTIVE_STREAK_TO_STOP = 5

# -------------------------------------------------------------------
# Continuation schedules — one entry per stage, applied in order.
# q_penal: low → convex alpha (gray-friendly); high → penalises intermediate densities.
# beta:    Heaviside sharpness; 1 = smooth sigmoid, 64 = near step function.
# move:    MMA move limit; large at start (topology formation), small at end (refinement).
# -------------------------------------------------------------------
Q_PENAL_SCHEDULE    = [0.05, 0.1, 0.1, 0.2, 0.5, 1.0, 1.0, 1.0]
MOVE_LIMIT_SCHEDULE = [0.08, 0.06, 0.04, 0.02, 0.01, 0.005, 0.003, 0.002]
BETA_PROJ_SCHEDULE  = [0.5,  1.0,  2.0,  4.0, 8.0,  16.0, 32.0, 64.0]

# -------------------------------------------------------------------
# Solver settings
# -------------------------------------------------------------------
SNES_LINEAR_SOLVER = "mumps"   # direct LU solver (adjoint + Stokes warm-start)

# Outer NS–SA coupling: solve NS → solve SA → repeat FROZEN_PICARD_STEPS times,
# then one final NS solve with the converged nu_tilde_frozen.
FROZEN_PICARD_STEPS = 2
NUT_RELAXATION_FACTOR = 0.35   # under-relaxation on SA nu_tilde update

# IPCS forward solver parameters:
#   dt                        : pseudo-time step (smaller → more stable, more iterations needed)
#   u_relaxation/p_relaxation : under-relaxation (lower → more stable at high Re, slower convergence)
#   rtol_u                    : ||Δu||/||u|| convergence threshold; tighter → smaller R_NS → better adjoint
FORWARD_IPCS_DT                 = 2.0e-4
FORWARD_IPCS_MAX_ITERS          = 400
FORWARD_IPCS_RTOL               = 3.0e-4
FORWARD_IPCS_PRESSURE_RTOL      = 1.0e-3
FORWARD_IPCS_U_RELAXATION       = 0.4
FORWARD_IPCS_P_RELAXATION       = 0.15
FORWARD_IPCS_VEL_SOLVER         = "bicgstab"
FORWARD_IPCS_VEL_PRECONDITIONER = "ilu"
FORWARD_IPCS_P_SOLVER           = "cg"
FORWARD_IPCS_P_PRECONDITIONER   = "ilu"
FORWARD_IPCS_MAX_RESTARTS       = 2
FORWARD_IPCS_DT_REDUCTION_FACTOR = 0.5
FORWARD_IPCS_RELAXATION_REDUCTION_FACTOR = 0.7
FORWARD_IPCS_RESTART_WITH_STOKES = False

# -------------------------------------------------------------------
# Projection, filter, and output
# -------------------------------------------------------------------
BETA_PROJ_VALUE = BETA_PROJ_SCHEDULE[0]  # initial projection sharpness (updated per stage)
ETA_I = 0.50           # projection threshold
QUADRATURE_DEGREE = 6  # raised to handle nonlinear terms accurately
FILTER_RADIUS_IN_CELLS = 2.0  # PDE filter radius in mesh cell widths

# Outlet BC toggle for the turbulent double-pipe case:
# Change this to "pressure" to impose p = OUTLET_PRESSURE_VALUE on both outlet markers.
# Keep "velocity" to impose the two outlet parabolic velocity profiles built below.
OUTLET_BC_TYPE = "pressure"
OUTLET_PRESSURE_VALUE = 0.0

ENABLE_PRESSURE_PIN = True
PRESSURE_PIN_POINT = (DOMAIN_X_MIN, DOMAIN_Y_MIN)
RESULTS_ROOT_NAME = "Results_DoublePipeBorrvall_TurbulentTO"

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
