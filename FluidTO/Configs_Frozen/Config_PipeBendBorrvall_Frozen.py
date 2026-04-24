import os
from math import pi
from dolfin import DOLFIN_EPS, Expression, MeshFunction, MPI, SubDomain, near
from Utilities_SharedTO import load_mesh_from_xdmf


# ===================================================================
# Configuration: Borrvall Pipe Bend - Turbulent Frozen (SA, Re = 2,000)
#
# One inlet on the left wall and one outlet on the bottom wall.
# ===================================================================

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Mesh path for this benchmark geometry.
# Generate via: python3 Meshes/generate_borrvall_guided_meshes.py
mesh_files = {
    'MESH_DIRECTORY': os.path.join(REPO_ROOT, 'Meshes/PipeBendBorrvall/mesh_guided.xdmf'),
}


def create_design_mesh():
    return load_mesh_from_xdmf(mesh_files['MESH_DIRECTORY'], MPI.comm_world)


# Geometry and reference meshing parameters.
L = 1.0
N = 120  # reference resolution used to generate the Gmsh mesh (LC = L/N)
INLET_EXTENSION_WIDTH = 0.2 * L
OUTLET_EXTENSION_HEIGHT = 0.2 * L
DESIGN_X_MIN = 0.0
DESIGN_Y_MIN = 0.0
DESIGN_X_MAX = L
DESIGN_Y_MAX = L
DOMAIN_X_MIN = DESIGN_X_MIN - INLET_EXTENSION_WIDTH
DOMAIN_Y_MIN = DESIGN_Y_MIN - OUTLET_EXTENSION_HEIGHT
DOMAIN_X_MAX = DESIGN_X_MAX
DOMAIN_Y_MAX = DESIGN_Y_MAX
TOL = DOLFIN_EPS

# Geometry parameters
INLET_HEIGHT = 0.2 * L
INLET_TOP_OFFSET = 0.2 * L
OUTLET_WIDTH = 0.2 * L
OUTLET_RIGHT_OFFSET = 0.2 * L

# Flow and Brinkman parameters for the frozen forward solve.
MU_FLUID_VALUE = 1.0e-4
RHO_FLUID_VALUE = 1.0
U_MAX_INLET = 1.0
U_MAX_OUTLET = 1.0

# ==============================================================================================
# Reynolds number: Re = U_MAX_INLET * INLET_HEIGHT * RHO_FLUID_VALUE / MU_FLUID_VALUE = 2,000
# ==============================================================================================

# SA transport parameters for the frozen turbulence update.
SA_TURBULENCE_INTENSITY = 0.075
SA_TURBULENCE_LENGTH_SCALE_RATIO = 0.05
SA_REFERENCE_LENGTH = INLET_HEIGHT
SA_SMOOTH_ABS_EPS = 1.0e-12
SA_INIT_WALL_DIST_SCALE = 0.05 * L
SA_NU_TILDE_FLOOR = 1.0e-12
SA_NU_TILDE_PENALTY_ALPHA = 1.0e3
SA_NU_TILDE_PENALTY_N = 3.0

# Relaxed wall equation parameters for the external reciprocal distance solve.
SA_WALL_SIGMA = 0.01                    # PDE regularisation length
SA_WALL_G0 = 20.0                       # reference reciprocal distance in solid
SA_WALL_PENALTY_ALPHA = 1.0e3           # penalty amplitude
SA_WALL_PENALTY_N = 3.0                 # penalty exponent
SA_WALL_G_FLOOR = 1.0e-8               # floor on reciprocal distance (avoids division by zero)

# MMA objective and continuation parameters for the topology update.
VOL_FRAC = 0.08 * pi  # Borrvall pipe-bend benchmark volume fraction
OBJECTIVE_CONVERGENCE_TOL = 1e-5
OBJECTIVE_STREAK_TO_STOP = 5

# -------------------------------------------------------------------
# Continuation schedule for the frozen-SA pipe bend
# -------------------------------------------------------------------
Q_PENAL_SCHEDULE    = [0.05, 0.10, 0.20, 0.40, 0.80, 1.50, 2.50, 3.00, 3.00]
MOVE_LIMIT_SCHEDULE = [0.08, 0.07, 0.06, 0.045, 0.03, 0.02, 0.01, 0.005, 0.002]
BETA_PROJ_SCHEDULE  = [0.10, 0.25, 0.50, 1.00, 2.00, 4.00, 8.00, 16.00, 24.00]
MAX_INNER_ITERATIONS_SCHEDULE = [60, 70, 80, 90, 90, 90, 100, 100, 100]

# Frozen flow/turbulence coupling and IPCS solve parameters.
LINEAR_SOLVER = "mumps"   # direct LU solver for the Stokes warm start and the adjoint

# Outer NS–SA coupling: solve NS → solve SA → repeat PICARD_STEPS times,
# then one final NS solve with the converged nu_tilde_frozen.
PICARD_STEPS = 3
TURBULENCE_RELAXATION = 0.35   # under-relaxation on the frozen SA update

# IPCS forward solver parameters:
#   dt                        : pseudo-time step (smaller → more stable, more iterations needed)
#   u_relaxation/p_relaxation : under-relaxation (lower → more stable at high Re, slower convergence)
#   rtol_u                    : ||Δu||/||u|| convergence threshold; tighter → smaller R_NS → better adjoint
FORWARD_IPCS_DT = 2.5e-5
FORWARD_IPCS_MAX_ITERS = 150
FORWARD_IPCS_VELOCITY_RTOL = 1.0e-4
FORWARD_IPCS_PRESSURE_RTOL = 2.0e-3
FORWARD_IPCS_VEL_RELAXATION = 0.21
FORWARD_IPCS_P_RELAXATION = 0.07
FORWARD_IPCS_VEL_SOLVER = "bicgstab"
FORWARD_IPCS_VEL_PRECONDITIONER = "ilu"
FORWARD_IPCS_P_SOLVER = "cg"
FORWARD_IPCS_P_PRECONDITIONER = "ilu"
FORWARD_IPCS_LOG_EVERY = 50
FORWARD_IPCS_MAX_RESTARTS = 3
FORWARD_IPCS_DT_REDUCTION_FACTOR = 0.5
FORWARD_IPCS_RELAXATION_REDUCTION_FACTOR = 0.7

# Projection, boundary-condition, and output settings for the optimization loop.
BETA_PROJ_VALUE = BETA_PROJ_SCHEDULE[0]  # initial projection sharpness (updated per stage)
ETA_I = 0.50
QUADRATURE_DEGREE = 6  # raised to handle nonlinear terms accurately
FILTER_RADIUS_IN_CELLS = 3.0  # PDE filter radius in mesh cell widths

# Use pressure outlets here; switch to "velocity" only when imposing the outlet profile below.
OUTLET_BC_TYPE = "pressure"
OUTLET_PRESSURE_VALUE = 0.0  # Used only when OUTLET_BC_TYPE == "pressure".

ENABLE_PRESSURE_PIN = False
RESULTS_ROOT_NAME = "Results_Frozen/Results_PipeBendBorrvall_TurbulentTO_Frozen"

MARK = {"generic": 0, "walls": 1, "inlet": 2, "outlet": 3}


def between(value, limits, eps=DOLFIN_EPS):
    return (limits[0] - eps <= value) and (value <= limits[1] + eps)


def compute_port_extents(l_box, inlet_top_offset, inlet_height, outlet_right_offset, outlet_width):
    inlet_y_max = l_box - inlet_top_offset
    inlet_y_min = inlet_y_max - inlet_height
    outlet_x_max = l_box - outlet_right_offset
    outlet_x_min = outlet_x_max - outlet_width
    return inlet_y_min, inlet_y_max, outlet_x_min, outlet_x_max


class InletBoundary(SubDomain):
    def __init__(self, x_location, tol, inlet_y_min, inlet_y_max):
        super().__init__()
        self._x_location = x_location
        self._tol = tol
        self._inlet_y_min = inlet_y_min
        self._inlet_y_max = inlet_y_max

    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], self._x_location, self._tol) and between(
            x[1], (self._inlet_y_min, self._inlet_y_max), self._tol
        )


class OutletBoundary(SubDomain):
    def __init__(self, y_location, tol, outlet_x_min, outlet_x_max):
        super().__init__()
        self._y_location = y_location
        self._tol = tol
        self._outlet_x_min = outlet_x_min
        self._outlet_x_max = outlet_x_max

    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], self._y_location, self._tol) and between(
            x[0], (self._outlet_x_min, self._outlet_x_max), self._tol
        )


class WallsBoundary(SubDomain):
    def __init__(
        self,
        design_x_min,
        design_y_min,
        design_x_max,
        design_y_max,
        domain_x_min,
        domain_y_min,
        tol,
        inlet_y_min,
        inlet_y_max,
        outlet_x_min,
        outlet_x_max,
    ):
        super().__init__()
        self._design_x_min = design_x_min
        self._design_y_min = design_y_min
        self._design_x_max = design_x_max
        self._design_y_max = design_y_max
        self._domain_x_min = domain_x_min
        self._domain_y_min = domain_y_min
        self._tol = tol
        self._inlet_y_min = inlet_y_min
        self._inlet_y_max = inlet_y_max
        self._outlet_x_min = outlet_x_min
        self._outlet_x_max = outlet_x_max

    def inside(self, x, on_boundary):
        left_wall_outside_inlet = near(x[0], self._design_x_min, self._tol) and not between(
            x[1], (self._inlet_y_min, self._inlet_y_max), self._tol
        )
        right_wall = near(x[0], self._design_x_max, self._tol)
        top_wall = near(x[1], self._design_y_max, self._tol)
        bottom_wall_outside_outlet = near(x[1], self._design_y_min, self._tol) and not between(
            x[0], (self._outlet_x_min, self._outlet_x_max), self._tol
        )
        inlet_strip_caps = (
            (near(x[1], self._inlet_y_min, self._tol) or near(x[1], self._inlet_y_max, self._tol))
            and between(x[0], (self._domain_x_min, self._design_x_min), self._tol)
        )
        outlet_strip_caps = (
            (near(x[0], self._outlet_x_min, self._tol) or near(x[0], self._outlet_x_max, self._tol))
            and between(x[1], (self._domain_y_min, self._design_y_min), self._tol)
        )
        return on_boundary and (
            left_wall_outside_inlet
            or right_wall
            or top_wall
            or bottom_wall_outside_outlet
            or inlet_strip_caps
            or outlet_strip_caps
        )


def mark_boundaries(mesh):
    inlet_y_min, inlet_y_max, outlet_x_min, outlet_x_max = compute_port_extents(
        L, INLET_TOP_OFFSET, INLET_HEIGHT, OUTLET_RIGHT_OFFSET, OUTLET_WIDTH
    )
    boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    boundaries.set_all(MARK["generic"])
    WallsBoundary(
        DESIGN_X_MIN,
        DESIGN_Y_MIN,
        DESIGN_X_MAX,
        DESIGN_Y_MAX,
        DOMAIN_X_MIN,
        DOMAIN_Y_MIN,
        TOL,
        inlet_y_min,
        inlet_y_max,
        outlet_x_min,
        outlet_x_max,
    ).mark(boundaries, MARK["walls"])
    InletBoundary(DOMAIN_X_MIN, TOL, inlet_y_min, inlet_y_max).mark(boundaries, MARK["inlet"])
    OutletBoundary(DOMAIN_Y_MIN, TOL, outlet_x_min, outlet_x_max).mark(boundaries, MARK["outlet"])
    return boundaries


def build_velocity_profile_sets():
    inlet_y_min, inlet_y_max, outlet_x_min, outlet_x_max = compute_port_extents(
        L, INLET_TOP_OFFSET, INLET_HEIGHT, OUTLET_RIGHT_OFFSET, OUTLET_WIDTH
    )
    u_inlet = Expression(
        ("u_max", "0.0"),
        degree=0,
        u_max=U_MAX_INLET,
    )
    if OUTLET_BC_TYPE == "pressure":
        return [u_inlet], []

    u_outlet = Expression(
        ("0.0", "-u_max"),
        degree=0,
        u_max=U_MAX_OUTLET,
    )
    return [u_inlet], [u_outlet]


def build_density_bounds(mesh, density_space):
    non_design_fluid_lower = Expression(
        "(x[0] < x_min || x[1] < y_min || x[0] > x_max || x[1] > y_max) ? 1.0 : 0.0",
        degree=0,
        x_min=DESIGN_X_MIN,
        x_max=DESIGN_X_MAX,
        y_min=DESIGN_Y_MIN,
        y_max=DESIGN_Y_MAX,
    )
    return non_design_fluid_lower, 1.0


def build_volume_region(mesh, density_space):
    return Expression(
        "(x[0] >= x_min && x[0] <= x_max && x[1] >= y_min && x[1] <= y_max) ? 1.0 : 0.0",
        degree=0,
        x_min=DESIGN_X_MIN,
        x_max=DESIGN_X_MAX,
        y_min=DESIGN_Y_MIN,
        y_max=DESIGN_Y_MAX,
    )
