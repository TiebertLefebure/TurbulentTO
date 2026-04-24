import os
from dolfin import DOLFIN_EPS, Expression, MeshFunction, MPI, SubDomain, near
from Utilities_SharedTO import load_mesh_from_xdmf


# ===================================================================
# Configuration: Borrvall Diffuser - Turbulent Full (SA, Re = 1,000)
#
# One inlet on the left wall and one outlet on the right wall.
# This config targets the Full adjoint: the reciprocal wall-distance G enters
# the monolithic primal state and the adjoint system together with
# (u, p, nu_tilde).
# ===================================================================

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Mesh path for this benchmark geometry.
# Generate via: python3 Meshes/generate_borrvall_guided_meshes.py
mesh_files = {
    'MESH_DIRECTORY': os.path.join(REPO_ROOT, 'Meshes/DiffuserBorrvall/mesh_guided.xdmf'),
}


def create_design_mesh():
    return load_mesh_from_xdmf(mesh_files['MESH_DIRECTORY'], MPI.comm_world)


# Geometry and reference meshing parameters.
L = 1.0
N = 120  # reference resolution used to generate the Gmsh mesh (LC = L/N)
INLET_EXTENSION_WIDTH = 0.2 * L
OUTLET_EXTENSION_WIDTH = 0.2 * L
DESIGN_X_MIN = 0.0
DESIGN_X_MAX = L
DOMAIN_X_MIN = DESIGN_X_MIN - INLET_EXTENSION_WIDTH
DOMAIN_Y_MIN = 0.0
DOMAIN_X_MAX = DESIGN_X_MAX + OUTLET_EXTENSION_WIDTH
DOMAIN_Y_MAX = L
TOL = DOLFIN_EPS

# Diffuser opening on right boundary
OUTLET_Y_MIN = 1.0 / 3.0
OUTLET_Y_MAX = 2.0 / 3.0

# Flow and Brinkman parameters for the monolithic primal solve.
MU_FLUID_VALUE = 1.0e-3
RHO_FLUID_VALUE = 1.0
U_MAX_INLET = 1.0
U_MAX_OUTLET = 3.0

# ===================================================================================
# Reynolds number: Re = U_MAX_INLET * L * RHO_FLUID_VALUE / MU_FLUID_VALUE = 1,000
# ===================================================================================

# SA transport parameters for the monolithic primal state.
SA_TURBULENCE_INTENSITY = 0.075
SA_TURBULENCE_LENGTH_SCALE_RATIO = 0.10
SA_REFERENCE_VELOCITY = U_MAX_INLET
SA_REFERENCE_LENGTH = L
SA_SMOOTH_ABS_EPS = 1.0e-12
SA_INIT_WALL_DIST_SCALE = 0.05 * L
SA_NU_TILDE_FLOOR = 1.0e-12
SA_NU_TILDE_PENALTY_ALPHA = 1.0e3
SA_NU_TILDE_PENALTY_N = 3.0

# Relaxed wall equation parameters for the coupled reciprocal distance state.
SA_WALL_SIGMA = 0.01
SA_WALL_G0 = 20.0
SA_WALL_PENALTY_ALPHA = 1.0e3
SA_WALL_PENALTY_N = 3.0
SA_WALL_G_FLOOR = 1.0e-8

# MMA objective and continuation parameters for the topology update.
VOL_FRAC = 0.50
OBJECTIVE_CONVERGENCE_TOL = 1e-5
OBJECTIVE_STREAK_TO_STOP = 5

# The raw diffuser design can become nearly binary while the filtered/projected
# physical density remains gray. Keep q fixed at its sharpest setting in the
# final stages, then raise beta and shrink the MMA move limit so the projected
# field can collapse toward 0/1 instead of stalling around the threshold.
Q_PENAL_SCHEDULE = [0.1, 0.1, 0.2, 0.4, 0.8, 1.5, 3.0, 3.0, 3.0, 3.0]
MOVE_LIMIT_SCHEDULE = [0.08, 0.06, 0.04, 0.03, 0.02, 0.015, 0.01, 0.005, 0.003, 0.0015]
BETA_PROJ_SCHEDULE = [0.1, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0]
MAX_INNER_ITERATIONS_SCHEDULE = [80, 80, 100, 120, 120, 140, 140, 160, 180, 220]

# Full primal-state (u, p, nu_tilde) solve parameters.
LINEAR_SOLVER = "mumps"
STATE_SOLVE_METHOD = "newtonls"
STATE_LINE_SEARCH = "bt"
STATE_RTOL = 1.0e-6
STATE_ATOL = 1.0e-8
STATE_MAX_ITERS = 50
STATE_INITIAL_SA_SWEEPS = 4
# The diffuser usually solves cleanly, so keep this continuation light. The
# goal is only to soften occasional startup failures without importing the much
# heavier double-pipe rescue schedule.
STATE_TURBULENCE_COUPLING_SCHEDULE = [
    {"weight": 0.0, "max_iters": 120, "atol": 8.0e-4},
    {"weight": 0.35, "max_iters": 100, "atol": 5.0e-4},
    {"weight": 0.70, "max_iters": 100, "atol": 2.0e-4},
    {"weight": 1.0},
]
STATE_RECOVERY_ATTEMPTS = [
    {
        "label": "current-iterate line-search retry",
        "method": "newtonls",
        "line_search": "bt",
        "max_iters": 120,
        "restart_with_stokes": False,
    },
    {
        "label": "Stokes rebuild line-search retry",
        "method": "newtonls",
        "line_search": "bt",
        "max_iters": 160,
        "restart_with_stokes": True,
    },
]

# Projection, boundary-condition, and output settings for the optimization loop.
BETA_PROJ_VALUE = BETA_PROJ_SCHEDULE[0]
ETA_I = 0.50
QUADRATURE_DEGREE = 6
FILTER_RADIUS_IN_CELLS = 2.0

OUTLET_BC_TYPE = "pressure"
OUTLET_PRESSURE_VALUE = 0.0

ENABLE_PRESSURE_PIN = False
RESULTS_ROOT_NAME = "Results_Full/Results_DiffuserBorrvall_TurbulentTO_Full"

MARK = {"generic": 0, "walls": 1, "inlet": 2, "outlet": 3}


def between(value, limits, eps=DOLFIN_EPS):
    return (limits[0] - eps <= value) and (value <= limits[1] + eps)


class WallsBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and (
            near(x[1], DOMAIN_Y_MIN, TOL)
            or near(x[1], DOMAIN_Y_MAX, TOL)
            or (near(x[0], DESIGN_X_MAX, TOL) and between(x[1], (DOMAIN_Y_MIN, OUTLET_Y_MIN), TOL))
            or (near(x[0], DESIGN_X_MAX, TOL) and between(x[1], (OUTLET_Y_MAX, DOMAIN_Y_MAX), TOL))
            or (near(x[0], DOMAIN_X_MAX, TOL) and between(x[1], (DOMAIN_Y_MIN, OUTLET_Y_MIN), TOL))
            or (near(x[0], DOMAIN_X_MAX, TOL) and between(x[1], (OUTLET_Y_MAX, DOMAIN_Y_MAX), TOL))
            or (near(x[1], OUTLET_Y_MIN, TOL) and between(x[0], (DESIGN_X_MAX, DOMAIN_X_MAX), TOL))
            or (near(x[1], OUTLET_Y_MAX, TOL) and between(x[0], (DESIGN_X_MAX, DOMAIN_X_MAX), TOL))
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


def build_density_bounds(mesh, density_space):
    non_design_fluid_lower = Expression(
        "(x[0] < x_min || x[0] > x_max) ? 1.0 : 0.0",
        degree=0,
        x_min=DESIGN_X_MIN,
        x_max=DESIGN_X_MAX,
    )
    return non_design_fluid_lower, 1.0


def build_volume_region(mesh, density_space):
    return Expression(
        "(x[0] >= x_min && x[0] <= x_max) ? 1.0 : 0.0",
        degree=0,
        x_min=DESIGN_X_MIN,
        x_max=DESIGN_X_MAX,
    )


def build_velocity_profile_sets():
    u_inlet = Expression(
        ("u_max", "0.0"),
        degree=0,
        u_max=U_MAX_INLET,
    )

    if OUTLET_BC_TYPE == "pressure":
        return [u_inlet], []

    u_outlet = Expression(
        ("u_max", "0.0"),
        degree=0,
        u_max=U_MAX_OUTLET,
    )

    return [u_inlet], [u_outlet]
