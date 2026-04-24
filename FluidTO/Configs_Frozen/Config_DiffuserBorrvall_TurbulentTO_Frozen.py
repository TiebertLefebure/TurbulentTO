import os
from dolfin import DOLFIN_EPS, Expression, MeshFunction, MPI, SubDomain, near
from Utilities_SharedTO import load_mesh_from_xdmf


# ===================================================================
# Configuration: Borrvall Diffuser - Turbulent Frozen (SA, Re = 1,000)
#
# One inlet on the left wall and one outlet on the right wall.
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

# Flow and Brinkman parameters for the frozen forward solve.
MU_FLUID_VALUE = 1.0e-3
RHO_FLUID_VALUE = 1.0
U_MAX_INLET = 1.0
U_MAX_OUTLET = 3.0

# =====================================================================================
# Reynolds number: Re = U_MAX_INLET * L * RHO_FLUID_VALUE / MU_FLUID_VALUE = 1,000
# =====================================================================================

# SA transport parameters for the frozen turbulence update.
# Prefer physical inlet-turbulence inputs over the raw eddy-viscosity ratio:
#   nu_t / nu_lam ~= 0.67 * I * Re * (ell / L_ref)
# The values below give nu_t / nu_lam ~= 5 at Re = 1000, close to the former
# SA_MUT_RATIO = 5.0 setting.
# Sensitivity check: keep SA_TURBULENCE_LENGTH_SCALE_RATIO = 0.07 fixed and
# sweep SA_TURBULENCE_INTENSITY = 0.03, 0.05, 0.10. If the topology changes
# qualitatively, report the result as inlet-turbulence-condition dependent.
SA_TURBULENCE_INTENSITY = 0.05
SA_TURBULENCE_LENGTH_SCALE_RATIO = 0.07
SA_REFERENCE_VELOCITY = U_MAX_INLET
SA_REFERENCE_LENGTH = L
SA_SMOOTH_ABS_EPS = 1.0e-12
SA_INIT_WALL_DIST_SCALE = 0.05 * L
SA_NU_TILDE_FLOOR = 1.0e-12
SA_NU_TILDE_PENALTY_ALPHA = 1.0e3
SA_NU_TILDE_PENALTY_N = 3.0

# Relaxed wall equation parameters for the external reciprocal distance solve.
SA_WALL_SIGMA = 0.01
SA_WALL_G0 = 20.0
SA_WALL_PENALTY_ALPHA = 1.0e3
SA_WALL_PENALTY_N = 3.0
SA_WALL_G_FLOOR = 1.0e-8

# MMA objective and continuation parameters for the topology update.
VOL_FRAC = 0.50
OBJECTIVE_CONVERGENCE_TOL = 1e-5
OBJECTIVE_STREAK_TO_STOP = 5

# Keep the first stages gray-friendly while the frozen-SA iterate settles, then
# raise q and beta once the topology is formed so the projected field does not
# stall with broad intermediate densities.
Q_PENAL_SCHEDULE = [0.1, 0.1, 0.2, 0.4, 0.8, 1.5, 3.0, 3.0]
MOVE_LIMIT_SCHEDULE = [0.03, 0.02, 0.015, 0.01, 0.01, 0.0075, 0.005, 0.005]
BETA_PROJ_SCHEDULE = [0.1, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
MAX_INNER_ITERATIONS_SCHEDULE = [80, 80, 100, 120, 120, 140, 140, 140]

# Frozen flow/turbulence coupling and IPCS solve parameters.
LINEAR_SOLVER = "mumps"
PICARD_STEPS = 1
TURBULENCE_RELAXATION = 0.05

# IPCS forward solver parameters:
#   These match the shared defaults in TurbulentTO_Frozen.py and are written here
#   explicitly so the case configuration is self-contained.
FORWARD_IPCS_DT = 2.5e-5
FORWARD_IPCS_MAX_ITERS = 400
FORWARD_IPCS_VELOCITY_RTOL = 1.0e-4
FORWARD_IPCS_PRESSURE_RTOL = 2.0e-3
FORWARD_IPCS_VEL_RELAXATION = 0.25
FORWARD_IPCS_P_RELAXATION = 0.07
FORWARD_IPCS_MAX_RESTARTS = 5
FORWARD_IPCS_DT_REDUCTION_FACTOR = 0.5
FORWARD_IPCS_RELAXATION_REDUCTION_FACTOR = 0.7
FORWARD_IPCS_VEL_SOLVER = "bicgstab" # Tentative velocity solve & velocity correction solver
FORWARD_IPCS_VEL_PRECONDITIONER = "ilu"
FORWARD_IPCS_P_SOLVER = "cg" # Pressure correction solve
FORWARD_IPCS_P_PRECONDITIONER = "ilu"
FORWARD_IPCS_LOG_EVERY = 50

FORWARD_IPCS_ACCEPT_BEST_SCORE = 1.20

# Projection, boundary-condition, and output settings for the optimization loop.
BETA_PROJ_VALUE = BETA_PROJ_SCHEDULE[0]
ETA_I = 0.50
QUADRATURE_DEGREE = 6
FILTER_RADIUS_IN_CELLS = 2.0

# Use pressure outlets here; switch to "velocity" only when imposing the outlet profile below.
OUTLET_BC_TYPE = "pressure"
OUTLET_PRESSURE_VALUE = 0.0  # Used only when OUTLET_BC_TYPE == "pressure".

ENABLE_PRESSURE_PIN = False
RESULTS_ROOT_NAME = "Results_Frozen/Results_DiffuserBorrvall_TurbulentTO_Frozen"

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
