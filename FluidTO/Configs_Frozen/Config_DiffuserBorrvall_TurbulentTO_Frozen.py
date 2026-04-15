import os
from dolfin import DOLFIN_EPS, Expression, Mesh, MeshFunction, MPI, SubDomain, XDMFFile, near
from Utilities_SharedTO import load_mesh_from_xdmf


# ===================================================================
# Configuration: Borrvall Diffuser - Turbulent (SA, Re = 1,000)
#
# One inlet on the left wall and one outlet on the right wall.
# ===================================================================

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)

# Mesh files
# Generate via: cd Meshes/DiffuserBorrvall && python3 diffuser_gmsh.py && python3 gmsh_to_xdmf.py
mesh_files = {
    'MESH_DIRECTORY': os.path.join(REPO_ROOT, 'Meshes/DiffuserBorrvall/mesh.xdmf'),
}


def create_design_mesh():
    return load_mesh_from_xdmf(mesh_files['MESH_DIRECTORY'], MPI.comm_world)


# Domain and mesh
L = 1.0
N = 120  # reference resolution used to generate the Gmsh mesh (LC = L/N)
DOMAIN_X_MIN = 0.0
DOMAIN_Y_MIN = 0.0
DOMAIN_X_MAX = L
DOMAIN_Y_MAX = L
TOL = DOLFIN_EPS

# Diffuser opening on right boundary
OUTLET_Y_MIN = 1.0 / 3.0
OUTLET_Y_MAX = 2.0 / 3.0

# Flow settings
MU_FLUID_VALUE = 1.0
RHO_FLUID_VALUE = 1.0
U_MAX_INLET = 1.0
U_MAX_OUTLET = 3.0

# ------------------------------------------------------------------------------------
# Reynolds number: Re = U_MAX_INLET * L * RHO_FLUID_VALUE / MU_FLUID_VALUE = 1
# ------------------------------------------------------------------------------------

# Spalart-Allmaras settings
# Use the same ratio-based inlet workflow as the other turbulent Borrvall cases.
# This tiny ratio preserves the previous effective inlet level from
# SA_NU_TILDE_INLET = 1.0e-4 with nu_lam = MU_FLUID_VALUE / RHO_FLUID_VALUE = 1.0e-3.
SA_MUT_RATIO = 2.794e-7
SA_DISTANCE_RELAXATION = 0.01
SA_SMOOTH_ABS_EPS = 1.0e-12
SA_INIT_WALL_DIST_SCALE = 0.05 * L
SA_NU_TILDE_FLOOR = 1.0e-12
SA_NU_TILDE_PENALTY_ALPHA = 1.0e3
SA_NU_TILDE_PENALTY_N = 3.0

# Penalized reciprocal wall-distance equation (Yoon 2016 Eq. 25)
SA_USE_PENALIZED_WALL_DISTANCE = True
SA_WALL_SIGMA = SA_DISTANCE_RELAXATION
SA_WALL_G0 = 20.0
SA_WALL_PENALTY_ALPHA = 1.0e3
SA_WALL_PENALTY_N = 3.0
SA_WALL_G_FLOOR = 1.0e-8

# Topology optimization settings
VOL_FRAC = 0.50
OBJECTIVE_CONVERGENCE_TOL = 1e-5
OBJECTIVE_STREAK_TO_STOP = 5

# Keep the first stages gray-friendly while the frozen-SA iterate settles, then
# raise q and beta once the topology is formed so the projected field does not
# stall with broad intermediate densities.
Q_PENAL_SCHEDULE = [0.1, 0.1, 0.2, 0.4, 0.8, 1.5, 3.0, 3.0]
MOVE_LIMIT_SCHEDULE = [0.08, 0.06, 0.04, 0.03, 0.02, 0.015, 0.01, 0.005]
BETA_PROJ_SCHEDULE = [0.1, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
MAX_INNER_ITERATIONS_SCHEDULE = [80, 80, 100, 120, 120, 140, 140, 140]

SNES_LINEAR_SOLVER = "mumps"
FROZEN_PICARD_STEPS = 1
NUT_RELAXATION_FACTOR = 0.5

# IPCS forward solver parameters:
#   These match the shared defaults in TurbulentTO_Frozen.py and are written here
#   explicitly so the case configuration is self-contained.
FORWARD_IPCS_DT = 2.0e-4
FORWARD_IPCS_MAX_ITERS = 300
FORWARD_IPCS_VELOCITY_RTOL = 1.0e-4
FORWARD_IPCS_PRESSURE_RTOL = 2.0e-3
FORWARD_IPCS_VEL_RELAXATION = 0.50
FORWARD_IPCS_P_RELAXATION = 0.10
FORWARD_IPCS_MAX_RESTARTS = 3
FORWARD_IPCS_DT_REDUCTION_FACTOR = 0.5
FORWARD_IPCS_RELAXATION_REDUCTION_FACTOR = 0.7
FORWARD_IPCS_RESTART_WITH_STOKES = False
FORWARD_IPCS_VEL_SOLVER = "bicgstab"
FORWARD_IPCS_VEL_PRECONDITIONER = "ilu"
FORWARD_IPCS_P_SOLVER = "cg"
FORWARD_IPCS_P_PRECONDITIONER = "ilu"
FORWARD_IPCS_LOG_EVERY = 25

BETA_PROJ_VALUE = BETA_PROJ_SCHEDULE[0]
ETA_I = 0.50
QUADRATURE_DEGREE = 6
FILTER_RADIUS_IN_CELLS = 2.0

# Outlet BC toggle for the turbulent diffuser case:
# Change this to "pressure" to impose p = OUTLET_PRESSURE_VALUE on the outlet opening.
# Keep "velocity" to impose the outlet parabolic velocity profile built below.
OUTLET_BC_TYPE = "pressure"
OUTLET_PRESSURE_VALUE = 0.0  # Used only when OUTLET_BC_TYPE == "pressure".

ENABLE_PRESSURE_PIN = False
PRESSURE_PIN_POINT = (DOMAIN_X_MIN, DOMAIN_Y_MIN)
RESULTS_ROOT_NAME = "Results_Frozen/Results_DiffuserBorrvall_TurbulentTO_Frozen"

MARK = {"generic": 0, "walls": 1, "inlet": 2, "outlet": 3}


def between(value, limits, eps=DOLFIN_EPS):
    return (limits[0] - eps <= value) and (value <= limits[1] + eps)


class WallsBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and (
            near(x[1], DOMAIN_Y_MIN, TOL)
            or near(x[1], DOMAIN_Y_MAX, TOL)
            or (near(x[0], DOMAIN_X_MAX, TOL) and between(x[1], (DOMAIN_Y_MIN, OUTLET_Y_MIN), TOL))
            or (near(x[0], DOMAIN_X_MAX, TOL) and between(x[1], (OUTLET_Y_MAX, DOMAIN_Y_MAX), TOL))
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


def build_velocity_profile_sets():
    inlet_center = 0.5 * (DOMAIN_Y_MIN + DOMAIN_Y_MAX)
    inlet_width = DOMAIN_Y_MAX - DOMAIN_Y_MIN
    u_inlet = Expression(
        ("u_max * (1 - pow(2.0 * (x[1] - y_c) / width, 2))", "0.0"),
        degree=2,
        u_max=U_MAX_INLET,
        y_c=inlet_center,
        width=inlet_width,
    )

    outlet_center = 0.5 * (OUTLET_Y_MIN + OUTLET_Y_MAX)
    outlet_width = OUTLET_Y_MAX - OUTLET_Y_MIN
    if OUTLET_BC_TYPE == "pressure":
        return [u_inlet], []

    u_outlet = Expression(
        ("u_max * (1 - pow(2.0 * (x[1] - y_c) / width, 2))", "0.0"),
        degree=2,
        u_max=U_MAX_OUTLET,
        y_c=outlet_center,
        width=outlet_width,
    )

    return [u_inlet], [u_outlet]
