import os
from dolfin import DOLFIN_EPS, Expression, MeshFunction, MPI, SubDomain, near
from Utilities_SharedTO import load_mesh_from_xdmf


# -------------------------------------------------------------------
# Configuration file for Borrvall Diffuser case (Laminar baseline)
# -------------------------------------------------------------------

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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

# ==================================================================================
# Reynolds number: Re = U_MAX_INLET * L * RHO_FLUID_VALUE / MU_FLUID_VALUE = 1
# ==================================================================================

# Topology optimization settings (match legacy baseline)
VOL_FRAC = 0.50
MAX_INNER_ITERATIONS = 100
OBJECTIVE_CONVERGENCE_TOL = 1e-5
OBJECTIVE_STREAK_TO_STOP = 5

# Keep the original gray-friendly starting point, then add continuation stages
# that sharpen the projection and reduce MMA moves to clean up residual gray zones.
Q_PENAL_SCHEDULE = [0.1, 0.1, 0.2, 0.4, 0.8, 1.5, 3.0, 3.0]
MOVE_LIMIT_SCHEDULE = [0.2, 0.12, 0.08, 0.05, 0.03, 0.02, 0.01, 0.005]
BETA_PROJ_SCHEDULE = [0.1, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]

SNES_LINEAR_SOLVER = "mumps"
FILTER_RADIUS_IN_CELLS = 2.0
# Forward solve
FORWARD_SNES_RTOL = 1.0e-4
FORWARD_SNES_ATOL = 1.0e-6
# Adjoint solve
ADJOINT_SNES_RTOL = 1.0e-4
ADJOINT_SNES_ATOL = 1.0e-6
SNES_MAX_ITERS = 200

BETA_PROJ_VALUE = BETA_PROJ_SCHEDULE[0]
ETA_I = 0.50

ENABLE_PRESSURE_PIN = False
PRESSURE_PIN_POINT = (DOMAIN_X_MIN, DOMAIN_Y_MIN)
RESULTS_ROOT_NAME = "Results_DiffuserBorrvall_LaminarTO"

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
    u_outlet = Expression(
        ("u_max * (1 - pow(2.0 * (x[1] - y_c) / width, 2))", "0.0"),
        degree=2,
        u_max=U_MAX_OUTLET,
        y_c=outlet_center,
        width=outlet_width,
    )

    return [u_inlet], [u_outlet]
