import os

from dolfin import DOLFIN_EPS, Expression, MeshFunction, MPI, SubDomain, near

from Utilities_SharedTO import build_cell_tag_restriction_functions, load_mesh_from_xdmf


# -------------------------------------------------------------
# Configuration: Yoon 2016 Diffuser - Laminar reference (Re = 1)
# -------------------------------------------------------------

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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


L = 1.0
N = 180
H_MAX = L / N
DESIGN_X_MIN = 0.0
DESIGN_X_MAX = L
DOMAIN_X_MIN = 0.0
DOMAIN_Y_MIN = 0.0
DOMAIN_X_MAX = L
DOMAIN_Y_MAX = L
OUTLET_Y_MIN = L / 3.0
OUTLET_Y_MAX = 2.0 * L / 3.0
TOL = DOLFIN_EPS

REYNOLDS_NUMBER = 1.0
RHO_FLUID_VALUE = 1.0
U_MAX_INLET = 1.0
U_MAX_OUTLET = 3.0 * U_MAX_INLET
MU_FLUID_VALUE = U_MAX_INLET * L * RHO_FLUID_VALUE / REYNOLDS_NUMBER

# Re = U_MAX_INLET * L * RHO_FLUID_VALUE / MU_FLUID_VALUE = 1.

VOL_FRAC = 0.30
INITIAL_DENSITY_VALUE = VOL_FRAC
MAX_INNER_ITERATIONS = 80
OBJECTIVE_TYPE = "dissipation"
OBJECTIVE_CONVERGENCE_TOL = 5e-5
OBJECTIVE_STREAK_TO_STOP = 5

Q_PENAL_SCHEDULE = [0.005, 0.01, 0.02, 0.03, 0.05, 0.10, 0.20, 0.35, 0.50]
MOVE_LIMIT_SCHEDULE = [0.05, 0.05, 0.04, 0.03, 0.02, 0.015, 0.01, 0.0075, 0.005]
BETA_PROJ_SCHEDULE = [0.3, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]

SNES_LINEAR_SOLVER = "mumps"
FILTER_BASE_LENGTH = H_MAX
FILTER_RADIUS_IN_CELLS = 3.0
FORWARD_SNES_RTOL = 1.0e-6
FORWARD_SNES_ATOL = 1.0e-9
ADJOINT_SNES_RTOL = 1.0e-6
ADJOINT_SNES_ATOL = 1.0e-9
SNES_MAX_ITERS = 200

BETA_PROJ_VALUE = BETA_PROJ_SCHEDULE[0]
ETA_I = 0.50

OUTLET_BC_TYPE = "velocity"
ENABLE_PRESSURE_PIN = True
PRESSURE_PIN_POINT = (DOMAIN_X_MIN, DOMAIN_Y_MIN)
RESULTS_ROOT_NAME = "Results_Laminar/Results_DiffuserYoon_LaminarTO"

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
