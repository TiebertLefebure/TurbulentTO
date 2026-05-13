import os

from dolfin import DOLFIN_EPS, Expression, MeshFunction, MPI, SubDomain, near

from Utilities_SharedTO import build_cell_tag_restriction_functions, load_mesh_from_xdmf


# ----------------------------------------------------------------
# Configuration: Dilgen 2018 U-Bend - Laminar reference (Re = 1)
#
# Uses the same non-design solid and fluid regions as the turbulent Dilgen
# U-bend case. The inlet profile is parabolic and Re is based on the maximum
# inlet velocity and full port height.
# ----------------------------------------------------------------

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

mesh_files = {
    "MESH_DIRECTORY": os.path.join(REPO_ROOT, "Meshes/UBendDilgen/mesh_yplus1.xdmf"),
    "CELL_DIRECTORY": os.path.join(REPO_ROOT, "Meshes/UBendDilgen/cell_yplus1.xdmf"),
}

DESIGN_DOMAIN_TAG = 1
NON_DESIGN_FLUID_TAG = 2
NON_DESIGN_SOLID_TAG = 3


def create_design_mesh():
    return load_mesh_from_xdmf(mesh_files["MESH_DIRECTORY"], MPI.comm_world)


build_density_bounds, build_volume_region, build_objective_region = build_cell_tag_restriction_functions(
    mesh_files["CELL_DIRECTORY"],
    design_tags=(DESIGN_DOMAIN_TAG,),
    non_design_fluid_tags=(NON_DESIGN_FLUID_TAG,),
    non_design_solid_tags=(NON_DESIGN_SOLID_TAG,),
)


H = 0.1
L = 10.0 * H
H_MAX = 1.0 / 120.0
LEAD_LENGTH = 2.0 * H
PORT_HEIGHT = 2.0 * H
TOP_PORT_Y_MIN = 5.5 * H
TOP_PORT_Y_MAX = TOP_PORT_Y_MIN + PORT_HEIGHT
BOTTOM_PORT_Y_MIN = 2.5 * H
BOTTOM_PORT_Y_MAX = BOTTOM_PORT_Y_MIN + PORT_HEIGHT

BAR_THICKNESS = H
BAR_RADIUS = 0.5 * BAR_THICKNESS
BAR_X_START = -LEAD_LENGTH
BAR_TIP_X = 5.0 * H
BAR_TOTAL_LENGTH = BAR_TIP_X - BAR_X_START
BAR_RECT_X_MAX = BAR_TIP_X - BAR_RADIUS
BAR_Y_MIN = 0.5 * (L - BAR_THICKNESS)
BAR_Y_MAX = BAR_Y_MIN + BAR_THICKNESS

DESIGN_X_MIN = 0.0
DESIGN_Y_MIN = 0.0
DESIGN_X_MAX = L
DESIGN_Y_MAX = L
DOMAIN_X_MIN = DESIGN_X_MIN - LEAD_LENGTH
DOMAIN_Y_MIN = DESIGN_Y_MIN
DOMAIN_X_MAX = DESIGN_X_MAX
DOMAIN_Y_MAX = DESIGN_Y_MAX
TOL = DOLFIN_EPS

REYNOLDS_NUMBER = 1.0
RHO_FLUID_VALUE = 1.0
U_MAX_INLET = 1.0
MU_FLUID_VALUE = U_MAX_INLET * PORT_HEIGHT * RHO_FLUID_VALUE / REYNOLDS_NUMBER

# Re = U_MAX_INLET * PORT_HEIGHT * RHO_FLUID_VALUE / MU_FLUID_VALUE = 1.

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
SNES_MAX_ITERS = 220

BETA_PROJ_VALUE = BETA_PROJ_SCHEDULE[0]
ETA_I = 0.50

OUTLET_BC_TYPE = "pressure"
OUTLET_PRESSURE_VALUE = 0.0
ENABLE_PRESSURE_PIN = False
RESULTS_ROOT_NAME = "Results_Laminar/Results_UBendDilgen_LaminarTO"

MARK = {"generic": 0, "walls": 1, "inlet": 2, "outlet": 3}


def between(value, limits, eps=DOLFIN_EPS):
    return (limits[0] - eps <= value) and (value <= limits[1] + eps)


def is_inlet_point(x):
    return near(x[0], DOMAIN_X_MIN, TOL) and between(x[1], (TOP_PORT_Y_MIN, TOP_PORT_Y_MAX), TOL)


def is_outlet_point(x):
    return near(x[0], DOMAIN_X_MIN, TOL) and between(x[1], (BOTTOM_PORT_Y_MIN, BOTTOM_PORT_Y_MAX), TOL)


class InletBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and is_inlet_point(x)


class OutletBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and is_outlet_point(x)


class WallsBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and not is_inlet_point(x) and not is_outlet_point(x)


def mark_boundaries(mesh):
    boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    boundaries.set_all(MARK["generic"])
    WallsBoundary().mark(boundaries, MARK["walls"])
    InletBoundary().mark(boundaries, MARK["inlet"])
    OutletBoundary().mark(boundaries, MARK["outlet"])
    return boundaries


def _parabolic_horizontal_profile(y_min, y_max, u_max):
    y_center = 0.5 * (y_min + y_max)
    width = y_max - y_min
    return Expression(
        ("u_max * (1 - pow(2.0 * (x[1] - y_c) / width, 2))", "0.0"),
        degree=2,
        u_max=u_max,
        y_c=y_center,
        width=width,
    )


def build_velocity_profile_sets():
    return [_parabolic_horizontal_profile(TOP_PORT_Y_MIN, TOP_PORT_Y_MAX, U_MAX_INLET)], []
