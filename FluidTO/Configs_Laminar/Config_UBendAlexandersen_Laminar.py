import os

from dolfin import DOLFIN_EPS, Expression, MeshFunction, MPI, SubDomain, near

from Utilities_SharedTO import build_cell_tag_restriction_functions, load_mesh_from_xdmf


# ----------------------------------------------------------------------
# Configuration: Alexandersen 2026 U-Bend - Laminar reference (Re = 1)
#
# Uses the same mesh and non-design inlet/outlet ducts as the turbulent
# Alexandersen U-bend case. The inlet profile is uniform, matching the
# turbulent comparison case, and Re is based on inlet velocity and port height.
# ----------------------------------------------------------------------

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

mesh_files = {
    "MESH_DIRECTORY": os.path.join(REPO_ROOT, "Meshes/UBendAlexandersen/mesh_yplus1.xdmf"),
    "CELL_DIRECTORY": os.path.join(REPO_ROOT, "Meshes/UBendAlexandersen/cell_yplus1.xdmf"),
}

DESIGN_DOMAIN_TAG = 1
NON_DESIGN_FLUID_TAG = 2


def create_design_mesh():
    return load_mesh_from_xdmf(mesh_files["MESH_DIRECTORY"], MPI.comm_world)


build_density_bounds, build_volume_region, build_objective_region = build_cell_tag_restriction_functions(
    mesh_files["CELL_DIRECTORY"],
    design_tags=(DESIGN_DOMAIN_TAG,),
    non_design_fluid_tags=(NON_DESIGN_FLUID_TAG,),
)


L = 1.0
H_MAX = 0.007
LEAD_LENGTH = 0.2 * L
PORT_HEIGHT = 0.2 * L
TOP_PORT_Y_MIN = 0.55 * L
TOP_PORT_Y_MAX = TOP_PORT_Y_MIN + PORT_HEIGHT
BOTTOM_PORT_Y_MIN = 0.25 * L
BOTTOM_PORT_Y_MAX = BOTTOM_PORT_Y_MIN + PORT_HEIGHT

BAR_THICKNESS = 0.10 * L
BAR_RADIUS = 0.5 * BAR_THICKNESS
BAR_X_START = -LEAD_LENGTH
BAR_TOTAL_LENGTH = 0.70 * L
BAR_TIP_X = BAR_X_START + BAR_TOTAL_LENGTH
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
ALPHA_FLUID = 0.0
ALPHA_SOLID = 100.0

# Re = U_MAX_INLET * PORT_HEIGHT * RHO_FLUID_VALUE / MU_FLUID_VALUE = 1.

VOL_FRAC = 0.27
OBJECTIVE_TYPE = "average_inlet_pressure"
OBJECTIVE_CONVERGENCE_TOL = 1.0e-5
OBJECTIVE_STREAK_TO_STOP = 5
INITIAL_DENSITY_VALUE = VOL_FRAC
INITIAL_DENSITY_MATCH_FILTERED_VOLUME = True

PAPER_Q_ALPHA_SCHEDULE = [150.0, 75.0, 35.0, 12.0]
Q_PENAL_SCHEDULE = [1.0 / q_alpha for q_alpha in PAPER_Q_ALPHA_SCHEDULE]
BETA_PROJ_SCHEDULE = [8.0, 10.0, 18.0, 18.0]
MOVE_LIMIT_SCHEDULE = [0.05, 0.04, 0.03, 0.02]
MAX_INNER_ITERATIONS_SCHEDULE = [25, 25, 25, 25]
MAX_INNER_ITERATIONS = MAX_INNER_ITERATIONS_SCHEDULE[0]

SNES_LINEAR_SOLVER = "mumps"
FILTER_BASE_LENGTH = H_MAX
FILTER_RADIUS_IN_CELLS = 4.0
FORWARD_SNES_RTOL = 1.0e-6
FORWARD_SNES_ATOL = 1.0e-8
FORWARD_SNES_MAX_ITERS = 260
ADJOINT_SNES_RTOL = 1.0e-6
ADJOINT_SNES_ATOL = 1.0e-9
SNES_MAX_ITERS = 200

BETA_PROJ_VALUE = BETA_PROJ_SCHEDULE[0]
ETA_I = 0.50

OUTLET_BC_TYPE = "pressure"
OUTLET_PRESSURE_VALUE = 0.0
PRESSURE_OUTLET_COMPONENT_BCS = [
    {"marker": "outlet", "component": 1, "value": 0.0},
]
ENABLE_PRESSURE_PIN = False
RESULTS_ROOT_NAME = "Results_Laminar/Results_UBendAlexandersen_LaminarTO"

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


def build_velocity_profile_sets():
    u_inlet = Expression(("u_max", "0.0"), degree=0, u_max=U_MAX_INLET)
    return [u_inlet], []
