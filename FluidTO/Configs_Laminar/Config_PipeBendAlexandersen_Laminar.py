import os

from dolfin import DOLFIN_EPS, Expression, MeshFunction, MPI, SubDomain, near

from Utilities_SharedTO import build_cell_tag_restriction_functions, load_mesh_from_xdmf


# -------------------------------------------------------------------------
# Configuration: Alexandersen 2026 Pipe-Bend - Laminar reference (Re = 1)
#
# Uses the same mesh and non-design inlet/outlet ducts as the turbulent
# Alexandersen pipe-bend case. The inlet profile is uniform, matching the
# turbulent comparison case, and Re is based on inlet velocity and height.
# -------------------------------------------------------------------------

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

mesh_files = {
    "MESH_DIRECTORY": os.path.join(REPO_ROOT, "Meshes/PipeBendAlexandersen/mesh_yplus1.xdmf"),
    "CELL_DIRECTORY": os.path.join(REPO_ROOT, "Meshes/PipeBendAlexandersen/cell_yplus1.xdmf"),
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
INLET_HEIGHT = 0.2 * L
OUTLET_WIDTH = 0.2 * L
INLET_Y_MIN = 0.7 * L
INLET_Y_MAX = INLET_Y_MIN + INLET_HEIGHT
OUTLET_X_MIN = 0.7 * L
OUTLET_X_MAX = OUTLET_X_MIN + OUTLET_WIDTH

DESIGN_X_MIN = 0.0
DESIGN_Y_MIN = 0.0
DESIGN_X_MAX = L
DESIGN_Y_MAX = L
DOMAIN_X_MIN = DESIGN_X_MIN - LEAD_LENGTH
DOMAIN_Y_MIN = DESIGN_Y_MIN - LEAD_LENGTH
DOMAIN_X_MAX = DESIGN_X_MAX
DOMAIN_Y_MAX = DESIGN_Y_MAX
TOL = DOLFIN_EPS

REYNOLDS_NUMBER = 1.0
RHO_FLUID_VALUE = 1.0
U_MAX_INLET = 1.0
MU_FLUID_VALUE = U_MAX_INLET * INLET_HEIGHT * RHO_FLUID_VALUE / REYNOLDS_NUMBER
ALPHA_FLUID = 0.0
ALPHA_SOLID = 100.0

# Re = U_MAX_INLET * INLET_HEIGHT * RHO_FLUID_VALUE / MU_FLUID_VALUE = 1.

VOL_FRAC = 0.25
OBJECTIVE_TYPE = "average_inlet_pressure"
OBJECTIVE_CONVERGENCE_TOL = 1.0e-5
OBJECTIVE_STREAK_TO_STOP = 5
INITIAL_DENSITY_VALUE = VOL_FRAC
INITIAL_DENSITY_MATCH_FILTERED_VOLUME = True

PAPER_Q_ALPHA_SCHEDULE = [150.0, 75.0, 30.0, 15.0]
Q_PENAL_SCHEDULE = [1.0 / q_alpha for q_alpha in PAPER_Q_ALPHA_SCHEDULE]
BETA_PROJ_SCHEDULE = [4.0, 6.0, 9.0, 13.0]
MOVE_LIMIT_SCHEDULE = [0.05, 0.04, 0.03, 0.02]
MAX_INNER_ITERATIONS_SCHEDULE = [25, 25, 25, 25]
MAX_INNER_ITERATIONS = MAX_INNER_ITERATIONS_SCHEDULE[0]

SNES_LINEAR_SOLVER = "mumps"
FILTER_BASE_LENGTH = H_MAX
FILTER_RADIUS_IN_CELLS = 4.0
FILTER_DENOMINATOR_FLOOR = 1.0e-12
FORWARD_SNES_RTOL = 1.0e-6
FORWARD_SNES_ATOL = 1.0e-8
FORWARD_SNES_MAX_ITERS = 220
ADJOINT_SNES_RTOL = 1.0e-6
ADJOINT_SNES_ATOL = 1.0e-9
SNES_MAX_ITERS = 200

BETA_PROJ_VALUE = BETA_PROJ_SCHEDULE[0]
ETA_I = 0.50

OUTLET_BC_TYPE = "pressure"
OUTLET_PRESSURE_VALUE = 0.0
PRESSURE_OUTLET_COMPONENT_BCS = [
    {"marker": "outlet", "component": 0, "value": 0.0},
]
ENABLE_PRESSURE_PIN = False
RESULTS_ROOT_NAME = "Results_Laminar/Results_PipeBendAlexandersen_LaminarTO"

MARK = {"generic": 0, "walls": 1, "inlet": 2, "outlet": 3}


def between(value, limits, eps=DOLFIN_EPS):
    return (limits[0] - eps <= value) and (value <= limits[1] + eps)


def is_inlet_point(x):
    return near(x[0], DOMAIN_X_MIN, TOL) and between(x[1], (INLET_Y_MIN, INLET_Y_MAX), TOL)


def is_outlet_point(x):
    return near(x[1], DOMAIN_Y_MIN, TOL) and between(x[0], (OUTLET_X_MIN, OUTLET_X_MAX), TOL)


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
