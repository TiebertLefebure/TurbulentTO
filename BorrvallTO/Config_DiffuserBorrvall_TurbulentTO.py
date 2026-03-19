import os
from dolfin import DOLFIN_EPS, DirichletBC, Expression, Mesh, MeshFunction, MPI, SubDomain, XDMFFile, inner, nabla_grad, near
from Utilities_LaminarTO import load_mesh_from_xdmf


# -------------------------------------------------------------------
# Configuration file for Borrvall Diffuser case (Turbulent)
# -------------------------------------------------------------------

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# Mesh files
# Generate via: cd Meshes/Diffuser && python3 diffuser_gmsh.py && python3 gmsh_to_xdmf.py
mesh_files = {
    'MESH_DIRECTORY': os.path.join(THIS_DIR, 'Meshes/Diffuser/mesh.xdmf'),
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

# ----------------------------------------------------------------------------------
# Reynolds number: Re = U_MAX_INLET * L * RHO_FLUID_VALUE / MU_FLUID_VALUE = 1.0
# ----------------------------------------------------------------------------------

# Spalart-Allmaras settings
SA_NU_TILDE_INLET = 1.0e-3
SA_NU_TILDE_INITIAL = SA_NU_TILDE_INLET
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
MAX_INNER_ITERATIONS = 80
OBJECTIVE_CONVERGENCE_TOL = 1e-5
OBJECTIVE_STREAK_TO_STOP = 5

# Keep the Brinkman penalization fixed and sharpen only the density projection
# so the diffuser can stay close to the current baseline while gaining a more
# robust continuation path for higher-Re runs.
Q_PENAL_SCHEDULE = [0.1, 0.1, 0.1, 0.1, 0.1]
MOVE_LIMIT_SCHEDULE = [0.20, 0.15, 0.10, 0.05, 0.03]
BETA_PROJ_SCHEDULE = [0.1, 0.5, 1.0, 2.0, 4.0]

SNES_LINEAR_SOLVER = "mumps"
FROZEN_PICARD_STEPS = 1
NUT_RELAXATION_FACTOR = 0.7
FORWARD_SNES_METHOD = "newtontr"
FORWARD_SNES_MAX_ITERS = 300
FORWARD_SNES_RTOL = 1.0e-4
FORWARD_SNES_ATOL = 1.0e-6
ADJOINT_SNES_RTOL = 1.0e-4
ADJOINT_SNES_ATOL = 1.0e-6

BETA_PROJ_VALUE = BETA_PROJ_SCHEDULE[0]
ETA_I = 0.50
QUADRATURE_DEGREE = 6
FILTER_RADIUS_IN_CELLS = 2.0

ENABLE_PRESSURE_PIN = False
PRESSURE_PIN_POINT = (DOMAIN_X_MIN, DOMAIN_Y_MIN)
RESULTS_ROOT_NAME = "Results_Diffuser_TurbulentTO"

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


def build_dissipation_density(u, mu_fluid):
    deformation = nabla_grad(u) + nabla_grad(u).T
    return 0.5 * mu_fluid * inner(deformation, deformation)


def build_adjoint_velocity_bcs(
    flow_space_adj,
    boundaries,
    wall_markers,
    inlet_markers,
    outlet_markers,
    u_noslip,
    inlet_profiles,
    outlet_profiles,
):
    bcu_walls_adj = [
        DirichletBC(flow_space_adj.sub(0), u_noslip, boundaries, marker) for marker in wall_markers
    ]
    bcu_inlet_adj = [
        DirichletBC(flow_space_adj.sub(0), profile, boundaries, marker)
        for profile, marker in zip(inlet_profiles, inlet_markers)
    ]
    bcu_outlet_adj = [
        DirichletBC(flow_space_adj.sub(0), profile, boundaries, marker)
        for profile, marker in zip(outlet_profiles, outlet_markers)
    ]
    return bcu_walls_adj + bcu_inlet_adj + bcu_outlet_adj
