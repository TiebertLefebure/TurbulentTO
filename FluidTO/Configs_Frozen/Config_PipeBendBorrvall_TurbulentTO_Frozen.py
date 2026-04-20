import os
import numpy as np
from math import pi
from dolfin import DOLFIN_EPS, Expression, Function, MeshFunction, MPI, SubDomain, cells, near
from Utilities_SharedTO import load_mesh_from_xdmf


# ===================================================================
# Configuration: Borrvall Pipe Bend - Turbulent Frozen (SA, Re = 2,000)
#
# One inlet on the left wall and one outlet on the bottom wall.
# ===================================================================

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Mesh path for this benchmark geometry.
# Generate via: cd Meshes/PipeBendBorrvall && python3 pipe_bend_gmsh.py && python3 gmsh_to_xdmf.py
mesh_files = {
    'MESH_DIRECTORY': os.path.join(REPO_ROOT, 'Meshes/PipeBendBorrvall/mesh.xdmf'),
}


def create_design_mesh():
    return load_mesh_from_xdmf(mesh_files['MESH_DIRECTORY'], MPI.comm_world)


# Geometry and reference meshing parameters for the pipe-bend design box.
L = 1.0
N = 120  # reference resolution used to generate the Gmsh mesh (LC = L/N)
TOL = DOLFIN_EPS

# Geometry parameters (Borrvall 2003 pipe bend case)
INLET_WIDTH = 0.2
INLET_TOP_OFFSET = 0.2
OUTLET_WIDTH = 0.2
OUTLET_RIGHT_OFFSET = 0.2

# Keep only a few x-cells behind the inlet facet non-design so the optimizer
# cannot choke the prescribed parabolic inflow immediately.
INLET_BLOCK_CELLS = 3
INLET_BLOCK_LENGTH = INLET_BLOCK_CELLS * (L / N)

# Flow and Brinkman parameters for the frozen forward solve.
MU_FLUID_VALUE = 1.0e-4
RHO_FLUID_VALUE = 1.0
U_MAX_INLET = 1.0
U_MAX_OUTLET = 1.0

# ----------------------------------------------------------------------------------------------
# Reynolds number: Re = U_MAX_INLET * INLET_WIDTH * RHO_FLUID_VALUE / MU_FLUID_VALUE = 2,000
# ----------------------------------------------------------------------------------------------

# SA transport parameters for the frozen turbulence update.
# SA_MUT_RATIO: target turbulent viscosity ratio nu_t / nu_lam at inlet.
# The main solver inverts the SA constitutive relation nu_t = nu_tilde * fv1(chi)
# to get the corresponding nu_tilde BC. At Re=2000 in an internal bend flow,
# a ratio of ~5-10 is physically reasonable.

SA_MUT_RATIO = 5.0                      # nu_t / nu_lam at inlet
SA_SMOOTH_ABS_EPS = 1.0e-12            # smoothing for |nu_tilde| in chi computation
SA_INIT_WALL_DIST_SCALE = 0.05 * L     # scale for initial nu_tilde ramp from walls
SA_NU_TILDE_FLOOR = 1.0e-12            # hard floor to prevent negative nu_tilde
SA_NU_TILDE_PENALTY_ALPHA = 1.0e3      # penalty strength for nu_tilde in solid regions
SA_NU_TILDE_PENALTY_N = 3.0            # penalty exponent (matches Brinkman)

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
PICARD_STEPS = 1
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


def compute_port_extents(l_box, inlet_top_offset, inlet_width, outlet_right_offset, outlet_width):
    inlet_y_max = l_box - inlet_top_offset
    inlet_y_min = inlet_y_max - inlet_width
    outlet_x_max = l_box - outlet_right_offset
    outlet_x_min = outlet_x_max - outlet_width
    return inlet_y_min, inlet_y_max, outlet_x_min, outlet_x_max


class InletBoundary(SubDomain):
    def __init__(self, tol, inlet_y_min, inlet_y_max):
        super().__init__()
        self._tol = tol
        self._inlet_y_min = inlet_y_min
        self._inlet_y_max = inlet_y_max

    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 0.0, self._tol) and between(
            x[1], (self._inlet_y_min, self._inlet_y_max), self._tol
        )


class OutletBoundary(SubDomain):
    def __init__(self, tol, outlet_x_min, outlet_x_max):
        super().__init__()
        self._tol = tol
        self._outlet_x_min = outlet_x_min
        self._outlet_x_max = outlet_x_max

    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], 0.0, self._tol) and between(
            x[0], (self._outlet_x_min, self._outlet_x_max), self._tol
        )


class WallsBoundary(SubDomain):
    def __init__(self, l_box, tol, inlet_y_min, inlet_y_max, outlet_x_min, outlet_x_max):
        super().__init__()
        self._l_box = l_box
        self._tol = tol
        self._inlet_y_min = inlet_y_min
        self._inlet_y_max = inlet_y_max
        self._outlet_x_min = outlet_x_min
        self._outlet_x_max = outlet_x_max

    def inside(self, x, on_boundary):
        left_wall_outside_inlet = near(x[0], 0.0, self._tol) and not between(
            x[1], (self._inlet_y_min, self._inlet_y_max), self._tol
        )
        bottom_wall_outside_outlet = near(x[1], 0.0, self._tol) and not between(
            x[0], (self._outlet_x_min, self._outlet_x_max), self._tol
        )
        right_wall = near(x[0], self._l_box, self._tol)
        top_wall = near(x[1], self._l_box, self._tol)
        return on_boundary and (
            left_wall_outside_inlet or bottom_wall_outside_outlet or right_wall or top_wall
        )


def mark_boundaries(mesh):
    inlet_y_min, inlet_y_max, outlet_x_min, outlet_x_max = compute_port_extents(
        L, INLET_TOP_OFFSET, INLET_WIDTH, OUTLET_RIGHT_OFFSET, OUTLET_WIDTH
    )
    boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    boundaries.set_all(MARK["generic"])
    WallsBoundary(L, TOL, inlet_y_min, inlet_y_max, outlet_x_min, outlet_x_max).mark(boundaries, MARK["walls"])
    InletBoundary(TOL, inlet_y_min, inlet_y_max).mark(boundaries, MARK["inlet"])
    OutletBoundary(TOL, outlet_x_min, outlet_x_max).mark(boundaries, MARK["outlet"])
    return boundaries


def build_velocity_profile_sets():
    inlet_y_min, inlet_y_max, outlet_x_min, outlet_x_max = compute_port_extents(
        L, INLET_TOP_OFFSET, INLET_WIDTH, OUTLET_RIGHT_OFFSET, OUTLET_WIDTH
    )
    y_inlet_center = 0.5 * (inlet_y_min + inlet_y_max)
    u_inlet = Expression(
        ("u_max * (1 - pow(2.0 * (x[1] - y_c) / width, 2))", "0.0"),
        degree=2, u_max=U_MAX_INLET, y_c=y_inlet_center, width=INLET_WIDTH,
    )
    # Pressure outlet: do not prescribe outlet velocity; the solver will apply outlet pressure instead.
    if OUTLET_BC_TYPE == "pressure":
        return [u_inlet], []

    x_outlet_center = 0.5 * (outlet_x_min + outlet_x_max)
    # Velocity outlet: prescribe a downward parabolic velocity profile on the outlet segment.
    u_outlet = Expression(
        ("0.0", "-u_max * (1 - pow(2.0 * (x[0] - x_c) / width, 2))"),
        degree=2, u_max=U_MAX_OUTLET, x_c=x_outlet_center, width=OUTLET_WIDTH,
    )
    return [u_inlet], [u_outlet]


def build_density_bounds(mesh, density_space):
    lower = Function(density_space)
    upper = Function(density_space)

    lower_values = np.zeros(density_space.dim())
    upper_values = np.ones(density_space.dim())
    dofmap = density_space.dofmap()
    inlet_y_min, inlet_y_max, _, _ = compute_port_extents(
        L, INLET_TOP_OFFSET, INLET_WIDTH, OUTLET_RIGHT_OFFSET, OUTLET_WIDTH
    )
    inlet_block_x_max = INLET_BLOCK_LENGTH

    for cell in cells(mesh):
        dof = dofmap.cell_dofs(cell.index())[0]
        midpoint = cell.midpoint()
        x_coord = midpoint.x()
        y_coord = midpoint.y()

        if 0.0 - DOLFIN_EPS <= x_coord <= inlet_block_x_max + DOLFIN_EPS:
            if between(y_coord, (inlet_y_min, inlet_y_max), TOL):
                lower_values[dof] = 1.0
                upper_values[dof] = 1.0
            else:
                lower_values[dof] = 0.0
                upper_values[dof] = 0.0

    lower.vector().set_local(lower_values)
    lower.vector().apply("insert")
    upper.vector().set_local(upper_values)
    upper.vector().apply("insert")
    return lower, upper


def build_volume_region(mesh, density_space):
    volume_region = Function(density_space)
    region_values = np.ones(density_space.dim())
    dofmap = density_space.dofmap()
    inlet_block_x_max = INLET_BLOCK_LENGTH

    for cell in cells(mesh):
        dof = dofmap.cell_dofs(cell.index())[0]
        x_coord = cell.midpoint().x()
        if 0.0 - DOLFIN_EPS <= x_coord <= inlet_block_x_max + DOLFIN_EPS:
            region_values[dof] = 0.0

    volume_region.vector().set_local(region_values)
    volume_region.vector().apply("insert")
    return volume_region
