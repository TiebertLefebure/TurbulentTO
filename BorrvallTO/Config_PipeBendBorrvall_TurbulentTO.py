from dolfin import DOLFIN_EPS, Expression, MeshFunction, SubDomain, near


# --------------------------------------------------------------
# Configuration file for Borrvall Pipe Bend case (Turbulent)
# -------------------------------------------------------------- 


# ------------------------------------------------------------
# User parameters
# ------------------------------------------------------------
L = 1.0
N = 96
NX = 96
NY = 96
TOL = DOLFIN_EPS

# Geometry parameters (Borrvall 2003 pipe bend case)
INLET_WIDTH = 0.2
INLET_TOP_OFFSET = 0.2
OUTLET_WIDTH = 0.2
OUTLET_RIGHT_OFFSET = 0.2

# Flow settings
MU_FLUID_VALUE = 1.0e-5 # MU_FLUID_VALUE = 1.0e-4 for Re = 2.0 x 10^3
RHO_FLUID_VALUE = 1.0
U_MAX_INLET = 1.0
U_MAX_OUTLET = 1.0

# ----------------------------------------------------------------------------------------------------
# Reynolds number: Re = U_MAX_INLET * INLET_WIDTH * RHO_FLUID_VALUE / MU_FLUID_VALUE = 2.0 x 10^4
# ----------------------------------------------------------------------------------------------------

# Spalart-Allmaras settings
SA_NU_TILDE_INLET = 1.0e-3
SA_NU_TILDE_INITIAL = SA_NU_TILDE_INLET
SA_DISTANCE_RELAXATION = 0.01
SA_SMOOTH_ABS_EPS = 1.0e-12
SA_INIT_WALL_DIST_SCALE = 0.05 * L
SA_NU_TILDE_FLOOR = 1.0e-12

# ----------------------------------------------------------------------------
# Validated baseline tuning from standalone SA Borrvall PipeBend simulation (TurbulenceModels/PipeBendBorrvallSimulation.py)
# Note: steady TO solver (BorrvallTO/Borrvall_TurbulentTO.py) does not use transient time stepping
# STEP_SIZE/CFL bounds are kept for traceability and future variants
# ----------------------------------------------------------------------------
STEP_SIZE = 2.0e-4
MIN_STEP_SIZE = 5.0e-6
MAX_STEP_SIZE = 2.0e-3
CFL_RELAXATION = 0.12
U_RELAXATION_FACTOR = 0.7

# Topology optimization settings
VOL_FRAC = 0.50
MAX_INNER_ITERATIONS = 80 # MAX_INNER_ITERATIONS = 60
OBJECTIVE_CONVERGENCE_TOL = 1e-4 # OBJECTIVE_CONVERGENCE_TOL = 1e-3
OBJECTIVE_STREAK_TO_STOP = 5

Q_PENAL_SCHEDULE = [0.005, 0.01, 0.03, 0.05, 0.1, 0.2, 0.3] # Q_PENAL_SCHEDULE = [0.005, 0.01, 0.03, 0.05, 0.1]
MOVE_LIMIT_SCHEDULE = [0.01, 0.008, 0.006, 0.004, 0.003, 0.0025, 0.002] # MOVE_LIMIT_SCHEDULE = [0.01, 0.008, 0.006, 0.004, 0.003]
BETA_PROJ_SCHEDULE = [0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 16.0] # BETA_PROJ_VALUE = 0.1
SNES_LINEAR_SOLVER = "mumps"
INLET_RAMP_STEPS = 100
FROZEN_PICARD_STEPS = 2 # FROZEN_PICARD_STEPS = 1
FORWARD_SNES_METHOD = "newtontr"
FORWARD_SNES_MAX_ITERS = 300
FORWARD_SNES_RTOL = 1.0e-3
FORWARD_SNES_ATOL = 1.0e-6
ADJOINT_SNES_RTOL = 1.0e-3
ADJOINT_SNES_ATOL = 1.0e-6
NUT_RELAXATION_FACTOR = 0.7

BETA_PROJ_VALUE = 0.5 # BETA_PROJ_VALUE = 0.1
ETA_I = 0.50
QUADRATURE_DEGREE = 6

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


def mark_pipe_bend_boundaries(mesh, l_box, tol, inlet_y_min, inlet_y_max, outlet_x_min, outlet_x_max):
    boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    boundaries.set_all(MARK["generic"])
    WallsBoundary(l_box, tol, inlet_y_min, inlet_y_max, outlet_x_min, outlet_x_max).mark(
        boundaries, MARK["walls"]
    )
    InletBoundary(tol, inlet_y_min, inlet_y_max).mark(boundaries, MARK["inlet"])
    OutletBoundary(tol, outlet_x_min, outlet_x_max).mark(boundaries, MARK["outlet"])
    return boundaries


def build_velocity_profiles(
    u_max_inlet,
    u_max_outlet,
    inlet_y_min,
    inlet_y_max,
    outlet_x_min,
    outlet_x_max,
    inlet_width,
    outlet_width,
):
    y_inlet_center = 0.5 * (inlet_y_min + inlet_y_max)
    u_inlet = Expression(
        ("u_max * (1 - pow(2.0 * (x[1] - y_c) / width, 2))", "0.0"),
        degree=2,
        u_max=u_max_inlet,
        y_c=y_inlet_center,
        width=inlet_width,
    )

    x_outlet_center = 0.5 * (outlet_x_min + outlet_x_max)
    u_outlet = Expression(
        ("0.0", "-u_max * (1 - pow(2.0 * (x[0] - x_c) / width, 2))"),
        degree=2,
        u_max=u_max_outlet,
        x_c=x_outlet_center,
        width=outlet_width,
    )

    return u_inlet, u_outlet
