from dolfin import (
    DOLFIN_EPS,
    Expression,
    MeshFunction,
    Point,
    RectangleMesh,
    SubDomain,
    near,
)


# -------------------------------------------------------------------------
# Configuration file for Borrvall Pipe Bend case (Turbulent, no TO)
# -------------------------------------------------------------------------

# Full 1x1 design-domain box (entire box filled with fluid)
DOMAIN_X_MIN = 0.0
DOMAIN_Y_MIN = 0.0
DOMAIN_X_MAX = 1.0
DOMAIN_Y_MAX = 1.0
NX = 96
NY = 96
MESH_DIAGONAL = "crossed"
TOL = DOLFIN_EPS

# Pipe-bend ports carved on domain boundaries
INLET_WIDTH = 0.2
INLET_TOP_OFFSET = 0.2
OUTLET_WIDTH = 0.2
OUTLET_RIGHT_OFFSET = 0.2

# Marker IDs
MARK = {"generic": 0, "walls": 1, "inlet": 2, "outlet": 3}

# Solver-facing marker map (same style as legacy SA scripts)
boundary_markers = {
    "INFLOW": [MARK["inlet"]],
    "OUTFLOW": [MARK["outlet"]],
    "WALLS": [MARK["walls"]],
}

# Flow/SA settings
MU_FLUID_VALUE = 1.0e-5
RHO_FLUID_VALUE = 1.0
U_MAX_INLET = 1.0
U_MAX_OUTLET = 1.0
SA_NU_TILDE_INLET = 1.0e-3
SA_NU_TILDE_INITIAL = SA_NU_TILDE_INLET

# --------------------------------------------------------------------------------------------------
# Reynolds number: Re = U_MAX_INLET * INLET_WIDTH * RHO_FLUID_VALUE / MU_FLUID_VALUE = 2.0 x 10^4
# --------------------------------------------------------------------------------------------------

physical_prm = {
    "VISCOSITY": MU_FLUID_VALUE / RHO_FLUID_VALUE, # kinematic viscosity
    "FORCE": (0.0, 0.0),
}

initial_conditions = {
    "U": (0.0, 0.0),
    "P": 0.0,
    "NU_TILDE": SA_NU_TILDE_INITIAL,
}

simulation_prm = {
    "QUADRATURE_DEGREE": 6,
    "MAX_ITERATIONS": 2500,
    "TOLERANCE": 1.0e-6,
    "CFL_RELAXATION": 0.12,
    "U_RELAXATION_FACTOR": 0.7,
    "NUT_RELAXATION_FACTOR": 0.7,
    "STEP_SIZE": 2.0e-4,
    "MIN_STEP_SIZE": 5.0e-6,
    "MAX_STEP_SIZE": 2.0e-3,
    "DISTANCE_RELAXATION": 0.01,
    "LINEAR_SOLVER": "mumps",
}

ENABLE_PRESSURE_PIN = True
PRESSURE_PIN_POINT = (DOMAIN_X_MIN, DOMAIN_Y_MIN)

saving_directory = {
    "PVD_FILES": "PipeBendBorrvall_SA_Results/PVD/",
    "H5_FILES": "PipeBendBorrvall_SA_Results/H5/",
    "RESIDUALS": "PipeBendBorrvall_SA_Results/Residuals/",
}

post_processing = {
    "PLOT": False,
    "SAVE": True,
}


def between(value, limits, eps=DOLFIN_EPS):
    return (limits[0] - eps <= value) and (value <= limits[1] + eps)


def compute_port_extents(l_box, inlet_top_offset, inlet_width, outlet_right_offset, outlet_width):
    inlet_y_max = l_box - inlet_top_offset
    inlet_y_min = inlet_y_max - inlet_width
    outlet_x_max = l_box - outlet_right_offset
    outlet_x_min = outlet_x_max - outlet_width
    return inlet_y_min, inlet_y_max, outlet_x_min, outlet_x_max


INLET_Y_MIN, INLET_Y_MAX, OUTLET_X_MIN, OUTLET_X_MAX = compute_port_extents(
    DOMAIN_X_MAX,
    INLET_TOP_OFFSET,
    INLET_WIDTH,
    OUTLET_RIGHT_OFFSET,
    OUTLET_WIDTH,
)


class InletBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], DOMAIN_X_MIN, TOL) and between(
            x[1], (INLET_Y_MIN, INLET_Y_MAX), TOL
        )


class OutletBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], DOMAIN_Y_MIN, TOL) and between(
            x[0], (OUTLET_X_MIN, OUTLET_X_MAX), TOL
        )


class WallsBoundary(SubDomain):
    def inside(self, x, on_boundary):
        left_wall_outside_inlet = near(x[0], DOMAIN_X_MIN, TOL) and not between(
            x[1], (INLET_Y_MIN, INLET_Y_MAX), TOL
        )
        bottom_wall_outside_outlet = near(x[1], DOMAIN_Y_MIN, TOL) and not between(
            x[0], (OUTLET_X_MIN, OUTLET_X_MAX), TOL
        )
        right_wall = near(x[0], DOMAIN_X_MAX, TOL)
        top_wall = near(x[1], DOMAIN_Y_MAX, TOL)
        return on_boundary and (
            left_wall_outside_inlet or bottom_wall_outside_outlet or right_wall or top_wall
        )


def create_mesh():
    return RectangleMesh(
        Point(DOMAIN_X_MIN, DOMAIN_Y_MIN),
        Point(DOMAIN_X_MAX, DOMAIN_Y_MAX),
        NX,
        NY,
        MESH_DIAGONAL,
    )


def mark_boundaries(mesh):
    boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    boundaries.set_all(MARK["generic"])
    WallsBoundary().mark(boundaries, MARK["walls"])
    InletBoundary().mark(boundaries, MARK["inlet"])
    OutletBoundary().mark(boundaries, MARK["outlet"])
    return boundaries


def create_mesh_and_boundaries():
    mesh = create_mesh()
    boundaries = mark_boundaries(mesh)
    return mesh, boundaries


def build_velocity_profiles():
    inlet_center = 0.5 * (INLET_Y_MIN + INLET_Y_MAX)
    outlet_center = 0.5 * (OUTLET_X_MIN + OUTLET_X_MAX)

    u_inlet = Expression(
        ("u_max * (1.0 - pow(2.0 * (x[1] - y_c) / width, 2))", "0.0"),
        degree=2,
        u_max=U_MAX_INLET,
        y_c=inlet_center,
        width=INLET_WIDTH,
    )
    u_outlet = Expression(
        ("0.0", "-u_max * (1.0 - pow(2.0 * (x[0] - x_c) / width, 2))"),
        degree=2,
        u_max=U_MAX_OUTLET,
        x_c=outlet_center,
        width=OUTLET_WIDTH,
    )
    return u_inlet, u_outlet


U_INLET_PROFILE, U_OUTLET_PROFILE = build_velocity_profiles()

boundary_conditions = {
    "INFLOW": {
        "U": U_INLET_PROFILE,
        "P": None,
        "NU_TILDE": SA_NU_TILDE_INLET,
    },
    "OUTFLOW": {
        "U": U_OUTLET_PROFILE,
        "P": None,
        "NU_TILDE": None,
    },
    "WALLS": {
        "U": (0.0, 0.0),
        "P": None,
        "NU_TILDE": 0.0,
    },
}
