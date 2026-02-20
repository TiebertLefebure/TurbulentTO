from dolfin import (
    DOLFIN_EPS,
    Expression,
    MeshFunction,
    Point,
    RectangleMesh,
    SubDomain,
    near,
)


# ----------------------------------------------------------------------------
# Configuration file for Borrvall Double Pipe case (Turbulent, no TO)
# ----------------------------------------------------------------------------

# Domain and mesh (same design domain as TO setup)
DOMAIN_X_MIN = 0.0
DOMAIN_Y_MIN = 0.0
DOMAIN_X_MAX = 1.5
DOMAIN_Y_MAX = 1.0
NX = 150
NY = 100
MESH_DIAGONAL = "crossed"
TOL = DOLFIN_EPS

# Port layout on left/right boundaries
PORT_WIDTH = 1.0 / 6.0
PORT_TOP_MARGIN = 1.0 / 4.0
PORT_BOTTOM_MARGIN = 1.0 / 4.0

TOP_PORT_Y_MAX = DOMAIN_Y_MAX - PORT_TOP_MARGIN
TOP_PORT_Y_MIN = TOP_PORT_Y_MAX - PORT_WIDTH
BOTTOM_PORT_Y_MIN = PORT_BOTTOM_MARGIN
BOTTOM_PORT_Y_MAX = BOTTOM_PORT_Y_MIN + PORT_WIDTH

INLET_SEGMENTS = [
    (TOP_PORT_Y_MIN, TOP_PORT_Y_MAX),
    (BOTTOM_PORT_Y_MIN, BOTTOM_PORT_Y_MAX),
]
OUTLET_SEGMENTS = [
    (TOP_PORT_Y_MIN, TOP_PORT_Y_MAX),
    (BOTTOM_PORT_Y_MIN, BOTTOM_PORT_Y_MAX),
]

MARK = {"generic": 0, "walls": 1, "inlet": (2, 3), "outlet": (4, 5)}

boundary_markers = {
    "INFLOW": list(MARK["inlet"]),
    "OUTFLOW": list(MARK["outlet"]),
    "WALLS": [MARK["walls"]],
}

# Flow/SA settings
MU_FLUID_VALUE = 1.0e-5
RHO_FLUID_VALUE = 1.0
U_MAX_INLETS = [1.0, 1.0]
U_MAX_OUTLETS = [1.0, 1.0]
SA_NU_TILDE_INLETS = [1.0e-3, 1.0e-3]
SA_NU_TILDE_INITIAL = 1.0e-3

# Re = U_max * port_width * rho / mu = 1.66e4

physical_prm = {
    "VISCOSITY": MU_FLUID_VALUE / RHO_FLUID_VALUE,
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
    "P_RELAXATION_FACTOR": 1.0,
}
# ========================================================================================
# If pressure oscillates or convergence is inconsistent, try:
# 1) CFL_RELAXATION: 0.12 -> 0.05
# 2) MAX_STEP_SIZE: 2.0e-3 -> 5.0e-4
# 3) Add pressure under-relaxation in solver (e.g., P_RELAXATION_FACTOR: 1.0 -> 0.7)
# 4) Assess convergence with flux imbalance + u + nu_tilde, not only monotonic p residual
# ========================================================================================

ENABLE_PRESSURE_PIN = True
PRESSURE_PIN_POINT = (DOMAIN_X_MIN, DOMAIN_Y_MIN)

saving_directory = {
    "PVD_FILES": "DoublePipeBorrvall_SA_Results/PVD/",
    "H5_FILES": "DoublePipeBorrvall_SA_Results/H5/",
    "RESIDUALS": "DoublePipeBorrvall_SA_Results/Residuals/",
}

post_processing = {
    "PLOT": False,
    "SAVE": True,
}


def between(value, limits, eps=DOLFIN_EPS):
    return (limits[0] - eps <= value) and (value <= limits[1] + eps)


class VerticalPortBoundary(SubDomain):
    def __init__(self, x_location, tol, y_min, y_max):
        super().__init__()
        self._x_location = x_location
        self._tol = tol
        self._y_min = y_min
        self._y_max = y_max

    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], self._x_location, self._tol) and between(
            x[1], (self._y_min, self._y_max), self._tol
        )


class WallsBoundary(SubDomain):
    def __init__(self, x_min, x_max, y_min, y_max, tol, inlet_segments, outlet_segments):
        super().__init__()
        self._x_min = x_min
        self._x_max = x_max
        self._y_min = y_min
        self._y_max = y_max
        self._tol = tol
        self._inlet_segments = inlet_segments
        self._outlet_segments = outlet_segments

    def _inside_any_segment(self, y_value, segments):
        return any(between(y_value, segment, self._tol) for segment in segments)

    def inside(self, x, on_boundary):
        left_wall_outside_inlets = near(x[0], self._x_min, self._tol) and not self._inside_any_segment(
            x[1], self._inlet_segments
        )
        right_wall_outside_outlets = near(x[0], self._x_max, self._tol) and not self._inside_any_segment(
            x[1], self._outlet_segments
        )
        top_wall = near(x[1], self._y_max, self._tol)
        bottom_wall = near(x[1], self._y_min, self._tol)
        return on_boundary and (
            left_wall_outside_inlets or right_wall_outside_outlets or top_wall or bottom_wall
        )


def _validate_port_configuration():
    if len(MARK["inlet"]) != len(INLET_SEGMENTS):
        raise ValueError("MARK['inlet'] length must match INLET_SEGMENTS length.")
    if len(MARK["outlet"]) != len(OUTLET_SEGMENTS):
        raise ValueError("MARK['outlet'] length must match OUTLET_SEGMENTS length.")
    if len(U_MAX_INLETS) != len(INLET_SEGMENTS):
        raise ValueError("U_MAX_INLETS length must match INLET_SEGMENTS length.")
    if len(U_MAX_OUTLETS) != len(OUTLET_SEGMENTS):
        raise ValueError("U_MAX_OUTLETS length must match OUTLET_SEGMENTS length.")
    if len(SA_NU_TILDE_INLETS) != len(INLET_SEGMENTS):
        raise ValueError("SA_NU_TILDE_INLETS length must match INLET_SEGMENTS length.")


_validate_port_configuration()


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

    WallsBoundary(
        DOMAIN_X_MIN,
        DOMAIN_X_MAX,
        DOMAIN_Y_MIN,
        DOMAIN_Y_MAX,
        TOL,
        INLET_SEGMENTS,
        OUTLET_SEGMENTS,
    ).mark(boundaries, MARK["walls"])

    for marker, segment in zip(MARK["inlet"], INLET_SEGMENTS):
        VerticalPortBoundary(DOMAIN_X_MIN, TOL, segment[0], segment[1]).mark(boundaries, marker)
    for marker, segment in zip(MARK["outlet"], OUTLET_SEGMENTS):
        VerticalPortBoundary(DOMAIN_X_MAX, TOL, segment[0], segment[1]).mark(boundaries, marker)
    return boundaries


def create_mesh_and_boundaries():
    mesh = create_mesh()
    boundaries = mark_boundaries(mesh)
    return mesh, boundaries


def _build_horizontal_profile(u_max, y_min, y_max):
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
    inlet_profiles = [
        _build_horizontal_profile(u_max, segment[0], segment[1])
        for u_max, segment in zip(U_MAX_INLETS, INLET_SEGMENTS)
    ]
    outlet_profiles = [
        _build_horizontal_profile(u_max, segment[0], segment[1])
        for u_max, segment in zip(U_MAX_OUTLETS, OUTLET_SEGMENTS)
    ]
    return inlet_profiles, outlet_profiles


INLET_PROFILES, OUTLET_PROFILES = build_velocity_profile_sets()

boundary_conditions = {
    "INFLOW": {
        "U": INLET_PROFILES,
        "P": None,
        "NU_TILDE": SA_NU_TILDE_INLETS,
    },
    "OUTFLOW": {
        "U": OUTLET_PROFILES,
        "P": None,
        "NU_TILDE": None,
    },
    "WALLS": {
        "U": (0.0, 0.0),
        "P": None,
        "NU_TILDE": 0.0,
    },
}
