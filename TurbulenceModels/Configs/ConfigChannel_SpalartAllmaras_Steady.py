import os


# Mesh and output paths.
# Medium_WallResolved is the default SA channel validation mesh. Use
# Coarse_WallResolved for quick diagnostics or Fine_WallResolved for final runs.
MESH_DIRECTORY = "Meshes/Channel/Fine_WallResolved/mesh.xdmf"
FACET_DIRECTORY = "Meshes/Channel/Fine_WallResolved/facet.xdmf"


def infer_mesh_label_from_path(path):
    path_parts = os.path.normpath(path).split(os.sep)
    for mesh_label in (
        "Coarse_WallResolved",
        "Medium_WallResolved",
        "Fine_WallResolved",
        "Coarse",
        "Medium",
        "Fine",
    ):
        if mesh_label in path_parts:
            return mesh_label
    return path_parts[-2] if len(path_parts) >= 2 else "UnknownMesh"


MESH_LABEL = infer_mesh_label_from_path(MESH_DIRECTORY)

# ============================================================================================
# Channel wall-distance mode:
#   "exact"        -> analytical d(y)=min(y-y_min, y_max-y)
#   "relaxed_yoon" -> previous Yoon Eq. (19) reciprocal-distance solve
WALL_DISTANCE_MODE = "relaxed_yoon"

# Channel forcing mode:
#   "dirichlet_pressure" -> p=2 Pa on left, p=0 Pa on right
#   "body_force"         -> fully periodic pressure with equivalent force f_x=Delta p/L
CHANNEL_DRIVE_MODE = "body_force"
_CHANNEL_DRIVE_ALIASES = {
    "dirichlet": "dirichlet_pressure",
    "dirichlet_pressure": "dirichlet_pressure",
    "pressure": "dirichlet_pressure",
    "pressure_bc": "dirichlet_pressure",
    "body_force": "body_force",
    "bodyforce": "body_force",
    "force": "body_force",
}
CHANNEL_DRIVE_MODE_NORMALIZED = _CHANNEL_DRIVE_ALIASES.get(
    CHANNEL_DRIVE_MODE.strip().lower()
)
if CHANNEL_DRIVE_MODE_NORMALIZED is None:
    raise ValueError(
        "Unknown CHANNEL_DRIVE_MODE '{}'. Use 'dirichlet_pressure' or 'body_force'.".format(
            CHANNEL_DRIVE_MODE
        )
    )

RESULTS_ROOT = "Results/Channel_SA/{}_{}_{}_Steady".format(
    MESH_LABEL,
    WALL_DISTANCE_MODE,
    CHANNEL_DRIVE_MODE_NORMALIZED,
)
# ============================================================================================

# Boundary markers from the channel mesh.
BOUNDARY_MARKERS = {
    "INFLOW": [4],
    "OUTFLOW": [2],
    "WALLS": [1, 3],
    "SYMMETRY": None,
}

# Physical parameters.
CHANNEL_LENGTH = 2.0 # [m]
CHANNEL_HEIGHT = 2.0 # [m]
HYDRAULIC_DIAMETER = 2.0 * CHANNEL_HEIGHT # [m]
REYNOLDS_LENGTH = CHANNEL_HEIGHT # [m], channel-height-based Reynolds number convention
INLET_BULK_VELOCITY = 18.5 # [m/s] (reference, not an imposed inlet velocity)
KINEMATIC_VISCOSITY = 0.00181818 # [m^2/s]
CHANNEL_PRESSURE_DROP = 2.0 # [Pa] in the kinematic-pressure convention
CHANNEL_PRESSURE_GRADIENT_MAGNITUDE = CHANNEL_PRESSURE_DROP / CHANNEL_LENGTH # [Pa/m]
if CHANNEL_DRIVE_MODE_NORMALIZED == "body_force":
    BODY_FORCE = (CHANNEL_PRESSURE_GRADIENT_MAGNITUDE, 0.0)
    INFLOW_PRESSURE = None
    OUTFLOW_PRESSURE = None
    INITIAL_PRESSURE = 0.0
    FLOW_IPCS_NORMALIZE_PRESSURE_MEAN = True
else:
    BODY_FORCE = (0.0, 0.0)
    INFLOW_PRESSURE = CHANNEL_PRESSURE_DROP
    OUTFLOW_PRESSURE = 0.0
    INITIAL_PRESSURE = INFLOW_PRESSURE
    FLOW_IPCS_NORMALIZE_PRESSURE_MEAN = False
REYNOLDS_NUMBER = INLET_BULK_VELOCITY * REYNOLDS_LENGTH / KINEMATIC_VISCOSITY


# ==================================================================================================
# Reynolds number: Re_H = 18.5 * 2.0 / 0.00181818 ~= 20,350
# ==================================================================================================
# Wall-resolved mesh check using Cf = 0.079 Re^(-0.25):
#   Cf ~= 6.61e-3
#   u_tau = INLET_BULK_VELOCITY * sqrt(Cf / 2) ~= 1.064 m/s
#   y_first(y+=1) = KINEMATIC_VISCOSITY / u_tau ~= 1.709e-3 m
#   wall-resolved channel meshes use y_first ~= 1.709e-3 m.
# ==================================================================================================


# SA field initialization.
#INITIAL_VELOCITY = (0.0, 0.0) 
INITIAL_VELOCITY = (INLET_BULK_VELOCITY, 0.0)
INITIAL_NU_TILDE = 5.0e-3
WALL_NU_TILDE = 0.0

# Boundary conditions for the steady IPCS/Picard solve.
BOUNDARY_CONDITIONS = {
    "INFLOW": {
        "U": None,
        "P": INFLOW_PRESSURE,
        "NU_TILDE": None,
    },
    "OUTFLOW": {
        "U": None,
        "P": OUTFLOW_PRESSURE,
        "NU_TILDE": None,
    },
    "WALLS": {
        "U": (0.0, 0.0),
        "P": None,
        "NU_TILDE": WALL_NU_TILDE,
    },
    "SYMMETRY": {
        "U": None,
        "P": None,
        "NU_TILDE": None,
    },
}

# Steady SA IPCS/Picard controls.
QUADRATURE_DEGREE = 2

# Outer coupled fixed-point loop:
# one Picard step = one flow-to-steady IPCS solve + SA_SWEEPS_PER_STEP SA solves.
COUPLED_PICARD_MAX_STEPS = 400
COUPLED_PICARD_VELOCITY_TOLERANCE = 1.0e-5
COUPLED_PICARD_PRESSURE_TOLERANCE = 1.0e-6
COUPLED_PICARD_NU_TILDE_TOLERANCE = 1.0e-6
COUPLED_PICARD_SA_SWEEPS_PER_STEP = 1
COUPLED_PICARD_SA_RELAXATION = 0.3
SA_NU_TILDE_FLOOR = 1.0e-12
PICARD_CHECKPOINT_EVERY = 1

PICARD_CHECKPOINT_REQUIRE_FLOW_CONVERGENCE = False
CHANNEL_BULK_CONVERGENCE_WINDOW = 250
CHANNEL_BULK_CONVERGENCE_RELATIVE_TOLERANCE = 1.0e-3

# Inner pseudo-time flow solve used inside each outer Picard step.
FLOW_IPCS_TIME_STEP = 5.0e-3
FLOW_IPCS_MAX_STEPS = 500
FLOW_IPCS_VELOCITY_TOLERANCE = 1.0e-6
FLOW_IPCS_PRESSURE_TOLERANCE = 1.0e-6
FLOW_IPCS_VELOCITY_RELAXATION = 0.3
FLOW_IPCS_PRESSURE_RELAXATION = 0.3
FLOW_IPCS_LOG_EVERY = 10
# If enabled, verbose IPCS diagnostics print to the terminal only.
FLOW_IPCS_VERBOSE = True

FLOW_IPCS_LINEAR_SOLVER = "mumps"
FLOW_IPCS_LINEAR_PRECONDITIONER = None
SA_TRANSPORT_LINEAR_SOLVER = "default"
SA_TRANSPORT_LINEAR_PRECONDITIONER = "default"

# Restart controls. With RESTART_REQUIRE_FILES disabled, the first run in a new
# result directory starts from the initial condition, while later runs resume
# from the checkpointed HDF5 state if all fields are present.
RESTART_FROM_SAVED_STATE = True
RESTART_REQUIRE_FILES = False
RESTART_H5_DIRECTORY = "{}/H5 files".format(RESULTS_ROOT)

# G-equation wall-distance parameters.
WALL_DISTANCE_SIGMA_W = 0.1
WALL_DISTANCE_G0 = 20.0
WALL_DISTANCE_G_FLOOR = 1.0e-8
WALL_DISTANCE_NEWTON_RTOL = 1.0e-8
WALL_DISTANCE_NEWTON_ATOL = 1.0e-10
WALL_DISTANCE_NEWTON_MAX_ITERATIONS = 80
WALL_DISTANCE_NEWTON_RELAXATION = 0.5

PLOT_RESULTS = True
SAVE_RESULTS = True

# Solver-facing dictionaries used by the steady simulation script.
mesh_files = {
    "MESH_DIRECTORY": MESH_DIRECTORY,
    "FACET_DIRECTORY": FACET_DIRECTORY,
}

boundary_markers = BOUNDARY_MARKERS
boundary_conditions = BOUNDARY_CONDITIONS

pressure_drop_metric = {
    "LABEL": "equivalent channel pressure drop",
    "PRESSURE_DROP": CHANNEL_PRESSURE_DROP,
    "SOURCE": "{} forcing".format(CHANNEL_DRIVE_MODE_NORMALIZED),
    "INLET_MARKERS": BOUNDARY_MARKERS["INFLOW"],
    "OUTLET_MARKERS": BOUNDARY_MARKERS["OUTFLOW"],
    "INLET_PRESSURE": BOUNDARY_CONDITIONS["INFLOW"]["P"],
    "OUTLET_PRESSURE": BOUNDARY_CONDITIONS["OUTFLOW"]["P"],
}

initial_conditions = {
    "U": INITIAL_VELOCITY,
    "P": INITIAL_PRESSURE,
    "NU_TILDE": INITIAL_NU_TILDE,
}

physical_prm = {
    "VISCOSITY": KINEMATIC_VISCOSITY,
    "FORCE": BODY_FORCE,
}

steady_sa_solver_parameters = {
    "QUADRATURE_DEGREE": QUADRATURE_DEGREE,
    "COUPLED_PICARD_MAX_STEPS": COUPLED_PICARD_MAX_STEPS,
    "COUPLED_PICARD_VELOCITY_TOLERANCE": COUPLED_PICARD_VELOCITY_TOLERANCE,
    "COUPLED_PICARD_PRESSURE_TOLERANCE": COUPLED_PICARD_PRESSURE_TOLERANCE,
    "COUPLED_PICARD_NU_TILDE_TOLERANCE": COUPLED_PICARD_NU_TILDE_TOLERANCE,
    "COUPLED_PICARD_SA_SWEEPS_PER_STEP": COUPLED_PICARD_SA_SWEEPS_PER_STEP,
    "COUPLED_PICARD_SA_RELAXATION": COUPLED_PICARD_SA_RELAXATION,
    "SA_NU_TILDE_FLOOR": SA_NU_TILDE_FLOOR,
    "PICARD_CHECKPOINT_EVERY": PICARD_CHECKPOINT_EVERY,
    "PICARD_CHECKPOINT_REQUIRE_FLOW_CONVERGENCE": PICARD_CHECKPOINT_REQUIRE_FLOW_CONVERGENCE,
    "FLOW_IPCS_TIME_STEP": FLOW_IPCS_TIME_STEP,
    "FLOW_IPCS_MAX_STEPS": FLOW_IPCS_MAX_STEPS,
    "FLOW_IPCS_VELOCITY_TOLERANCE": FLOW_IPCS_VELOCITY_TOLERANCE,
    "FLOW_IPCS_PRESSURE_TOLERANCE": FLOW_IPCS_PRESSURE_TOLERANCE,
    "FLOW_IPCS_VELOCITY_RELAXATION": FLOW_IPCS_VELOCITY_RELAXATION,
    "FLOW_IPCS_PRESSURE_RELAXATION": FLOW_IPCS_PRESSURE_RELAXATION,
    "FLOW_IPCS_LOG_EVERY": FLOW_IPCS_LOG_EVERY,
    "FLOW_IPCS_VERBOSE": FLOW_IPCS_VERBOSE,
    "FLOW_IPCS_NORMALIZE_PRESSURE_MEAN": FLOW_IPCS_NORMALIZE_PRESSURE_MEAN,
    "FLOW_IPCS_LINEAR_SOLVER": FLOW_IPCS_LINEAR_SOLVER,
    "FLOW_IPCS_LINEAR_PRECONDITIONER": FLOW_IPCS_LINEAR_PRECONDITIONER,
    "SA_TRANSPORT_LINEAR_SOLVER": SA_TRANSPORT_LINEAR_SOLVER,
    "SA_TRANSPORT_LINEAR_PRECONDITIONER": SA_TRANSPORT_LINEAR_PRECONDITIONER,
    "RESTART_FROM_SAVED_STATE": RESTART_FROM_SAVED_STATE,
    "RESTART_REQUIRE_FILES": RESTART_REQUIRE_FILES,
    "RESTART_H5_DIRECTORY": RESTART_H5_DIRECTORY,
}

saving_directory = {
    "PVD_FILES": "{}/PVD files/".format(RESULTS_ROOT),
    "H5_FILES": "{}/H5 files/".format(RESULTS_ROOT),
    "RESIDUALS": "{}/Residual files/".format(RESULTS_ROOT),
}

post_processing = {
    "PLOT": PLOT_RESULTS,
    "SAVE": SAVE_RESULTS,
}
