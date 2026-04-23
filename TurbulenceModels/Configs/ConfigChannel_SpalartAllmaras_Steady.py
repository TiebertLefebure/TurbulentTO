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
RESULTS_ROOT = "Results/Channel_SA/{}_Steady".format(MESH_LABEL)

# Boundary markers from the channel mesh.
BOUNDARY_MARKERS = {
    "INFLOW": [4],
    "OUTFLOW": [2],
    "WALLS": [1, 3],
    "SYMMETRY": None,
}

# Physical parameters.
CHANNEL_HEIGHT = 2.0 # [m]
HYDRAULIC_DIAMETER = 2.0 * CHANNEL_HEIGHT # [m]
REYNOLDS_LENGTH = CHANNEL_HEIGHT # [m], channel-height-based Reynolds number convention
INLET_BULK_VELOCITY = 18.5 # [m/s] (reference, not an imposed inlet velocity)
KINEMATIC_VISCOSITY = 0.00181818 # [m^2/s]
BODY_FORCE = (0.0, 0.0)
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
INITIAL_VELOCITY = (INLET_BULK_VELOCITY, 0.0)
INITIAL_PRESSURE = 2.0
INITIAL_NU_TILDE = 5.0e-3
WALL_NU_TILDE = 0.0

# Boundary conditions for the steady IPCS/Picard solve.
BOUNDARY_CONDITIONS = {
    "INFLOW": {
        "U": None,
        "P": 2.0,
        "NU_TILDE": None,
    },
    "OUTFLOW": {
        "U": None,
        "P": 0.0,
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
COUPLED_PICARD_MAX_STEPS = 200
COUPLED_PICARD_VELOCITY_TOLERANCE = 1.0e-6
COUPLED_PICARD_PRESSURE_TOLERANCE = 1.0e-6
COUPLED_PICARD_NU_TILDE_TOLERANCE = 1.0e-6
COUPLED_PICARD_SA_SWEEPS_PER_STEP = 1
COUPLED_PICARD_SA_RELAXATION = 0.3
SA_NU_TILDE_FLOOR = 1.0e-12

# Inner pseudo-time flow solve used inside each outer Picard step.
FLOW_IPCS_TIME_STEP = 0.005
FLOW_IPCS_MAX_STEPS = 500
FLOW_IPCS_VELOCITY_TOLERANCE = 1.0e-6
FLOW_IPCS_PRESSURE_TOLERANCE = 1.0e-6
FLOW_IPCS_VELOCITY_RELAXATION = 0.3
FLOW_IPCS_PRESSURE_RELAXATION = 0.3
FLOW_IPCS_LOG_EVERY = 100
FLOW_IPCS_NORMALIZE_PRESSURE_MEAN = False

FLOW_IPCS_LINEAR_SOLVER = "mumps"
FLOW_IPCS_LINEAR_PRECONDITIONER = None
SA_TRANSPORT_LINEAR_SOLVER = "default"
SA_TRANSPORT_LINEAR_PRECONDITIONER = "default"

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
    "FLOW_IPCS_TIME_STEP": FLOW_IPCS_TIME_STEP,
    "FLOW_IPCS_MAX_STEPS": FLOW_IPCS_MAX_STEPS,
    "FLOW_IPCS_VELOCITY_TOLERANCE": FLOW_IPCS_VELOCITY_TOLERANCE,
    "FLOW_IPCS_PRESSURE_TOLERANCE": FLOW_IPCS_PRESSURE_TOLERANCE,
    "FLOW_IPCS_VELOCITY_RELAXATION": FLOW_IPCS_VELOCITY_RELAXATION,
    "FLOW_IPCS_PRESSURE_RELAXATION": FLOW_IPCS_PRESSURE_RELAXATION,
    "FLOW_IPCS_LOG_EVERY": FLOW_IPCS_LOG_EVERY,
    "FLOW_IPCS_NORMALIZE_PRESSURE_MEAN": FLOW_IPCS_NORMALIZE_PRESSURE_MEAN,
    "FLOW_IPCS_LINEAR_SOLVER": FLOW_IPCS_LINEAR_SOLVER,
    "FLOW_IPCS_LINEAR_PRECONDITIONER": FLOW_IPCS_LINEAR_PRECONDITIONER,
    "SA_TRANSPORT_LINEAR_SOLVER": SA_TRANSPORT_LINEAR_SOLVER,
    "SA_TRANSPORT_LINEAR_PRECONDITIONER": SA_TRANSPORT_LINEAR_PRECONDITIONER,
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
