import os


# Mesh and output paths.
# Medium_WallResolved is the default wall-resolved SA validation mesh
# (first layer gives y+ ~= 1). Use Coarse_WallResolved for quick diagnostics
# or Fine_WallResolved for final refinement.
MESH_DIRECTORY = "Meshes/U-Bend/Medium_WallResolved/mesh.xdmf"
FACET_DIRECTORY = "Meshes/U-Bend/Medium_WallResolved/facet.xdmf"


def infer_mesh_label_from_path(path):
    path_parts = os.path.normpath(path).split(os.sep)
    for mesh_label in ("Coarse", "Medium", "Fine",
                       "Coarse_WallResolved", "Medium_WallResolved",
                       "Fine_WallResolved", "Medium_WallRefinement",
                       "Ansys_Inflation"):
        if mesh_label in path_parts:
            return mesh_label
    return path_parts[-2] if len(path_parts) >= 2 else "UnknownMesh"


MESH_LABEL = infer_mesh_label_from_path(MESH_DIRECTORY)
RESULTS_ROOT = "Results/U-Bend_SA/{}_Steady".format(MESH_LABEL)

# Boundary markers from the U-bend mesh.
BOUNDARY_MARKERS = {
    "INFLOW": [2],
    "OUTFLOW": [3],
    "WALLS": [4],
}

# Physical parameters.
PIPE_RADIUS = 0.014 # [m]
HYDRAULIC_DIAMETER = 2.0 * PIPE_RADIUS # [m]
INLET_BULK_VELOCITY = 1.42 # [m/s]
KINEMATIC_VISCOSITY = 8.9e-7 # [m^2/s]
BODY_FORCE = (0.0, 0.0)
REYNOLDS_NUMBER = INLET_BULK_VELOCITY * HYDRAULIC_DIAMETER / KINEMATIC_VISCOSITY

# =================================================================================================
# Reynolds number: Re = INLET_BULK_VELOCITY * HYDRAULIC_DIAMETER / KINEMATIC_VISCOSITY = 44,700
# =================================================================================================

# SA inlet estimate from the ANSYS inlet turbulence specification.
C_MU = 0.09
TURBULENCE_INTENSITY = 0.05
TURBULENCE_LENGTH_SCALE = 0.07 * HYDRAULIC_DIAMETER
INLET_TURBULENT_KINETIC_ENERGY = 1.5 * (INLET_BULK_VELOCITY * TURBULENCE_INTENSITY) ** 2
INLET_TURBULENT_DISSIPATION = (
    (C_MU ** 0.75)
    * (INLET_TURBULENT_KINETIC_ENERGY ** 1.5)
    / TURBULENCE_LENGTH_SCALE
)
INLET_NU_TILDE = C_MU * (INLET_TURBULENT_KINETIC_ENERGY ** 2) / INLET_TURBULENT_DISSIPATION
WALL_NU_TILDE = 0.0

# Initial and boundary conditions for the steady IPCS/Picard solve.
INITIAL_VELOCITY = (0.0, 0.0)
INITIAL_PRESSURE = 0.0
INITIAL_NU_TILDE = INLET_NU_TILDE
INFLOW_VELOCITY = (0.0, -INLET_BULK_VELOCITY)

BOUNDARY_CONDITIONS = {
    "INFLOW": {
        "U": INFLOW_VELOCITY,
        "P": None,
        "NU_TILDE": INLET_NU_TILDE,
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
}

# Steady SA IPCS/Picard controls.
QUADRATURE_DEGREE = 4

# Outer coupled fixed-point loop:
# one Picard step = one flow-to-steady IPCS solve + SA_SWEEPS_PER_STEP SA solves.
COUPLED_PICARD_MAX_STEPS = 180
COUPLED_PICARD_VELOCITY_TOLERANCE = 1.0e-4
COUPLED_PICARD_PRESSURE_TOLERANCE = 1.0e-4
COUPLED_PICARD_NU_TILDE_TOLERANCE = 1.0e-6
COUPLED_PICARD_SA_SWEEPS_PER_STEP = 1
# The wall-resolved mesh is stiff near the wall. Keep the SA fixed-point update
# damped so a single SA solve cannot inject a large turbulent-viscosity jump.
COUPLED_PICARD_SA_RELAXATION = 0.05
SA_NU_TILDE_FLOOR = 1.0e-12

# Inner pseudo-time flow solve used inside each outer Picard step.
FLOW_IPCS_TIME_STEP = 1.0e-4
FLOW_IPCS_MAX_STEPS = 120
FLOW_IPCS_VELOCITY_TOLERANCE = 1.0e-5
FLOW_IPCS_PRESSURE_TOLERANCE = 1.0e-5
FLOW_IPCS_VELOCITY_RELAXATION = 0.3
FLOW_IPCS_PRESSURE_RELAXATION = 0.1
FLOW_IPCS_LOG_EVERY = 10
FLOW_IPCS_NORMALIZE_PRESSURE_MEAN = False

# Fallback solver used by any IPCS block without an explicit block-specific setting.
FLOW_IPCS_LINEAR_SOLVER = "gmres"
FLOW_IPCS_LINEAR_PRECONDITIONER = "hypre_amg"
FLOW_IPCS_VELOCITY_LINEAR_SOLVER = "gmres"
FLOW_IPCS_VELOCITY_LINEAR_PRECONDITIONER = "ilu"
FLOW_IPCS_PRESSURE_LINEAR_SOLVER = "gmres"
FLOW_IPCS_PRESSURE_LINEAR_PRECONDITIONER = "hypre_amg"
FLOW_IPCS_CORRECTION_LINEAR_SOLVER = "gmres"
FLOW_IPCS_CORRECTION_LINEAR_PRECONDITIONER = "ilu"
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

PLOT_RESULTS = False
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
    "FLOW_IPCS_VELOCITY_LINEAR_SOLVER": FLOW_IPCS_VELOCITY_LINEAR_SOLVER,
    "FLOW_IPCS_VELOCITY_LINEAR_PRECONDITIONER": FLOW_IPCS_VELOCITY_LINEAR_PRECONDITIONER,
    "FLOW_IPCS_PRESSURE_LINEAR_SOLVER": FLOW_IPCS_PRESSURE_LINEAR_SOLVER,
    "FLOW_IPCS_PRESSURE_LINEAR_PRECONDITIONER": FLOW_IPCS_PRESSURE_LINEAR_PRECONDITIONER,
    "FLOW_IPCS_CORRECTION_LINEAR_SOLVER": FLOW_IPCS_CORRECTION_LINEAR_SOLVER,
    "FLOW_IPCS_CORRECTION_LINEAR_PRECONDITIONER": FLOW_IPCS_CORRECTION_LINEAR_PRECONDITIONER,
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
