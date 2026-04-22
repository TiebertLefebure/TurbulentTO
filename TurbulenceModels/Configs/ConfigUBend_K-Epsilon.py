
import os

# File paths for mesh and boundary data
mesh_files = {
    'MESH_DIRECTORY': 'Meshes/U-Bend/Ansys_Inflation/mesh.xdmf',
    'FACET_DIRECTORY': 'Meshes/U-Bend/Ansys_Inflation/facet.xdmf'
}


def _infer_mesh_label_from_path(path):
    parts = os.path.normpath(path).split(os.sep)
    for label in ('Coarse', 'Medium', 'Fine'):
        if label in parts:
            return label
    return parts[-2] if len(parts) >= 2 else 'UnknownMesh'


UBEND_KE_MESH_LABEL = _infer_mesh_label_from_path(mesh_files['MESH_DIRECTORY'])
UBEND_KE_RESULTS_ROOT = f'Results/U-Bend_K-E/{UBEND_KE_MESH_LABEL}'

# Specify type of boundaries
boundary_markers = {
    'INFLOW': [2],
    'OUTFLOW': [3],
    'WALLS': [4]
}

# ----------------
# 2D U-bend 
# ----------------

# pipe radius R = 14 mm
# pipe diameter D = 2 * R = 28 mm
# radius of curvature R_c = 125 mm
# straight section lengths H_LEG = 30 * D = 840 mm

# hydraulic diameter D_h = 2 * R = 0.028 m

# inlet bulk velocity U = 1.42 m/s

# choose turbulence intensity I = 5%

# set turbulent length scale l ≈ 0.07 * D = 2.0 mm

# initial & inflow boundary conditions for K & E:
# K = 1.5 * (U * I)^2 = 0.008
# E = Cµ^(3/4) * K^(3/2) / l = 0.054, using Cµ = 0.09



# Initial conditions
initial_conditions = {
    'U': (0.0, 0.0), 
    'P': 0.0,
    'K': 0.008,    
    'E': 0.054,    
}

# Boundary conditions
boundary_conditions = {
    'INFLOW':{
        'U': (0.0, -1.42), 
        'P': None, 
        'K': 0.008,
        'E': 0.054,
    },
    'OUTFLOW':{
        'U': None,
        'P': 0.0,
        'K': None,
        'E': None,
    },
    'WALLS':{
        'U': (0.0, 0.0),
        'P': None,
        'K': 0.0,
        'E': 0.0,
    }
}

# Physical quantities
physical_prm = {
    'VISCOSITY': 8.9e-7,  # kinematic viscosity
    'FORCE': (0.0, 0.0)
}

# -------------------------------------------------------------------------------
# Reynolds number: Re = U_ref * D_h / ν = 1.42 * 0.028 / 8.9e-7 ≈ 4.5 x 10^4
# -------------------------------------------------------------------------------

# Simulation parameters
simulation_prm = {
    'QUADRATURE_DEGREE': 2,
    'MAX_ITERATIONS': 3000,
    'TOLERANCE': 1e-6,
    # Optional field-specific tolerances. If omitted, each falls back to TOLERANCE.
    'TOLERANCE_U': 1e-6,
    'TOLERANCE_P': 1e-5,
    'TOLERANCE_K': 1e-6,
    'TOLERANCE_E': 1e-5,
    'CFL_RELAXATION': 0.3,
    'STEP_SIZE': 5e-4,
    'MIN_STEP_SIZE': 1e-5,
    'MAX_STEP_SIZE': 1e-3,

    # Relaxation factors
    'U_RELAXATION_FACTOR': 0.7,
    'TURB_RELAXATION_FACTOR': 0.7,

    # Linear solvers
    'NS_LINEAR_SOLVER': 'mumps',
    'NS_LINEAR_PRECONDITIONER': None,
    'KE_LINEAR_SOLVER': 'default',
    'KE_LINEAR_PRECONDITIONER': 'default',

    # Runtime output for long fine-mesh runs
    'RUNTIME_WRITE_INTERVAL': 25,
    'RUNTIME_WRITE_PVD': True,
    'RUNTIME_WRITE_RESIDUALS': True,
}

# Specify where results are saved
saving_directory = {
    'PVD_FILES': f'{UBEND_KE_RESULTS_ROOT}/PVD files/',
    'H5_FILES':  f'{UBEND_KE_RESULTS_ROOT}/H5 files/',
    'RESIDUALS': f'{UBEND_KE_RESULTS_ROOT}/Residual files/'
}

# Specify what to do after simulation
post_processing = {
    'PLOT': True,
    'SAVE': True,
}
