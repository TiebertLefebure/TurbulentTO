
# File paths for mesh and boundary data
mesh_files = {
    'MESH_DIRECTORY': 'Meshes/Channel/Medium_WallResolved/mesh.xdmf',
    'FACET_DIRECTORY': 'Meshes/Channel/Medium_WallResolved/facet.xdmf'
}

# Specify type of boundaries
boundary_markers = {
    'INFLOW': [4],
    'OUTFLOW': [2],
    'WALLS': [1, 3],
    'SYMMETRY': None
}

# inlet bulk velocity U_ref = 20.0 m/s
# channel height H = 1.0 m

# hydraulic diameter D_h = 2 * H = 2.0 m

# initial boundary conditions for NU_TILDE: 
# NU_TILDE = 3 * ν = 3 * 0.00181818 = 0.005


# Initial conditions
initial_conditions = {
    'U': (20.0, 0.0),
    'P': 2.0,
    'NU_TILDE': 5e-3
}

# Boundary conditions
boundary_conditions = {
    'INFLOW':{
        'U': None,
        'P': 2.0, # 'P': 2.0
    },
    'OUTFLOW':{
        'U': None,
        'P': 0.0, # 'P': 0.0
    },
    'WALLS':{
        'U': (0.0, 0.0),
        'P': None,
        'NU_TILDE': 0.0
    },
    'SYMMETRY':{
        'U': None,
        'P': None,
    }
}

# kinematic 'VISCOSITY' ν = 0.00181818

# Physical quantities
physical_prm = {
    'VISCOSITY': 0.00181818, # kinematic viscosity 
    'FORCE': (0.0, 0.0) # 'FORCE': (0.0, 0.0)
}

# =============================================================
# Reynolds number: Re = U * D_h / VISCOSITY = 22,000
# =============================================================


# Simulation parameters for the transient SA model.
simulation_prm_SA = {
    'QUADRATURE_DEGREE': 2,
    'MAX_ITERATIONS': 6000,
    'TOLERANCE': 1e-6,
    'CFL_RELAXATION': 0.25,
    'STEP_SIZE': 0.005,
    'U_RELAXATION_FACTOR': 0.3,
    'NUT_RELAXATION_FACTOR': 0.3,
    # SA transport-equation SUPG multiplier. Set to 0.0 to disable SA SUPG.
    # Used only by the transient SA driver.
    'SA_SUPG_FACTOR': 1.0
}

# Specify where SA results are saved
saving_directory_SA = {
    'PVD_FILES': 'Results/Channel_SA/PVD files/',
    'H5_FILES':  'Results/Channel_SA/H5 files/',
    'RESIDUALS': 'Results/Channel_SA/Residual files/'
}

# Specify what to do after simulation
post_processing = {
    'PLOT': True,
    'SAVE': True,
}
