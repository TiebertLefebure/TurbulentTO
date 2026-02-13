
# File paths for mesh and boundary data
mesh_files = {
    'MESH_DIRECTORY': 'Meshes/Channel/Coarse/mesh.xdmf',
    'FACET_DIRECTORY': 'Meshes/Channel/Coarse/facet.xdmf'
}

# Specify type of boundaries
boundary_markers = {
    'INFLOW': [4],
    'OUTFLOW': [2],
    'WALLS': [1, 3],
    'SYMMETRY': None
}

# inlet bulk velocity U = 0.2 m/s
# channel height H = 1.0 m

# hydraulic diameter D = 2 * H = 2.0 m


# Initial conditions
initial_conditions = {
    'U': (0.2, 0.0),
    'P': 0.0,
    'K': 1e-10,
    'E': 1e-10,
}

# Boundary conditions
boundary_conditions = {
    'INFLOW':{
        'U': None,
        'P': None, # periodic BCs: no pressure Dirichlet BCs
        'K': None,
        'E': None
    },
    'OUTFLOW':{
        'U': None,
        'P': None, # periodic BCs: no pressure Dirichlet BCs
        'K': None,
        'E': None
    },
    'WALLS':{
        'U': (0.0, 0.0),
        'P': None,
        'K': None,
        'E': None,
    },
    'SYMMETRY':{
        'U': None,
        'P': None,
        'K': None,
        'E': None
    }
}
# 'VISCOSITY' ν = 0.00181818
# 
# for laminar flow: Poiseuille flow with parabolic velocity profile
# 'FORCE' = 12 * ν * U_avg / H^2 

# Physical quantities
physical_prm = {
    'VISCOSITY': 0.00181818, # kinematic viscosity 
    'FORCE': (0.004363632, 0.0) # 'FORCE' = 12 * ν * U_avg / H^2 with U_avg = 0.2 m/s, H = 1 m
}

# Reynolds number:
# Re = U * D / ν = 0.2 * 2.0 / 0.001818 ≈ 2.2 x 10^2


# Simulation parameters
simulation_prm = {
    'QUADRATURE_DEGREE': 2,
    'MAX_ITERATIONS': 20000,
    'TOLERANCE': 1e-5,
    'CFL_RELAXATION': 0.05,
    'STEP_SIZE': 0.001,
    'MIN_STEP_SIZE': 1e-5,
    'MAX_STEP_SIZE': 1e-2
}

# Specify where results are saved
saving_directory = {
    'PVD_FILES': 'Results/Channel_Laminar/PVD files/',
    'H5_FILES':  'Results/Channel_Laminar/H5 files/',
    'RESIDUALS': 'Results/Channel_Laminar/Residual files/'
}

# Specify what to do after simulation
post_processing = {
    'PLOT': True,
    'SAVE': True,
}
