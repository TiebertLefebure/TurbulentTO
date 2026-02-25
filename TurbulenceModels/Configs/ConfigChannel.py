
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

# inlet bulk velocity U = 20.0 m/s
# channel height H = 1.0 m

# hydraulic diameter D = 2 * H = 2.0 m

# choose turbulence intensity I = 5%

# set turbulent length scale l ≈ 0.14 * H = 0.14 m

# initial boundary conditions for K & E:
# K = 1.5 * (U * I)^2 = 1.5 
# E = Cµ^(3/4) * K^(3/2) / l = 2.20, using Cµ = 0.09


# Initial conditions
initial_conditions = {
    'U': (20.0, 0.0),
    'P': 2.0,
    'K': 1.5,
    'E': 2.20,
}

# Boundary conditions
boundary_conditions = {
    'INFLOW':{
        'U': None,
        'P': 2.0, # 'P': 2.0
        'K': None,
        'E': None
    },
    'OUTFLOW':{
        'U': None,
        'P': 0.0, # 'P': 0.0
        'K': None,
        'E': None
    },
    'WALLS':{
        'U': (0.0, 0.0),
        'P': None,
        'K': 0.0,
        'E': None,
    },
    'SYMMETRY':{
        'U': None,
        'P': None,
        'K': None,
        'E': None
    }
}
# VISCOSITY ν = 0.00181818 m^2/s
# 
# for laminar flow: Poiseuille flow with parabolic velocity profile
# FORCE = 12 * ν * U_avg / H^2 = 0.4363632, with U_avg = 20.0 m/s, H = 1.0 m

# Physical quantities
physical_prm = {
    'VISCOSITY': 0.00181818, # kinematic viscosity 
    'FORCE': (0.0, 0.0) # 'FORCE': (0.0, 0.0)
}

# Reynolds number:
# Re = U_avg * D / ν = 20.0 * 2.0 / 0.001818 ≈ 2.2 x 10^4


# Simulation parameters
simulation_prm = {
    'QUADRATURE_DEGREE': 2,
    'MAX_ITERATIONS': 3000,
    'TOLERANCE': 1e-6,
    'CFL_RELAXATION': 0.25,
    'STEP_SIZE': 0.005
}

# Specify where results are saved
saving_directory = {
    'PVD_FILES': 'Results/Channel/PVD files/',
    'H5_FILES':  'Results/Channel/H5 files/',
    'RESIDUALS': 'Results/Channel/Residual files/'
}

# Specify what to do after simulation
post_processing = {
    'PLOT': True,
    'SAVE': True,
}
