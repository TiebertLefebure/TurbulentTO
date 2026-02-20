
# File paths for mesh and boundary data
mesh_files = {
    'MESH_DIRECTORY': 'Meshes/U-Bend/Coarse/mesh.xdmf',
    'FACET_DIRECTORY': 'Meshes/U-Bend/Coarse/facet.xdmf'
}

# Specify type of boundaries
boundary_markers = {
    'INFLOW': [2],
    'OUTFLOW': [3],
    'WALLS': [4]
}

# 2D U-bend (Ansys Manual VMFL048)

# radius R = 14 mm
# radius of curvature = 125 mm
# straight section lenghts H_LEG = 1555 mm

# hydraulic diameter D = 2 * R = 0.028 m

# inlet bulk velocity U = 0.02 m/s (laminar target)

# 'K' and 'E' are set to very small placeholder values (1e-10) in laminar mode


# Initial conditions
initial_conditions = {
    'U': (0.0, 0.0, 0.0), 
    'P': 0.0,
    'K': 1e-10,
    'E': 1e-10,
}

# Boundary conditions
boundary_conditions = {
    'INFLOW':{
        'U': (0.0, -0.02, 0.0),
        'P': None, 
        'K': None,
        'E': None,
    },
    'OUTFLOW':{
        'U': None,
        'P': 0.0,
        'K': None,
        'E': None,
    },
    'WALLS':{
        'U': (0.0, 0.0, 0.0),
        'P': None,
        'K': None,
        'E': None,
    }
}

# Physical quantities
physical_prm = {
    'VISCOSITY': 8.9e-7,  # kinematic viscosity 
    'FORCE': (0.0, 0.0, 0.0)
}

# Reynolds number:
# Re = U * D / ν = 0.02 * 0.028 / 8.9e-7 ≈ 6.3 x 10^2

# Simulation parameters
simulation_prm = {
    'QUADRATURE_DEGREE': 2,
    'MAX_ITERATIONS': 6000,
    'TOLERANCE': 1e-5,
    'CFL_RELAXATION': 0.05,
    'U_RELAXATION_FACTOR': 0.7,
    'TURB_RELAXATION_FACTOR': 0.7,
    'STEP_SIZE': 5e-4 
}

# Specify where results are saved
saving_directory = {
    'PVD_FILES': 'Results/U-Bend_Laminar/PVD files/',
    'H5_FILES':  'Results/U-Bend_Laminar/H5 files/',
    'RESIDUALS': 'Results/U-Bend_Laminar/Residual files/'
}

# Specify what to do after simulation
post_processing = {
    'PLOT': True,
    'SAVE': True,
}


