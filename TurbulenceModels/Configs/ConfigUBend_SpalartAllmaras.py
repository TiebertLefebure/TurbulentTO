
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
# straight section lenghts = 1555 mm

# hydraulic diameter D = 2 * R = 0.028 m

# inlet bulk velocity U = 1.42 m/s

# 'VISCOSITY' ν = 8.9e-7 

# Spalart-Allmaras inlet estimate based on k-epsilon inlet values:
# NU_TILDE = Cµ * K^2 / E, with Cµ = 0.09

K_EPSILON_INLET = 0.008
E_EPSILON_INLET = 0.054
C_MU = 0.09
SA_INLET_SCALE = 0.35
NU_TILDE_INLET = SA_INLET_SCALE * C_MU * (K_EPSILON_INLET ** 2) / E_EPSILON_INLET

# NU_TILDE_INLET = max(NU_TILDE_INLET, 3 * ν)
NU_TILDE_INLET = max(NU_TILDE_INLET, 3.0 * 8.9e-7)



# Initial conditions
initial_conditions = {
    'U': (0.0, 0.0, 0.0), 
    'P': 0.0,
    'NU_TILDE': NU_TILDE_INLET 
}

# Boundary conditions
boundary_conditions = {
    'INFLOW':{
        'U': (0.0, -1.42, 0.0), 
        'P': None, 
        'NU_TILDE': NU_TILDE_INLET
    },
    'OUTFLOW':{
        'U': None,
        'P': 0.0,
        'NU_TILDE': None
    },
    'WALLS':{
        'U': (0.0, 0.0, 0.0),
        'P': None,
        'NU_TILDE': 0.0
    }
}

# Physical quantities
physical_prm = {
    'VISCOSITY': 8.9e-7,  
    'FORCE': (0.0, 0.0, 0.0)
}

# Reynolds number:
# Re = U * D / ν = 1.42 * 0.028 / 8.9e-7 ≈ 4.5 × 10^4


# Simulation parameters for SA model
simulation_prm_SA = {
    'QUADRATURE_DEGREE': 2,
    'MAX_ITERATIONS': 9000,
    'TOLERANCE': 1e-6,
    'CFL_RELAXATION': 0.1,
    'U_RELAXATION_FACTOR': 0.7, # 'U_RELAXATION_FACTOR': 1.0
    'NUT_RELAXATION_FACTOR': 0.7, # 'NUT_RELAXATION_FACTOR': 1.0
    'STEP_SIZE': 5e-4,
    'MIN_STEP_SIZE': 1e-6,
    'MAX_STEP_SIZE': 1e-3 
}

# Specify where results are saved for SA model
saving_directory_SA = {
    'PVD_FILES': 'Results/U-Bend_SA/PVD files/',
    'H5_FILES':  'Results/U-Bend_SA/H5 files/',
    'RESIDUALS': 'Results/U-Bend_SA/Residual files/'
}

# Specify what to do after simulation
post_processing = {
    'PLOT': True,
    'SAVE': True,
}


#docker run -ti \
#    -v $(pwd):/home/fenics/shared \
#    -w /home/fenics/shared \
#    quay.io/fenicsproject/stable:current


#cd ~/shared

#ls 

#python3 UBendSimulation_SpalartAllmaras.py
