
# File paths for mesh and boundary data
mesh_files = {
    'MESH_DIRECTORY': 'Meshes/BackStep/Fine/mesh.xdmf',
    'FACET_DIRECTORY': 'Meshes/BackStep/Fine/facet.xdmf'
}

# Specify type of boundaries
boundary_markers = {
    'INFLOW': [4],
    'OUTFLOW': [2],
    'WALLS': [1, 3],
    'SYMMETRY': [5]
}

# Initial conditions
initial_conditions = {
    'U': (0.0, 0.0),
    'P': 0.0,
    'K': 1.73,
    'E': 1.46
}

# Boundary conditions
boundary_conditions = {
    'INFLOW':{
        'U': (25.0, 0.0),
        'P': None,
        'K': 1.73,
        'E': 1.46
    },
    'OUTFLOW':{
        'U': None,
        'P': 0.0,
        'K': None,
        'E': None
    },
    'WALLS':{
        'U': (0.0, 0.0),
        'P': None,
        'K': 0.0,
        'E': None
    },
    'SYMMETRY':{
        'U': 0.0,
        'P': None,
        'K': 1.73,
        'E': 1.46
    }
}

# Physical quantities
physical_prm = {
    'VISCOSITY': 0.000181818,
    'FORCE': (0.0, 0.0)
}

# Reynolds-number reference lengths from the actual BackStep mesh markers (Fine mesh):
#   inflow marker [4]: x = -32.5, y in [0.0, 2.0]      -> inlet opening height H_in = 2.0
#   outflow marker [2]: x =  12.5, y in [-0.25, 2.0]   -> outlet height H_out = 2.25
#   step drop at x = -27.5: upstream lower boundary y=0.0 to downstream lower wall y=-0.25
#       -> step height h = 0.25
#
# With U_ref = 25.0 m/s and nu = 1.81818e-4 m^2/s:
#   Re_h      = U_ref * h     / nu ≈ 3.44e4   (common Backward-Facing Step definition)
#   Re_Hin    = U_ref * H_in  / nu ≈ 2.75e5
#   Re_Dh,in  = U_ref * Dh_in / nu ≈ 5.50e5   with Dh_in = 2 * H_in (2D channel convention)
# Use the same reference length as the benchmark/paper you compare against.

# -----------------------------------------------------------------------------------
# Reynolds number: Re_h = U_ref * h / ν = 25.0 * 0.25 / 0.000181818 = 3.44 x 10^4
# -----------------------------------------------------------------------------------

# Simulation parameters
simulation_prm = {
    'QUADRATURE_DEGREE': 2,
    'MAX_ITERATIONS': 3000,
    'TOLERANCE': 1e-6,
    'PICARD_RELAXATION': 0.1
}

# Specify where results are saved
saving_directory = {
    'PVD_FILES': 'Results/BackStep/PVD files/',
    'H5_FILES':  'Results/BackStep/H5 files/',
    'RESIDUALS': 'Results/BackStep/Residual files/'
}

# Specify what to do after simulation
post_processing = {
    'PLOT': True,
    'SAVE': True,
}
