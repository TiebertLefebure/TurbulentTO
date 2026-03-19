import os

# File paths for mesh and boundary data
# Fine_WallRefinement: inflation mesh with first layer ≈ 1.2e-5 m (y+ ≈ 1).
# This is required for wall-resolved SA — the Ansys_Inflation mesh has
# first_layer ≈ 2.6e-4 m (y+ ≈ 11), which is too coarse for SA to capture
# the viscous sublayer correctly and is the main cause of profile mismatch.
# Generate via: cd Meshes/U-Bend/Fine_WallRefinement && python3 u_tube_gmsh.py && python3 gmsh_to_xdmf.py
mesh_files = {
    'MESH_DIRECTORY': 'Meshes/U-Bend/Fine_WallRefinement/mesh.xdmf',
    'FACET_DIRECTORY': 'Meshes/U-Bend/Fine_WallRefinement/facet.xdmf'
}


def _infer_mesh_label_from_path(path):
    parts = os.path.normpath(path).split(os.sep)
    for label in ('Coarse', 'Medium', 'Fine'):
        if label in parts:
            return label
    # Fallback to parent folder name for custom mesh variants.
    return parts[-2] if len(parts) >= 2 else 'UnknownMesh'


UBEND_SA_MESH_LABEL = _infer_mesh_label_from_path(mesh_files['MESH_DIRECTORY'])
UBEND_SA_RESULTS_ROOT = f'Results/U-Bend_SA/{UBEND_SA_MESH_LABEL}'
UBEND_SA_STEADY_RESULTS_ROOT = f'{UBEND_SA_RESULTS_ROOT}_Steady'
# Warm-start source can be the same mesh (default) or another U-bend mesh label
# (e.g. 'Coarse' to accelerate a 'Medium' run via interpolation/projection).
# Change 'UBEND_SA_WARM_START_LABEL' to desired mesh label.
UBEND_SA_WARM_START_LABEL = 'Fine_WallRefinement'
UBEND_SA_WARM_START_ROOT = f'Results/U-Bend_SA/{UBEND_SA_WARM_START_LABEL}'
UBEND_SA_WARM_START_MESH_XDMF = f'Meshes/U-Bend/{UBEND_SA_WARM_START_LABEL}/mesh.xdmf'
UBEND_SA_WARM_START_FACET_XDMF = f'Meshes/U-Bend/{UBEND_SA_WARM_START_LABEL}/facet.xdmf'

# Specify type of boundaries
boundary_markers = {
    'INFLOW': [2],
    'OUTFLOW': [3],
    'WALLS': [4]
}

# ------------------------
# 2D U-bend
# ------------------------

# pipe radius R_PIPE = 14 mm
# pipe diameter D_PIPE = 2 * R_PIPE = 28 mm
# radius of curvature R_c = 125 mm
# straight section length H_LEG = 30 * D_PIPE = 840 mm

# hydraulic diameter D_h = 2 * R_PIPE = 28 mm

# inlet bulk velocity U_ref = 1.42 m/s

# kinematic viscosity ν = 8.9e-7 m^2/s

# Spalart-Allmaras inlet estimate from ANSYS inlet specification:
# turbulence intensity I = 5% and hydraulic diameter D_h = 28 mm.
# Standard auxiliary relations (also used to derive k-epsilon inlet values):
#   k = 1.5 * (U * I)^2
#   epsilon = C_mu^(3/4) * k^(3/2) / l,   l = 0.07 * D_h
# Then map to SA with nu_tilde ~= nu_t = C_mu * k^2 / epsilon
# (for positive chi, f_v1 is close to 1).

C_MU = 0.09
TURBULENCE_INTENSITY_INLET = 0.05
HYDRAULIC_DIAMETER_INLET = 0.028  # [m]
INLET_BULK_VELOCITY = 1.42        # [m/s]
TURBULENCE_LENGTH_SCALE_INLET = 0.07 * HYDRAULIC_DIAMETER_INLET

K_EPSILON_INLET = 1.5 * (INLET_BULK_VELOCITY * TURBULENCE_INTENSITY_INLET) ** 2
E_EPSILON_INLET = (C_MU ** 0.75) * (K_EPSILON_INLET ** 1.5) / TURBULENCE_LENGTH_SCALE_INLET
NU_TILDE_INLET = C_MU * (K_EPSILON_INLET ** 2) / E_EPSILON_INLET


# Initial conditions
initial_conditions = {
    'U': (0.0, 0.0),
    'P': 0.0,
    'NU_TILDE': NU_TILDE_INLET 
}

# Boundary conditions
boundary_conditions = {
    'INFLOW':{
        'U': (0.0, -1.42),
        'P': None, 
        'NU_TILDE': NU_TILDE_INLET
    },
    'OUTFLOW':{
        'U': None,
        'P': 0.0,
        'NU_TILDE': None
    },
    'WALLS':{
        'U': (0.0, 0.0),
        'P': None,
        'NU_TILDE': 0.0
    }
}

# Physical quantities
physical_prm = {
    'VISCOSITY': 8.9e-7, # kinematic viscosity ν (m^2/s)
    'FORCE': (0.0, 0.0)
}

# ---------------------------------------------------------------------------------
# Reynolds number: Re = U_ref * D_h / ν = 1.42 * 0.028 / 8.9e-7 ≈ 4.5 × 10^4
# ---------------------------------------------------------------------------------


# Simulation parameters for SA model
simulation_prm_SA = {
    # Degree 4 is sufficient for the nonlinear SA terms (chi^3, f_w) on CG2/CG1
    # elements and is noticeably faster than degree 6 without loss of accuracy.
    'QUADRATURE_DEGREE': 4,
    'MAX_ITERATIONS': 9000,

    # ---------------
    # Tolerances
    # ---------------
    'TOLERANCE': 1e-6,
    # Optional field-specific tolerances (defaults fall back to TOLERANCE if omitted).
    # Keep U/p tolerances stricter for mesh-comparison runs; the medium mesh can
    # otherwise stop early and look artificially over-diffusive at the probe line.
    'TOLERANCE_U': 1e-5,
    'TOLERANCE_P': 1e-5,
    'TOLERANCE_NU_TILDE': 1e-6,

    # ----------------------
    # Relaxation factors
    # ----------------------
    'U_RELAXATION_FACTOR': 0.7,
    'NUT_RELAXATION_FACTOR': 0.3,
    # Used by steady Picard solver (UBendSimulation_SpalartAllmaras_Steady.py).
    'PICARD_RELAXATION': 0.2,

    # ---------------------
    # Linear solver
    # --------------------- 
    # Separate linear-solver controls for NS and SA.
    # NS (pressure-correction blocks F1, F2, F3):
    # Keep NS_LINEAR_SOLVER='mumps' + NS_LINEAR_PRECONDITIONER=None for robustness
    # For lower RAM try: NS_LINEAR_SOLVER='bicgstab' + NS_LINEAR_PRECONDITIONER='hypre_amg' (or 'ilu').
    'NS_LINEAR_SOLVER': 'gmres',
    'NS_LINEAR_PRECONDITIONER': 'hypre_amg',
    # SA (nu_tilde transport equation):
    # SA_LINEAR_SOLVER='default' + SA_LINEAR_PRECONDITIONER='default' uses DOLFIN backend default solve(A,x,b).
    # For explicit control, set e.g. SA_LINEAR_SOLVER='bicgstab' + SA_LINEAR_PRECONDITIONER='hypre_amg'.
    'SA_LINEAR_SOLVER': 'default',
    'SA_LINEAR_PRECONDITIONER': 'default',

    # ----------------------------
    # SA iterations parameters
    # ----------------------------
    # Number of SA (nu_tilde) solves per outer NS/SA coupling iteration.
    # 2 inner iterations per outer step gives a good balance: the second solve
    # picks up the nonlinear coefficient update from the first, which is the
    # largest correction, without the overhead of 5 inner solves that the
    # original staged scheme used.
    # Setting all three to the same value disables staged reduction entirely.
    'SA_INNER_ITERS': 2,
    'SA_INNER_ITERS_MID': 2,   # same as MAX → no stage-1 reduction
    'SA_INNER_ITERS_MIN': 2,   # same as MAX → no stage-2 reduction
    'SA_INNER_ITERS_REDUCE_NU_TILDE_FACTOR': 0.1,
    'SA_INNER_ITERS_REDUCE_NU_TILDE_FACTOR_FINAL': 0.01,
    'SA_INNER_ITERS_REDUCE_STREAK': 20,
    'SA_INNER_ITERS_REDUCE_STREAK_FINAL': 10,

    # ------------------------
    # Wall-distance model
    # ------------------------
    # 'OriginalEikonal'    -> original smoothed Eikonal distance
    # 'RelaxedWallEikonal' -> Yoon 2016 relaxed wall equation (Eq. 19, reciprocal-distance form)
    'WALL_DISTANCE_METHOD': 'OriginalEikonal',
    'WALL_DISTANCE_EIKONAL_RELAXATION': 0.01,
    # Yoon 2016 Eq.(19) parameters (used only when WALL_DISTANCE_METHO = 'RelaxedWallEikonal')
    'WALL_DISTANCE_YOON_SIGMA_W': 0.1,      # sigma_w < 0.5; Yoon 2016 uses sigma_w = 0.1
    'WALL_DISTANCE_YOON_G0': 20.0,          # [1/m], reference reciprocal distance (Eq. 15)
    'WALL_DISTANCE_YOON_G_FLOOR': 1.0e-12,  # numerical floor for G
    
    # ------------------------------------------------
    # Time-stepping parameters (using CFL condition)
    # ------------------------------------------------
    'CFL_RELAXATION': 0.1, # maximum CFL number
    'STEP_SIZE': 5e-4, # initial time step
    'MIN_STEP_SIZE': 1e-6,
    'MAX_STEP_SIZE': 1e-3,

    # -------------------------
    # Live monitoring 
    # -------------------------
    # Runtime output cadence (for live monitoring while the solver runs).
    # RUNTIME_WRITE_INTERVAL = 0 disables runtime snapshots; set e.g. 20 to write every 20 iterations.
    'RUNTIME_WRITE_INTERVAL': 50,
    'RUNTIME_WRITE_PVD': True,
    'RUNTIME_WRITE_RESIDUALS': False,

    # -------------------------
    # Warm-start
    # -------------------------
    # Optional warm-start from saved H5 fields (same mesh/function spaces required).
    # Set to False to disable warm-start.
    'WARM_START_ENABLED': False,
    'WARM_START_SOURCE_MESH_XDMF': UBEND_SA_WARM_START_MESH_XDMF,
    'WARM_START_SOURCE_FACET_XDMF': UBEND_SA_WARM_START_FACET_XDMF,
    # Velocity warm-start
    'WARM_START_U_H5': f'{UBEND_SA_WARM_START_ROOT}/H5 files/u.h5',
    # Pressure warm-start
    'WARM_START_P_H5': f'{UBEND_SA_WARM_START_ROOT}/H5 files/p.h5',
    # Modified turbulent viscosity warm-start
    'WARM_START_NU_TILDE_H5': f'{UBEND_SA_WARM_START_ROOT}/H5 files/nu_tilde.h5',

}

# Specify where results are saved for SA model
saving_directory_SA = {
    'PVD_FILES': f'{UBEND_SA_RESULTS_ROOT}/PVD files/',
    'H5_FILES':  f'{UBEND_SA_RESULTS_ROOT}/H5 files/',
    'RESIDUALS': f'{UBEND_SA_RESULTS_ROOT}/Residual files/'
}

# Separate output folders for strict steady SA runs.
saving_directory_SA_STEADY = {
    'PVD_FILES': f'{UBEND_SA_STEADY_RESULTS_ROOT}/PVD files/',
    'H5_FILES':  f'{UBEND_SA_STEADY_RESULTS_ROOT}/H5 files/',
    'RESIDUALS': f'{UBEND_SA_STEADY_RESULTS_ROOT}/Residual files/'
}

# Specify what to do after simulation
post_processing = {
    'PLOT': True,
    'SAVE': True,
}
