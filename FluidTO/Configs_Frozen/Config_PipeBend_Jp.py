import os as _os
import sys as _sys

_CONFIG_DIR = _os.path.dirname(_os.path.abspath(__file__))
_REPO_ROOT = _os.path.dirname(_CONFIG_DIR)
if _REPO_ROOT not in _sys.path:
    _sys.path.insert(0, _REPO_ROOT)

from Configs_Frozen.Config_PipeBendAlexandersen_Frozen import *  # noqa: F401,F403


# ===================================================================
# Pressure-objective pipe-bend restart candidate.
#
# This variant is meant to recover the Alexandersen-like J_p topology
# without the large objective jumps seen in the old run from commit
# 8fedc94.  It keeps the current robust flow/SA solve settings, but
# makes the topology continuation much less aggressive than the old
# q=[1/150, 1/75, 1/30, 1/15], beta=[4, 6, 9, 13],
# move=[0.05, 0.04, 0.03, 0.02] schedule.
# ===================================================================

RESUME_OPTIMIZATION = True

VOL_FRAC = 0.25
OBJECTIVE_TYPE = "average_inlet_pressure"
OBJECTIVE_CONVERGENCE_TOL = 2.0e-4
OBJECTIVE_STREAK_TO_STOP = 8
INITIAL_DENSITY_VALUE = VOL_FRAC
INITIAL_DENSITY_MATCH_FILTERED_VOLUME = True

# Keep the same solid impedance as the old pressure run.  This is one of the
# differences most likely to affect whether the J_p topology follows the
# Alexandersen/Bayat bend instead of becoming overly locked by the Brinkman term.
ALPHA_SOLID = 100.0

# Same final pressure-objective penalization as the old run, but introduced
# through smaller continuation jumps.
PAPER_Q_ALPHA_SCHEDULE = [150.0, 100.0, 75.0, 45.0, 30.0, 15.0]
Q_PENAL_SCHEDULE = [1.0 / q_alpha for q_alpha in PAPER_Q_ALPHA_SCHEDULE]
BETA_PROJ_SCHEDULE = [2.0, 3.0, 4.5, 6.5, 9.0, 13.0]
MOVE_LIMIT_SCHEDULE = [0.015, 0.012, 0.009, 0.006, 0.004, 0.003]
MAX_INNER_ITERATIONS_SCHEDULE = [45, 45, 50, 60, 70, 90]
MAX_INNER_ITERATIONS = MAX_INNER_ITERATIONS_SCHEDULE[0]
BETA_PROJ_VALUE = BETA_PROJ_SCHEDULE[0]

# Current SA settings are more stable than the old J_p run.  Make them explicit
# here because the old uploaded config disabled SUPG and used stronger SA
# relaxation, both of which can expose cell-scale nu_tilde oscillations.
SA_SUPG_STABILIZATION = True
SA_SUPG_TAU_SCALE = 1.0
SA_PSEUDO_TIME_STABILIZATION = False

PICARD_STEPS = 5
TURBULENCE_RELAXATION = 0.04

SA_NU_TILDE_PENALTY_ALPHA = 1.0e5
SA_NU_TILDE_PENALTY_ALPHA_SCHEDULE = [1.0e3, 3.0e3, 1.0e4, 3.0e4, 1.0e5, 1.0e5]
SA_NU_TILDE_PENALTY_N = 3.0
SA_NU_TILDE_PENALTY_INTERPOLATION = "power"

SA_WALL_SIGMA = 0.1
SA_WALL_PENALTY_ALPHA = 1.0e5
SA_WALL_PENALTY_ALPHA_SCHEDULE = [1.0e3, 3.0e3, 1.0e4, 3.0e4, 1.0e5, 1.0e5]
SA_WALL_PENALTY_N = 3.0
SA_WALL_PENALTY_INTERPOLATION = "power"

# Keep the current forward solver safeguards active.  They cannot remove all
# nonconvexity from J_p, but they do prevent one bad density step from becoming
# a failed flow solve.
FORWARD_RETRY_BACKTRACK_DESIGN = True
FORWARD_RETRY_BACKTRACK_FACTORS = [0.5, 0.25, 0.10, 0.05]

SAVE_SA_CLIPPING_DIAGNOSTICS = True
SAVE_DF0DX_VECTOR = False
SAVE_IPCS_RESIDUAL_PLOTS = False
SAVE_IPCS_RESIDUAL_SVGS = False

RESULTS_ROOT_BASE_NAME = "Results_Frozen/Results_PipeBend_Jp"
RESULTS_ROOT_NAME = RESULTS_ROOT_BASE_NAME
