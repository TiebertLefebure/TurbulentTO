from Configs_Frozen.Config_PipeBendAlexandersen_Frozen_Jp_Old import *


# One-shot sensitivity verification for the Alexandersen pipe-bend case.
#
# This config intentionally inherits the J_p retry path used for the current
# Bayat bend cases: reciprocal_penalized wall distance, design wall-density
# source, and power penalization. The finite-difference perturbations keep the
# SA eddy-viscosity and reciprocal wall-distance fields frozen, matching the
# frozen-turbulence adjoint derivative being checked.

RESULTS_ROOT_NAME = "Results_Frozen/Results_PipeBendAlexandersen_Frozen_Sensitivity"

RUN_FINITE_DIFFERENCE_CHECKS = True
FINITE_DIFFERENCE_CHECK_ITERATIONS = (0,)
FINITE_DIFFERENCE_CHECK_STEP = 1.0e-6
FINITE_DIFFERENCE_CHECK_SAMPLES = 5
FINITE_DIFFERENCE_CHECK_SEED = 13
FINITE_DIFFERENCE_CHECK_CLIP_TO_BOUNDS = True
FINITE_DIFFERENCE_CHECK_UPDATED_TURBULENCE = False

RUN_TAYLOR_SENSITIVITY_CHECKS = True
TAYLOR_CHECK_STEPS = (1.0e-3, 3.0e-4, 1.0e-4, 3.0e-5)
TAYLOR_CHECK_SEED = 29

SENSITIVITY_VERIFICATION_MODE_NAME = "frozen-turbulence"
SENSITIVITY_VERIFICATION_DERIVATIVE_COLUMN = "FrozenTurbulence"
STOP_AFTER_FINITE_DIFFERENCE_CHECKS = True
