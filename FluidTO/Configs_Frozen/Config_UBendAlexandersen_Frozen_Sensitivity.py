from Configs_Frozen.Config_UBendAlexandersen_Frozen import *


# One-shot sensitivity verification for the Alexandersen U-bend case.
#
# This config intentionally inherits the current pressure-objective U-bend
# setup, including the frozen-SA coupling, pressure outlet, reciprocal
# wall-distance treatment, and J_p average inlet-pressure objective. The
# finite-difference perturbations keep the SA working variable and reciprocal
# wall-distance field frozen, matching the frozen-turbulence adjoint derivative.

RESUME_OPTIMIZATION = False
RESULTS_ROOT_NAME = "Results_Frozen/Results_UBendAlexandersen_Frozen_Sensitivity"

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

SENSITIVITY_VERIFICATION_MODE_NAME = "frozen-turbulence-Jp"
SENSITIVITY_VERIFICATION_DERIVATIVE_COLUMN = "FrozenTurbulenceJp"
STOP_AFTER_FINITE_DIFFERENCE_CHECKS = True
