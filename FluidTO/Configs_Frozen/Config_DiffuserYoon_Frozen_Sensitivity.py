from Configs_Frozen.Config_DiffuserYoon_Frozen import *


# One-shot sensitivity verification for the Yoon diffuser case.
#
# This intentionally inherits the current diffuser setup, including the J_D
# dissipation objective, pressure-outlet default, reciprocal_penalized wall
# distance, design wall-density source, and power-law topology penalties. The
# finite-difference perturbations keep the SA working variable and reciprocal
# wall-distance field frozen, matching the frozen-turbulence adjoint derivative.

OUTLET_BC_TYPE = "pressure"
ENABLE_PRESSURE_PIN = True
RESUME_OPTIMIZATION = False
RESULTS_ROOT_NAME = "Results_Frozen/Results_DiffuserYoon_Frozen_PressureOutlet_Sensitivity"

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

SENSITIVITY_VERIFICATION_MODE_NAME = "frozen-turbulence-JD"
SENSITIVITY_VERIFICATION_DERIVATIVE_COLUMN = "FrozenTurbulenceJD"
STOP_AFTER_FINITE_DIFFERENCE_CHECKS = True
