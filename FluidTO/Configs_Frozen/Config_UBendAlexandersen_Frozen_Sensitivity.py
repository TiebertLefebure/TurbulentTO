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

# Keep the one-shot Jp sensitivity check on the pressure-objective wall/SA
# analogue values. This mirrors psi_max=1000 and P_con=4 instead of the
# stronger JD continuation ramp.
SA_NU_TILDE_PENALTY_ALPHA = 1.0e3
SA_NU_TILDE_PENALTY_ALPHA_SCHEDULE = [1.0e3]
SA_NU_TILDE_PENALTY_N = 4.0
SA_WALL_PENALTY_ALPHA = 1.0e3
SA_WALL_PENALTY_ALPHA_SCHEDULE = [1.0e3]
SA_WALL_PENALTY_N = 4.0

# The sensitivity run starts from the uniform initial design. In that state
# the U-bend Jp forward solve reaches a repeatable small residual plateau near
# full convection, then PETSc's line search reports divergence just above the
# production accept gate. Use finer high-convection steps and an acceptance
# band sized for finite-difference verification rather than topology updates.
FORWARD_SNES_STARTUP_CONVECTION_SCHEDULE = [
    {"convection_weight": 0.00, "max_iters": 80, "atol": 2.0e-7, "accept_norm": 2.0e-4, "accept_nonconverged": True},
    {"convection_weight": 0.20, "max_iters": 100, "atol": 1.5e-7, "accept_norm": 1.5e-4, "accept_nonconverged": True},
    {"convection_weight": 0.40, "max_iters": 120, "atol": 1.0e-7, "accept_norm": 1.2e-4, "accept_nonconverged": True},
    {"convection_weight": 0.60, "max_iters": 140, "atol": 8.0e-8, "accept_norm": 1.0e-4, "accept_nonconverged": True},
    {"convection_weight": 0.80, "max_iters": 170, "atol": 5.0e-8, "accept_norm": 1.0e-4, "accept_nonconverged": True},
    {"convection_weight": 0.90, "max_iters": 190, "atol": 2.5e-8, "accept_norm": 1.25e-4, "accept_nonconverged": True},
    {"convection_weight": 0.95, "max_iters": 220, "atol": 2.5e-8, "accept_norm": 1.5e-4, "accept_nonconverged": True},
    {"convection_weight": 1.00, "max_iters": 260, "atol": 1.0e-8, "accept_norm": 2.0e-4, "accept_nonconverged": True},
]
FORWARD_SNES_CONVECTION_SCHEDULE = FORWARD_SNES_STARTUP_CONVECTION_SCHEDULE
FORWARD_SNES_MAX_ACCEPTED_ABSOLUTE_RESIDUAL = 5.0e-4

RUN_TAYLOR_SENSITIVITY_CHECKS = True
TAYLOR_CHECK_STEPS = (1.0e-3, 3.0e-4, 1.0e-4, 3.0e-5)
TAYLOR_CHECK_SEED = 29

SENSITIVITY_VERIFICATION_MODE_NAME = "frozen-turbulence-Jp"
SENSITIVITY_VERIFICATION_DERIVATIVE_COLUMN = "FrozenTurbulenceJp"
STOP_AFTER_FINITE_DIFFERENCE_CHECKS = True
