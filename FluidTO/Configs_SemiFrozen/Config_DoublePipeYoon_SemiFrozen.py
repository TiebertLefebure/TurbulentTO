from Configs_Frozen.Config_DoublePipeYoon_Frozen import *  # noqa: F401,F403


# Yoon's double pipe uses the reciprocal wall-distance state G. The
# SemiFrozen entrypoint solves the primal state in (u, p, nu_tilde) while
# keeping G as an external wall-distance update.
STATE_SOLVE_METHOD = "newtontr"
STATE_LINE_SEARCH = "bt"
STATE_RTOL = FORWARD_SNES_RTOL
STATE_ATOL = FORWARD_SNES_ATOL
STATE_MAX_ITERS = 180
STATE_ERROR_ON_NONCONVERGENCE = False
STATE_ACCEPTED_RESIDUAL_FACTOR = 1.0
STATE_ACCEPT_NONCONVERGED_WITH_ACCEPT_NORM = True
STATE_ACCEPT_INITIAL_IF_WITHIN_ACCEPT_NORM = True
STATE_INITIAL_SA_SWEEPS = PICARD_STEPS
STATE_INITIAL_SA_RELAXATION = TURBULENCE_RELAXATION
STATE_TURBULENCE_COUPLING_SCHEDULE = [
    {"convection_weight": 0.00, "weight": 0.00, "max_iters": 120, "atol": 8.0e-4, "accept_norm": 2.0e-3},
    {"convection_weight": 0.25, "weight": 0.00, "max_iters": 150, "atol": 6.0e-4, "accept_norm": 4.0e-3, "accept_growth": 1.25},
    {"convection_weight": 0.50, "weight": 0.15, "max_iters": 180, "atol": 5.0e-4, "accept_norm": 6.0e-3, "accept_growth": 1.25},
    {"convection_weight": 0.75, "weight": 0.50, "max_iters": 220, "atol": 4.0e-4, "accept_norm": 8.0e-3, "accept_growth": 1.25},
    {"convection_weight": 1.00, "weight": 0.75, "max_iters": 260, "atol": 3.0e-4, "accept_norm": 1.0e-2, "accept_growth": 1.25},
    {"convection_weight": 1.00, "weight": 1.00, "max_iters": 320},
]
STATE_RECOVERY_ATTEMPTS = [
    {
        "label": "current-iterate line-search retry",
        "method": "newtonls",
        "line_search": "bt",
        "max_iters": 240,
        "restart_with_stokes": False,
    },
    {
        "label": "Stokes rebuild line-search retry",
        "method": "newtonls",
        "line_search": "bt",
        "max_iters": 300,
        "restart_with_stokes": True,
    },
]

RESULTS_ROOT_NAME = "Results_SemiFrozen/Results_DoublePipeYoon_SemiFrozen"
