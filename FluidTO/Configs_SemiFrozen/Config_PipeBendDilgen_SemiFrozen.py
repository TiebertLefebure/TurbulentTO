from Configs_Frozen.Config_PipeBendDilgen_Frozen import *  # noqa: F401,F403


# Dilgen's pipe-bend case uses a Poisson wall-distance model, so there is no
# reciprocal G state and no G* adjoint. The SemiFrozen entrypoint includes the
# SA working variable in the coupled state/adjoint: (u*, p*, nu_tilde*).
RESULTS_ROOT_NAME = DILGEN_SEMIFROZEN_RESULTS_ROOT_NAME
SENSITIVITY_VERIFICATION_MODE_NAME = "semi-frozen-SA"
SENSITIVITY_VERIFICATION_DERIVATIVE_COLUMN = "AdjointSA"
