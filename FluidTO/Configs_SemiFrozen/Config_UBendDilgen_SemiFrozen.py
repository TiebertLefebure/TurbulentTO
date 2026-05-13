from Configs_Frozen.Config_UBendDilgen_Frozen import *  # noqa: F401,F403


# Dilgen's U-bend case uses a Poisson wall-distance model, so there is no
# reciprocal G state and no G* adjoint. The SemiFrozen entrypoint includes the
# SA working variable in the coupled state/adjoint: (u*, p*, nu_tilde*).
RESULTS_ROOT_NAME = "Results_SemiFrozen/Results_UBendDilgen_SemiFrozen"
