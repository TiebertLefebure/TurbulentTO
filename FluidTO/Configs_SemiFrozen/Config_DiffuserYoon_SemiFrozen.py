from Configs_Full.Config_DiffuserYoon_Full import *  # noqa: F401,F403


# Yoon's diffuser uses the reciprocal wall-distance state G. The SemiFrozen
# entrypoint keeps G as an external wall-distance update, so the adjoint state
# is (u*, p*, nu_tilde*); the Full entrypoint adds G*.
RESULTS_ROOT_NAME = "Results_SemiFrozen/Results_DiffuserYoon_SemiFrozen"
