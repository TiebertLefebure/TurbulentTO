from Configs_SemiFrozen.Config_UBendDilgen_SemiFrozen import *  # noqa: F401,F403


# The TurbulentTO_Full wrapper is kept for naming continuity. With Dilgen's
# Poisson wall-distance mode it resolves to the same 3-field full-turbulence
# SA adjoint as the SemiFrozen entrypoint.
RESULTS_ROOT_NAME = "Results_Full/Results_UBendDilgen_TurbulentTO_Full"
