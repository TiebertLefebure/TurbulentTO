import os


# Thin entrypoint for the 4-field Full adjoint. The shared implementation
# stays in TurbulentTO_SemiFrozen.py; this wrapper only activates inclusion of
# the G-state in the primal/adjoint system.
os.environ["TURBULENTTO_INCLUDE_G_STATE"] = "1"

import TurbulentTO_SemiFrozen  # noqa: F401
