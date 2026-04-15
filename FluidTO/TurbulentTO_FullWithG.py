import os


# Thin entrypoint for the 4-field full solver. The main implementation stays in
# TurbulentTO_Full.py; this wrapper only activates inclusion of the G-state.
os.environ["TURBULENTTO_FULL_INCLUDE_G_STATE"] = "1"

import TurbulentTO_Full  # noqa: F401
