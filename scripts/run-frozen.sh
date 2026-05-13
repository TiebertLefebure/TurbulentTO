#!/usr/bin/env bash
# Zelfde als Tijbert: vanuit FluidTO → python3 TurbulentTO_Frozen.py --config …
# Voorbeeld vanaf repo-root:
#   ./scripts/run-frozen.sh Configs_Frozen/Config_DiffuserYoon_Frozen.py

set -euo pipefail
FLUIDTO="$(cd "$(dirname "$0")/../FluidTO" && pwd)"
cd "$FLUIDTO"
config_arg="${1:?Gebruik: $0 Configs_Frozen/<Config_…_Frozen.py>}"
exec python3 TurbulentTO_Frozen.py --config "$config_arg"
