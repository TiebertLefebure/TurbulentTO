#!/usr/bin/env bash
# Run a frozen-SA optimization config from the FluidTO working directory.
# Example from the repository root:
#   ./scripts/run-frozen.sh Configs_Frozen/Config_DiffuserYoon_Frozen.py

set -euo pipefail
FLUIDTO="$(cd "$(dirname "$0")/../FluidTO" && pwd)"
cd "$FLUIDTO"
config_arg="${1:?Usage: $0 Configs_Frozen/<Config_Frozen.py>}"
exec python3 TurbulentTO_Frozen.py --config "$config_arg"
