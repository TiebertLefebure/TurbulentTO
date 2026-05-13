#!/usr/bin/env bash
# Draait alle Configs_Frozen cases achter elkaar (TurbulentTO_Frozen.py).
# Repo-root = parent van deze scriptmap.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/FluidTO"
configs=(
  Configs_Frozen/Config_DiffuserYoon_Frozen.py
  Configs_Frozen/Config_DoublePipeBorrvall_Frozen.py
  Configs_Frozen/Config_DoublePipeYoon_Frozen.py
  Configs_Frozen/Config_PipeBendAlexandersen_Frozen.py
  Configs_Frozen/Config_PipeBendDilgen_Frozen.py
  Configs_Frozen/Config_UBendAlexandersen_Frozen.py
  Configs_Frozen/Config_UBendDilgen_Frozen.py
)
for c in "${configs[@]}"; do
  echo "========== START $c =========="
  python3 TurbulentTO_Frozen.py --config "$c"
  echo "========== DONE  $c =========="
done
