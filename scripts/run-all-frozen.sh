#!/usr/bin/env bash
# Alle Configs_Frozen achter elkaar — zelfde commando als Tijbert, per case.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUN="$ROOT/scripts/run-frozen.sh"
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
  "$RUN" "$c"
  echo "========== DONE  $c =========="
done
