#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTAINER_WORKDIR="/home/fenics/shared"
FENICS_IMAGE="${FENICS_IMAGE:-quay.io/fenicsproject/stable:current}"
DOCKER_PLATFORM="${DOCKER_PLATFORM:-linux/amd64}"
RESULTS_ROOT="${SCRIPT_DIR}/Results/BackStep_SA/Medium_Steady"
LOG_DIR="${RESULTS_ROOT}/Run logs"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/backstep_medium_verification_${TIMESTAMP}.log"

mkdir -p "${LOG_DIR}"

echo "BackStep Medium verification run"
echo "  Image: ${FENICS_IMAGE}"
echo "  Platform: ${DOCKER_PLATFORM}"
echo "  Results: ${RESULTS_ROOT}"
echo "  Log: ${LOG_FILE}"

docker run --rm \
  --platform "${DOCKER_PLATFORM}" \
  -e HDF5_USE_FILE_LOCKING="${HDF5_USE_FILE_LOCKING:-FALSE}" \
  -e PYTHONUNBUFFERED=1 \
  -v "${SCRIPT_DIR}:${CONTAINER_WORKDIR}" \
  -w "${CONTAINER_WORKDIR}" \
  --entrypoint /bin/bash \
  "${FENICS_IMAGE}" \
  -lc "python3 -u BackStepSimulation_SpalartAllmaras_Steady.py" 2>&1 | tee "${LOG_FILE}"

exit "${PIPESTATUS[0]}"
