#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTAINER_WORKDIR="/home/fenics/shared"

if [ "$#" -eq 0 ]; then
  exec docker run --rm -it \
    --platform linux/amd64 \
    -e HDF5_USE_FILE_LOCKING="${HDF5_USE_FILE_LOCKING:-FALSE}" \
    -e UBEND_RESTART_CHECKPOINT="${UBEND_RESTART_CHECKPOINT:-}" \
    -v "${SCRIPT_DIR}:${CONTAINER_WORKDIR}" \
    -w "${CONTAINER_WORKDIR}" \
    --entrypoint /bin/bash \
    quay.io/fenicsproject/stable:current \
    -l
fi

exec docker run --rm -it \
  --platform linux/amd64 \
  -e HDF5_USE_FILE_LOCKING="${HDF5_USE_FILE_LOCKING:-FALSE}" \
  -e UBEND_RESTART_CHECKPOINT="${UBEND_RESTART_CHECKPOINT:-}" \
  -v "${SCRIPT_DIR}:${CONTAINER_WORKDIR}" \
  -w "${CONTAINER_WORKDIR}" \
  --entrypoint /bin/bash \
  quay.io/fenicsproject/stable:current \
  -lc 'exec "$@"' bash "$@"
