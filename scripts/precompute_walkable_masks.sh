#!/usr/bin/env bash
# Precompute CAT-Seg walkable masks with the catseg conda environment.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
CATSEG_PYTHON="${CATSEG_PYTHON:-${HOME}/anaconda3/envs/catseg/bin/python}"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
if [[ ! -x "${CATSEG_PYTHON}" ]]; then
  echo "catseg python not found: ${CATSEG_PYTHON} (set CATSEG_PYTHON)" >&2
  exit 2
fi
exec "${CATSEG_PYTHON}" -m memory_nav.segmentation.precompute_masks "$@"
