#!/usr/bin/env bash
set -euo pipefail

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
  shift
fi

CONFIG="${1:-batch_datasets.json}"
DATASETS="${2:-}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONIOENCODING="${PYTHONIOENCODING:-utf-8}"

CMD=(python run_batch.py --config "${CONFIG}")
if [[ -n "${DATASETS}" ]]; then
  CMD+=(--datasets "${DATASETS}")
fi
if [[ "${DRY_RUN}" == "1" ]]; then
  CMD+=(--dry_run)
fi

"${CMD[@]}"
