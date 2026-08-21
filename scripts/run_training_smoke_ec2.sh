#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

TRAIN_PYTHON="${PROJECT_ROOT}/.venv-train/bin/python"
if [[ ! -x "${TRAIN_PYTHON}" ]]; then
  echo "학습 환경이 없습니다. 먼저 bash scripts/setup_training_ec2.sh를 실행하세요." >&2
  exit 1
fi

CONDITION_MODE="${TRAIN_CONDITION_MODE:-motion_only}"
DEFAULT_OUTPUT="${PROJECT_ROOT}/outputs/training/person_01_${CONDITION_MODE}"

"${TRAIN_PYTHON}" -m face_realistic.modeling.train \
  --project-root "${PROJECT_ROOT}" \
  --manifest "${TRAIN_MANIFEST:-${PROJECT_ROOT}/outputs/identity_registry/person_01/manifest.json}" \
  --output-dir "${TRAIN_OUTPUT_DIR:-${DEFAULT_OUTPUT}}" \
  --image-size "${TRAIN_IMAGE_SIZE:-256}" \
  --condition-mode "${CONDITION_MODE}" \
  --target-bottleneck "${TRAIN_TARGET_BOTTLENECK:-16}" \
  --base-channels "${TRAIN_BASE_CHANNELS:-32}" \
  --max-channels "${TRAIN_MAX_CHANNELS:-256}" \
  --steps "${TRAIN_STEPS:-10}" \
  --batch-size "${TRAIN_BATCH_SIZE:-2}" \
  --device "${TRAIN_DEVICE:-cuda}"
