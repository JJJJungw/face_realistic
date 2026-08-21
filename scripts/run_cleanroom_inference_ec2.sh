#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

TRAIN_PYTHON="${PROJECT_ROOT}/.venv-train/bin/python"
if [[ ! -x "${TRAIN_PYTHON}" ]]; then
  echo "학습 환경이 없습니다. 먼저 bash scripts/setup_training_ec2.sh를 실행하세요." >&2
  exit 1
fi

"${TRAIN_PYTHON}" -m face_realistic.modeling.infer \
  --project-root "${PROJECT_ROOT}" \
  --input "${INFERENCE_INPUT:-${PROJECT_ROOT}/assets/source/swap2.mp4}" \
  --checkpoint "${INFERENCE_CHECKPOINT:-${PROJECT_ROOT}/outputs/training/person_01_2k/checkpoint.pt}" \
  --output "${INFERENCE_OUTPUT:-${PROJECT_ROOT}/outputs/cleanroom_swap_3s.mp4}" \
  --max-seconds "${INFERENCE_MAX_SECONDS:-3}" \
  --motion-smoothing "${INFERENCE_MOTION_SMOOTHING:-0.65}" \
  --device "${INFERENCE_DEVICE:-cuda}"
