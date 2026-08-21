#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAIN_PYTHON="${PROJECT_ROOT}/.venv-train/bin/python"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

if [[ ! -x "${TRAIN_PYTHON}" ]]; then
  echo "학습 환경이 없습니다. 먼저 bash scripts/setup_training_ec2.sh를 실행하세요." >&2
  exit 1
fi

"${TRAIN_PYTHON}" -m pytest -q \
  "${PROJECT_ROOT}/tests/test_v1_geometry.py" \
  "${PROJECT_ROOT}/tests/test_v1_model.py" \
  "${PROJECT_ROOT}/tests/test_v1_compositor.py" \
  "${PROJECT_ROOT}/tests/test_v1_smoke.py"

"${TRAIN_PYTHON}" -m face_realistic.modeling.v1.smoke \
  --device "${V1_SMOKE_DEVICE:-cuda}" \
  --image-size "${V1_SMOKE_IMAGE_SIZE:-256}" \
  --references "${V1_SMOKE_REFERENCES:-1}" \
  --warmup "${V1_SMOKE_WARMUP:-10}" \
  --iterations "${V1_SMOKE_ITERATIONS:-100}"
