#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

python3 -m face_realistic.dataset.builder \
  --identity-id "${DATASET_IDENTITY_ID:-person_01}" \
  --identity-dir "${DATASET_IDENTITY_DIR:-${PROJECT_ROOT}/assets/identities/person_01}" \
  --source-image "${DATASET_SOURCE_IMAGE:-${PROJECT_ROOT}/assets/identities/person_01/front_neutral.png}" \
  --driving-dir "${DATASET_DRIVING_DIR:-${PROJECT_ROOT}/assets/source}" \
  --output-root "${DATASET_OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/datasets}" \
  --liveportrait-dir "${LIVEPORTRAIT_DIR:-${PROJECT_ROOT}/third_party/LivePortrait}" \
  --max-seconds "${DATASET_MAX_SECONDS:-0}" \
  --minimum-frames "${DATASET_MINIMUM_FRAMES:-750}" \
  --device-id "${LIVEPORTRAIT_DEVICE_ID:-0}" \
  --driving-multiplier "${LIVEPORTRAIT_DRIVING_MULTIPLIER:-1.0}" \
  --animation-region "${LIVEPORTRAIT_ANIMATION_REGION:-all}" \
  --render "$@"
