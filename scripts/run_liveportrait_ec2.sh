#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

python3 -m face_realistic.performance.liveportrait \
  --project-root "${PROJECT_ROOT}" \
  --liveportrait-dir "${LIVEPORTRAIT_DIR:-${PROJECT_ROOT}/third_party/LivePortrait}" \
  --source-image "${LIVEPORTRAIT_SOURCE:-${PROJECT_ROOT}/assets/identities/person_01/front_neutral.png}" \
  --input-video "${LIVEPORTRAIT_DRIVING:-${PROJECT_ROOT}/assets/source/swap2.mp4}" \
  --output "${LIVEPORTRAIT_OUTPUT:-${PROJECT_ROOT}/outputs/liveportrait_exp_baseline.mp4}" \
  --report "${LIVEPORTRAIT_REPORT:-${PROJECT_ROOT}/outputs/liveportrait_exp_baseline.json}" \
  --max-seconds "${LIVEPORTRAIT_MAX_SECONDS:-3}" \
  --device-id "${LIVEPORTRAIT_DEVICE_ID:-0}" \
  --driving-multiplier "${LIVEPORTRAIT_DRIVING_MULTIPLIER:-1.0}" \
  --animation-region "${LIVEPORTRAIT_ANIMATION_REGION:-exp}"
