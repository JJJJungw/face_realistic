#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

ORIGINAL="${MASTER_DRIVING:-${PROJECT_ROOT}/assets/source/swap2.mp4}"
REENACTED="${MASTER_REENACTED:-${PROJECT_ROOT}/outputs/master_reenacted_raw.mp4}"

LIVEPORTRAIT_SOURCE="${MASTER_IMAGE:-${PROJECT_ROOT}/assets/identities/person_01/front_neutral.png}" \
LIVEPORTRAIT_DRIVING="${ORIGINAL}" \
LIVEPORTRAIT_OUTPUT="${REENACTED}" \
LIVEPORTRAIT_REPORT="${PROJECT_ROOT}/outputs/master_reenacted_raw.json" \
LIVEPORTRAIT_MAX_SECONDS="${MASTER_MAX_SECONDS:-3}" \
LIVEPORTRAIT_ANIMATION_REGION=all \
bash "${PROJECT_ROOT}/scripts/run_liveportrait_ec2.sh"

"${PROJECT_ROOT}/.venv/bin/python" -m face_realistic.performance.reenactment_compositor \
  --original "${ORIGINAL}" \
  --reenacted "${REENACTED}" \
  --output "${MASTER_OUTPUT:-${PROJECT_ROOT}/outputs/master_reenactment_swap.mp4}" \
  --report "${MASTER_REPORT:-${PROJECT_ROOT}/outputs/master_reenactment_swap.json}" \
  --max-seconds "${MASTER_MAX_SECONDS:-3}" \
  --mask-contract "${MASTER_MASK_CONTRACT:-0.88}" \
  --feather-pixels "${MASTER_FEATHER_PIXELS:-9}" \
  --transform-alpha "${MASTER_TRANSFORM_ALPHA:-0.35}"

