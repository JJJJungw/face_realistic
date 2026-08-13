#!/usr/bin/env bash
set -euo pipefail

test -f assets/source/swap2.mp4 || {
  echo "assets/source/swap2.mp4가 없습니다." >&2
  exit 1
}
test -f assets/identities/person_01/front_neutral.png || {
  echo "assets/identities/person_01/front_neutral.png가 없습니다." >&2
  exit 1
}
test -f models/inswapper_128.onnx || {
  echo "models/inswapper_128.onnx가 없습니다." >&2
  exit 1
}
test -x .venv/bin/face-swap || {
  echo "가상환경이 없습니다. 먼저 scripts/setup_ec2.sh를 실행하세요." >&2
  exit 1
}

.venv/bin/face-swap \
  --input assets/source/swap2.mp4 \
  --source-identity assets/identities/person_01/front_neutral.png \
  --output outputs/swap_passthrough.mp4 \
  --max-seconds 3 \
  --provider "${FACE_SWAP_PROVIDER:-auto}" \
  --preserve-performance \
  --eye-expansion "${PASSTHROUGH_EYE_EXPANSION:-1.15}" \
  --mouth-expansion "${PASSTHROUGH_MOUTH_EXPANSION:-1.35}" \
  --feather-ratio "${PASSTHROUGH_FEATHER_RATIO:-0.012}"
