#!/usr/bin/env bash
# passthrough 부위별 ablation. 스왑은 한 번만 돌고 설정별 합성·측정만 반복한다.
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
test -f models/face_detection_yunet_2023mar.onnx || {
  echo "models/face_detection_yunet_2023mar.onnx가 없습니다." >&2
  exit 1
}
test -f models/face_recognition_sface_2021dec.onnx || {
  echo "models/face_recognition_sface_2021dec.onnx가 없습니다." >&2
  exit 1
}
test -x .venv/bin/swap-ablation || {
  echo "swap-ablation 진입점이 없습니다. 'uv sync --extra dev --extra swap --extra swap-gpu'를 다시 실행하세요." >&2
  exit 1
}

.venv/bin/swap-ablation \
  --input "${ABLATION_INPUT:-assets/source/swap2.mp4}" \
  --identity "${ABLATION_IDENTITY:-assets/identities/person_01/front_neutral.png}" \
  --cache-dir "${ABLATION_CACHE:-outputs/ablation_cache}" \
  --max-seconds "${ABLATION_SECONDS:-3}" \
  --provider "${FACE_SWAP_PROVIDER:-auto}" \
  --feather-ratio "${PASSTHROUGH_FEATHER_RATIO:-0.012}" \
  "${@}"
