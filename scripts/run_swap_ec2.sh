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
  echo "models/inswapper_128.onnx가 없습니다. 정식으로 확보한 모델을 업로드하세요." >&2
  exit 1
}

uv run face-swap \
  --input assets/source/swap2.mp4 \
  --source-identity assets/identities/person_01/front_neutral.png \
  --output outputs/swap_baseline.mp4 \
  --max-seconds 3 \
  --provider auto
