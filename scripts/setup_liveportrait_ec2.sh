#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LIVEPORTRAIT_DIR="${LIVEPORTRAIT_DIR:-${PROJECT_ROOT}/third_party/LivePortrait}"
LIVEPORTRAIT_REPO="${LIVEPORTRAIT_REPO:-https://github.com/KlingAIResearch/LivePortrait.git}"
LIVEPORTRAIT_REF="${LIVEPORTRAIT_REF:-main}"

for command in git uv ffmpeg; do
  if ! command -v "${command}" >/dev/null 2>&1; then
    echo "필수 명령을 찾을 수 없습니다: ${command}" >&2
    exit 1
  fi
done

mkdir -p "$(dirname "${LIVEPORTRAIT_DIR}")"
if [[ ! -d "${LIVEPORTRAIT_DIR}/.git" ]]; then
  git clone "${LIVEPORTRAIT_REPO}" "${LIVEPORTRAIT_DIR}"
fi

git -C "${LIVEPORTRAIT_DIR}" fetch --depth 1 origin "${LIVEPORTRAIT_REF}"
git -C "${LIVEPORTRAIT_DIR}" checkout --detach FETCH_HEAD

if [[ ! -x "${LIVEPORTRAIT_DIR}/.venv/bin/python" ]]; then
  uv venv --python 3.10 "${LIVEPORTRAIT_DIR}/.venv"
fi
LIVEPORTRAIT_PYTHON="${LIVEPORTRAIT_DIR}/.venv/bin/python"

# Official LivePortrait Linux baseline versions, isolated from face_realistic/.venv.
uv pip install --python "${LIVEPORTRAIT_PYTHON}" \
  torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 \
  --index-url https://download.pytorch.org/whl/cu121
uv pip install --python "${LIVEPORTRAIT_PYTHON}" \
  -r "${LIVEPORTRAIT_DIR}/requirements.txt" \
  "huggingface_hub>=0.23,<1"

"${LIVEPORTRAIT_DIR}/.venv/bin/hf" download KlingTeam/LivePortrait \
  --local-dir "${LIVEPORTRAIT_DIR}/pretrained_weights" \
  --exclude "*.git*" \
  --exclude "README.md" \
  --exclude "docs"

"${LIVEPORTRAIT_PYTHON}" -c \
  'import torch; assert torch.cuda.is_available(), "CUDA is not available"; print("LivePortrait CUDA:", torch.cuda.get_device_name(0))'

echo "LivePortrait commit: $(git -C "${LIVEPORTRAIT_DIR}" rev-parse HEAD)"
echo "전용 환경 준비 완료: ${LIVEPORTRAIT_DIR}/.venv"
