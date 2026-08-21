#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv가 없습니다. 먼저 bash scripts/setup_ec2.sh를 실행하세요." >&2
  exit 1
fi

export UV_PROJECT_ENVIRONMENT="${PROJECT_ROOT}/.venv-train"
uv sync --extra dev --extra train

"${PROJECT_ROOT}/.venv-train/bin/python" - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA PyTorch가 준비되지 않았습니다.")
print("GPU:", torch.cuda.get_device_name(0))
PY
