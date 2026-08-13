#!/usr/bin/env bash
set -euo pipefail

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$PWD/.local/bin:$PATH"
  export PATH="$HOME/.local/bin:$PATH"
fi

need_ffmpeg=0
need_gles=0
command -v ffmpeg >/dev/null 2>&1 || need_ffmpeg=1
ldconfig -p 2>/dev/null | grep -q 'libGLESv2\.so\.2' || need_gles=1

if (( need_ffmpeg || need_gles )); then
  if command -v apt-get >/dev/null 2>&1; then
    packages=(libgl1 libegl1 libgles2 libglib2.0-0)
    (( need_ffmpeg )) && packages+=(ffmpeg)
    sudo apt-get update
    sudo apt-get install -y "${packages[@]}"
  elif command -v dnf >/dev/null 2>&1; then
    packages=(mesa-libGL mesa-libEGL mesa-libGLES glib2)
    (( need_ffmpeg )) && packages+=(ffmpeg-free)
    sudo dnf install -y "${packages[@]}"
  else
    echo "지원되지 않는 패키지 관리자입니다. FFmpeg와 EGL/GLES 런타임을 직접 설치하세요." >&2
    exit 1
  fi
fi

face_swap_device="${FACE_SWAP_DEVICE:-cpu}"
if [[ "$face_swap_device" == "gpu" ]]; then
  uv sync --extra dev --extra swap --extra swap-gpu
elif [[ "$face_swap_device" == "cpu" ]]; then
  uv sync --extra dev --extra swap --extra swap-cpu
else
  echo "FACE_SWAP_DEVICE는 cpu 또는 gpu여야 합니다." >&2
  exit 1
fi

echo
echo "$face_swap_device 환경 준비 완료"
echo "그다음 models/inswapper_128.onnx 존재 여부를 확인하세요."
