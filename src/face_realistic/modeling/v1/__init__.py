"""FRS-v1 face-swap architecture draft.

The package is intentionally isolated from the legacy v0 overfit model.  It
contains a trainable, checkpoint-free implementation skeleton and does not
load InSwapper, CanonSwap, or LivePortrait weights.
"""

from .config import V1ModelConfig
from .conditioning import build_target_conditions
from .geometry import MediaPipeGeometryRasterizer
from .model import FaceRealisticSwapV1

__all__ = [
    "FaceRealisticSwapV1",
    "MediaPipeGeometryRasterizer",
    "V1ModelConfig",
    "build_target_conditions",
]
