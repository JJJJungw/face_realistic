"""Original-preserving face compositing primitives."""

from .masks import build_visible_alpha
from .pasteback import CompositorConfig, compose_aligned_crop, pasteback_to_frame

__all__ = [
    "CompositorConfig",
    "build_visible_alpha",
    "compose_aligned_crop",
    "pasteback_to_frame",
]
