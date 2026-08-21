"""Small state container for model-free mask stabilization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class TemporalMaskState:
    smoothing: float = 0.65
    _previous: np.ndarray | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.smoothing < 1.0:
            raise ValueError("smoothing must be in [0, 1)")

    def update(self, current: np.ndarray, *, reset: bool = False) -> np.ndarray:
        current = np.asarray(current, dtype=np.float32)
        if reset or self._previous is None or self._previous.shape != current.shape:
            output = current.copy()
        else:
            output = self.smoothing * self._previous + (1.0 - self.smoothing) * current
        self._previous = output.copy()
        return output

    def clear(self) -> None:
        self._previous = None
