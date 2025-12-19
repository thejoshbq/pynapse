# zscore.py
# Joshua Boquiren (@thejoshbq)
# boquiren@musc.edu
"""
Whole-trace z-scoring using a user-defined baseline window (in ms).
"""

from __future__ import annotations
from typing import Tuple
import numpy as np
from numpy.typing import NDArray


class ZScore:
    def __init__(
            self,
            window_ms: Tuple[float, float] = (-3000.0, -500.0),
            frame_duration_ms: float = 133.333,  # 1000/7.5 averaged
            full_trace: bool = False,  # NEW: Match original global Z
    ) -> None:
        self.window_ms = (float(window_ms[0]), float(window_ms[1]))
        self.frame_duration_ms = float(frame_duration_ms)
        self.full_trace = bool(full_trace)

    def __call__(self, signals: NDArray[np.floating]) -> NDArray[np.float32]:
        return self.apply(signals)

    def apply(self, signals: NDArray[np.floating]) -> NDArray[np.float32]:
        signals = np.asarray(signals, dtype=np.float64)
        if signals.ndim not in (2, 3):
            raise ValueError(f"Expected 2D or 3D array, got {signals.ndim}D")
        if self.full_trace:  # full-trace mean/std
            means = np.nanmean(signals, axis=-1, keepdims=True)
            stds = np.nanstd(signals, axis=-1, keepdims=True)
        else:
            if signals.ndim == 2:
                start = max(0, int(self.window_ms[0] / self.frame_duration_ms))
                end = min(int(self.window_ms[1] / self.frame_duration_ms), signals.shape[-1])
                baseline = signals[:, start:end]
            else:
                start, end = 0, signals.shape[-1]
                baseline = signals[..., start:end]
            means = np.nanmean(baseline, axis=-1, keepdims=True)
            stds = np.nanstd(baseline, axis=-1, keepdims=True)
        stds = np.where(stds == 0, 1.0, stds)
        if signals.ndim == 2:
            tiled_means = np.tile(means, (1, signals.shape[-1]))
            tiled_stds = np.tile(stds, (1, signals.shape[-1]))
        else:
            tiled_means = np.tile(means, (1, 1, signals.shape[-1]))
            tiled_stds = np.tile(stds, (1, 1, signals.shape[-1]))
        z = (signals - tiled_means) / tiled_stds
        return z.astype(np.float32)