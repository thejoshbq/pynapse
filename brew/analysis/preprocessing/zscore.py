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


# class ZScore:
#     def __init__(
#         self,
#         window_ms: Tuple[float, float] = (-3000.0, -500.0),
#         frame_duration_ms: float = 1000.0 / 7.5,  # ~30 Hz default # FIXME: used to be 33.3
#     ) -> None:
#         self.window_ms = (float(window_ms[0]), float(window_ms[1]))
#         self.frame_duration_ms = float(frame_duration_ms)
#
#     def __call__(self, signals: NDArray[np.floating]) -> NDArray[np.float32]:
#         return self.apply(signals)
#
#     def apply(self, signals: NDArray[np.floating]) -> NDArray[np.float32]:
#         signals = np.asarray(signals, dtype=np.float64)
#         if signals.ndim not in (2, 3):
#             raise ValueError(f"Expected 2D or 3D array, got {signals.ndim}D")
#         start = max(0, int(self.window_ms[0] / self.frame_duration_ms))
#         end = int(self.window_ms[1] / self.frame_duration_ms) if self.window_ms[1] > 0 else signals.shape[-1]
#         end = min(end, signals.shape[-1])
#         baseline = signals[..., start:end]
#         mean = baseline.mean(axis=-1, keepdims=True)
#         std = baseline.std(axis=-1, keepdims=True)
#         std = np.where(std == 0, 1.0, std)
#         mean = np.tile(mean, (1, 1, signals.shape[-1]) if signals.ndim == 3 else (1, signals.shape[-1]))
#         std = np.tile(std, (1, 1, signals.shape[-1]) if signals.ndim == 3 else (1, signals.shape[-1]))
#         return ((signals - mean) / std).astype(np.float32)

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

        if self.full_trace:  # Match original: full-trace mean/std
            means = np.nanmean(signals, axis=-1, keepdims=True)
            stds = np.nanstd(signals, axis=-1, keepdims=True)
        else:  # Original windowed
            if signals.ndim == 2:
                start = max(0, int(self.window_ms[0] / self.frame_duration_ms))
                end = min(int(self.window_ms[1] / self.frame_duration_ms), signals.shape[-1])
                baseline = signals[:, start:end]
            else:  # 3D: per-trial windows
                # Assume last dim=time; tile baseline across trials if needed
                start, end = 0, signals.shape[-1]  # Fallback; adjust for event-aligned
                baseline = signals[..., start:end]
            means = np.nanmean(baseline, axis=-1, keepdims=True)
            stds = np.nanstd(baseline, axis=-1, keepdims=True)

        stds = np.where(stds == 0, 1.0, stds)
        # Tile means/stds across time (and trials if 3D)
        if signals.ndim == 2:
            tiled_means = np.tile(means, (1, signals.shape[-1]))
            tiled_stds = np.tile(stds, (1, signals.shape[-1]))
        else:  # 3D trials x neurons x time? Wait, align to (neurons, time, trials) or adjust
            tiled_means = np.tile(means, (1, 1, signals.shape[-1]))
            tiled_stds = np.tile(stds, (1, 1, signals.shape[-1]))
        z = (signals - tiled_means) / tiled_stds
        return z.astype(np.float32)