# baseline_subtraction.py
# Joshua Boquiren (@thejoshbq)
# boquiren@musc.edu
"""
Simple baseline subtraction (mean or median) from a time window.
Legacy? Yes. Still useful for quick checks and teaching? Absolutely.
"""

from __future__ import annotations
from typing import Literal, Tuple, Optional
from warnings import warn
import numpy as np
from numpy.typing import NDArray
from pynapse.analysis.preprocessing.base import Preprocessor


class BaselineSubtraction(Preprocessor):
    def __init__(
        self,
        method: Literal["mean", "median"] = "mean",
        window_ms: Optional[Tuple[float, float]] = None,
        frame_duration_ms: Optional[float] = None,
    ) -> None:
        warn(
            "BaselineSubtraction is legacy. Prefer DFOverF + ZScore for publication figures!",
            FutureWarning,
            stacklevel=2,
        )
        if method not in {"mean", "median"}:
            raise ValueError("method must be 'mean' or 'median'")
        self.method = method
        self.window_ms = None if window_ms is None else (float(window_ms[0]), float(window_ms[1]))
        self.frame_duration_ms = frame_duration_ms

    def __call__(self, signals: NDArray[np.floating]) -> NDArray[np.float32]:
        return self.apply(signals)

    def apply(self, signals: NDArray[np.floating]) -> NDArray[np.float32]:
        signals = np.asarray(signals, dtype=np.float64)
        if signals.ndim not in (2, 3):
            raise ValueError(f"Expected 2D or 3D array, got {signals.ndim}D")

        if self.window_ms is None:
            axis = -1
            if self.method == "mean":
                baseline = signals.mean(axis=axis, keepdims=True)
            else:
                baseline = np.median(signals, axis=axis, keepdims=True)
        else:
            if self.frame_duration_ms is None:
                raise ValueError("frame_duration_ms required when using window_ms")
            start = max(0, int(self.window_ms[0] / self.frame_duration_ms))
            end = int(self.window_ms[1] / self.frame_duration_ms) if self.window_ms[1] > 0 else signals.shape[-1]
            end = min(end, signals.shape[-1])
            window = signals[..., start:end]
            if self.method == "mean":
                baseline = window.mean(axis=-1, keepdims=True)
            else:
                baseline = np.median(window, axis=-1, keepdims=True)
            baseline = np.tile(baseline, (1, 1, signals.shape[-1]) if signals.ndim == 3 else (1, signals.shape[-1]))
        return (signals - baseline).astype(np.float32)