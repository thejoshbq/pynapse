# baseline_subtraction.py
# Joshua Boquiren (@thejoshbq)
# boquiren@musc.edu
"""
Simple baseline subtraction (mean or median) from a time window.
Useful when you want raw fluorescence minus slow drift.
"""

from __future__ import annotations
from typing import Literal
from warnings import warn
import numpy as np
from numpy.typing import NDArray
from brew.core.sample import Sample
from brew.preprocessing.base import Preprocessor


class BaselineSubtraction(Preprocessor):
    def __init__(
        self,
        method: Literal["mean", "median"] = "median",
        window_ms: tuple[float, float] | None = None,
    ):
        super().__init__()
        warn(
            "BaselineSubtraction is a legacy preprocessor and is not used in current "
            "Otis Lab analysis (2023–2025). Prefer DFOverF + ZScore instead.",
            category=FutureWarning,
            stacklevel=2,
        )
        if method not in {"mean", "median"}:
            raise ValueError("method must be 'mean' or 'median'")
        self.method = method
        self.window_ms = None if window_ms is None else (float(window_ms[0]), float(window_ms[1]))

    def apply(self, sample: Sample) -> NDArray[np.float32]:
        signals = sample.get_signals()
        if self.window_ms is None:
            baseline = signals.mean(axis=1, keepdims=True) if self.method == "mean" else np.median(signals, axis=1, keepdims=True)
        else:
            frame_dur = sample.frame_duration * 1000
            start = max(0, int(self.window_ms[0] / frame_dur))
            end = min(signals.shape[1], int(self.window_ms[1] / frame_dur)) if self.window_ms[1] > 0 else signals.shape[1]
            window = signals[:, start:end]
            baseline = window.mean(axis=1, keepdims=True) if self.method == "mean" else np.median(window, axis=1, keepdims=True)
        return (signals - baseline).astype(np.float32)

if __name__ == "__main__":
    test = BaselineSubtraction()
    print(test)

