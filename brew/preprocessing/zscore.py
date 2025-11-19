# zscore.py
# Joshua Boquiren (@thejoshbq)
# boquiren@musc.edu
"""
Whole-trace z-scoring using a user-defined baseline window (in ms).
"""

from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from brew.core.sample import Sample
from .base import Preprocessor


class ZScore(Preprocessor):
    def __init__(self, window_ms: tuple[float, float] = (-3000.0, -500.0)):
        self.window_ms = (float(window_ms[0]), float(window_ms[1]))

    def apply(self, sample: Sample) -> NDArray[np.float32]:
        signals = sample.get_signals()
        frame_dur = sample.frame_duration * 1000
        start_frame = int(self.window_ms[0] / frame_dur)
        end_frame = int(self.window_ms[1] / frame_dur)
        start_frame = max(0, start_frame)
        end_frame = min(signals.shape[1], end_frame) if end_frame > 0 else signals.shape[1]
        baseline = signals[:, start_frame:end_frame]
        mean = baseline.mean(axis=1, keepdims=True)
        std = baseline.std(axis=1, keepdims=True)
        std[std == 0] = 1.0
        return ((signals - mean) / std).astype(np.float32)