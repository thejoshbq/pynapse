# gaussian_smoothing.py
# Joshua Boquiren (@thejoshbq)
# boquiren@musc.edu
"""
Temporal Gaussian smoothing — removes high-frequency noise while preserving event timing.
"""

from __future__ import annotations
import numpy as np
from scipy.ndimage import gaussian_filter1d
from numpy.typing import NDArray


class GaussianSmoothing:
    def __init__(self, sigma_frames: float = 2.0) -> None:
        if sigma_frames <= 0:
            raise ValueError("sigma_frames must be > 0")
        self.sigma_frames = float(sigma_frames)

    def __call__(self, signals: NDArray[np.floating]) -> NDArray[np.float32]:
        return self.apply(signals)

    def apply(self, signals: NDArray[np.floating]) -> NDArray[np.float32]:
        signals = np.asarray(signals)
        if signals.ndim not in (2, 3):
            raise ValueError(f"Expected 2D or 3D array, got {signals.ndim}D")

        smoothed = gaussian_filter1d(signals, sigma=self.sigma_frames, axis=-1)
        return smoothed.astype(np.float32)