# gaussian_smoothing.py
# Joshua Boquiren (@thejoshbq)
# boquiren@musc.edu
"""
Temporal Gaussian smoothing — removes high-frequency noise while preserving event timing.
"""

from __future__ import annotations
import numpy as np
from scipy.ndimage import gaussian_filter1d
from brew.core.sample import Sample
from .base import Preprocessor


class GaussianSmoothing(Preprocessor):
    """
    Apply a Gaussian kernel along the time axis.

    Parameters
    ----------
    sigma_frames : float, default 2.0
        Standard deviation of the Gaussian in frames (not ms!).
        1–3 frames is typical for 7–30 Hz imaging after averaging.
    """

    def __init__(self, sigma_frames: float = 2.0):
        if sigma_frames <= 0:
            raise ValueError("sigma_frames must be > 0")
        self.sigma_frames = float(sigma_frames)

    def apply(self, sample: Sample) -> np.ndarray:
        signals = sample.get_signals()
        smoothed = gaussian_filter1d(signals, sigma=self.sigma_frames, axis=1)
        return smoothed.astype(np.float32)