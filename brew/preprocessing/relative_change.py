# relative_change.py
# Joshua Boquiren (@thejoshbq)
# boquiren@musc.edu
"""
ΔF/F calculation — the most common first preprocessing step in 2-photon imaging.
"""

from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from brew.core.sample import Sample
from .base import Preprocessor


class DFOverF(Preprocessor):
    def __init__(self, percentile: float = 8.0):
        if not 0 <= percentile <= 100:
            raise ValueError("percentile must be in [0, 100]")
        self.percentile = float(percentile)

    def apply(self, sample: Sample) -> NDArray[np.float32]:
        signals = sample.get_signals()
        F0 = np.percentile(signals, self.percentile, axis=1, keepdims=True)
        F0[F0 == 0] = np.nan
        return ((signals - F0) / F0).astype(np.float32)

