# relative_change.py
# Joshua Boquiren (@thejoshbq)
# boquiren@musc.edu
"""
ΔF/F calculation — the most common first preprocessing step in 2-photon imaging.
"""

from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from pynapse.analysis.preprocessing.base import Preprocessor


class DFOverF(Preprocessor):
    def __init__(self, percentile: float = 8.0):
        if not 0 <= percentile <= 100:
            raise ValueError("percentile must be in [0, 100]")
        self.percentile = float(percentile)

    def __call__(self, signals: NDArray[np.floating]) -> NDArray[np.float32]:
        return self.apply(signals)

    def apply(self, signals: NDArray[np.floating]) -> NDArray[np.float32]:
        signals = np.asarray(signals, dtype=np.float64)  # work in double precision internally
        if signals.ndim not in (2, 3):
            raise ValueError(f"Expected 2D or 3D array, got {signals.ndim}D")
        axis = -1
        F0 = np.percentile(signals, self.percentile, axis=axis, keepdims=True)
        F0_safe = np.where(F0 == 0, np.nan, F0)
        df_f = (signals - F0_safe) / F0_safe
        return df_f.astype(np.float32)

