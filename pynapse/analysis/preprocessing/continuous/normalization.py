# normalization.py
# Joshua Boquiren (@thejoshbq)
# boquiren@musc.edu
"""
Legacy normalization that matches the original Jupyter notebook logic.
This normalizes by mean and then z-scores the FULL TRACE.
"""

from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from pynapse.analysis.preprocessing.base import Preprocessor


class LegacyNormalize(Preprocessor):
    def __init__(self, z_score: bool = True):
        self.z_score = z_score
    
    def __call__(self, signals: NDArray[np.floating]) -> NDArray[np.float32]:
        return self.apply(signals)
    
    def apply(self, signals: NDArray[np.floating]) -> NDArray[np.float32]:
        signals = np.asarray(signals, dtype=np.float64)
        
        if signals.ndim != 2:
            raise ValueError(f"LegacyNormalize expects 2D array (neurons x frames), got {signals.ndim}D")
        
        # Step 1: Normalize by mean
        means = np.nanmean(signals, axis=1, keepdims=True)
        means_safe = np.where(means == 0, np.nan, means)
        signals_norm = signals / means_safe
        
        # Step 2: Z-score full trace (if enabled)
        if self.z_score:
            for neuron in range(signals_norm.shape[0]):
                mean = np.nanmean(signals_norm[neuron])
                std = np.nanstd(signals_norm[neuron])
                if std > 0:
                    signals_norm[neuron] = (signals_norm[neuron] - mean) / std
        return signals_norm.astype(np.float32)
