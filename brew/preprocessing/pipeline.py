# pipeline.py
# Joshua Boquiren (@thejoshbq)
# boquiren@musc.edu
"""
Composable preprocessing pipeline — the single source of truth for every figure.
"""

from __future__ import annotations
from typing import Sequence
import numpy as np
from numpy.typing import NDArray
from brew.core.sample import Sample
from .base import Preprocessor


class ProcessingPipeline:
    def __init__(self, steps: Sequence[Preprocessor]):
        self.steps = list(steps)

    def apply(self, sample: Sample) -> NDArray[np.float32]:
        signals = sample.get_signals().copy()
        for step in self.steps:
            signals = step.apply(sample)
        return signals

    def __repr__(self) -> str:
        return f"<ProcessingPipeline>"