# pipeline.py
# Joshua Boquiren (@thejoshbq)
# boquiren@musc.edu
"""
Composable preprocessing pipeline — the single source of truth for every figure.
"""

from __future__ import annotations
from typing import Sequence, Callable
import numpy as np
from numpy.typing import NDArray


Processor = Callable[[NDArray[np.floating]], NDArray[np.float32]]


class Pipeline:
    def __init__(self, steps: Sequence[Processor]) -> None:
        self.steps = list(steps)

    def __call__(self, signals: NDArray[np.floating]) -> NDArray[np.float32]:
        return self.apply(signals)

    def apply(self, signals: NDArray[np.floating]) -> NDArray[np.float32]:
        data = np.asarray(signals, dtype=np.float64)
        for step in self.steps:
            data = step(data)
        return data.astype(np.float32)