# base.py
# Joshua Boquiren (@thejoshbq)
# boquiren@musc.edu
"""
Defines the Preprocessor abstract base class for processing data samples.

This module provides a base class that must be subclassed to create specific
preprocessors. Preprocessors are designed to modify and transform data samples,
enforcing a consistent interface with the apply method that must be implemented
by subclasses.

Classes:
    Preprocessor: An abstract base class requiring the implementation of the
    apply method to process samples.
"""

from abc import ABC, abstractmethod
from brew.core.sample import Sample
import numpy as np
from numpy.typing import NDArray


class Preprocessor(ABC):
    @abstractmethod
    def apply(self, sample: Sample) -> NDArray[np.float32]:
        pass

    def __repr__(self) -> str:
        params = ", ".join(f"{k}={v!r}" for k, v in self.__dict__.items() if not k.startswith("_"))
        return f"{self.__class__.__name__}({params})"