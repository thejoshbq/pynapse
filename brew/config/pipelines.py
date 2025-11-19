# pipelines.py
# Joshua Boquiren (@thejoshbq)
# boquiren@musc.edu
"""
Laboratory-standard preprocessing pipelines.

This is the **single source of truth** for all analysis pipelines used in figures,
papers, and the local database. Import these instead of constructing pipelines manually.
"""

from __future__ import annotations
from brew.preprocessing import (
    DFOverF,
    ZScore,
    GaussianSmoothing,
    BaselineSubtraction,
    ProcessingPipeline,
)

OTIS_PIPE = ProcessingPipeline(
    steps=[
        DFOverF(percentile=8),
        GaussianSmoothing(sigma_frames=20),
        ZScore(window_ms=(-3000.0, -500.0)),
    ]
)

LEGACY_PIPE = ProcessingPipeline(
    steps=[
        BaselineSubtraction(method="median", window_ms=None),
        DFOverF(percentile=8),
        ZScore(window_ms=(-3000.0, -500.0)),
    ]
)