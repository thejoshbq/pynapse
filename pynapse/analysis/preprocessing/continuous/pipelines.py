# continuous/pipelines.py
# Joshua Boquiren (@thejoshbq)
# boquiren@musc.edu
"""
Laboratory-standard preprocessing pipelines.

This is the **single source of truth** for all analysis pipelines used in figures,
papers, and the local database. Import these instead of constructing pipelines manually.
"""


from __future__ import annotations
from pynapse.analysis.preprocessing.continuous import (
    LegacyNormalize,
)
