# peri_event.py
# Joshua Boquiren (@thejoshbq)
# boquiren@musc.edu

"""
Peri-event trace extraction — clean, composable, database-future-proof.

Contains:
  - PeriEventTraces                single Sample/FOV
  - PopulationPeriEventTraces   multiple Samples (Population or list)
"""

import numpy as np
from numpy.typing import NDArray
from typing import Literal, Sequence, overload

from brew.core.sample import Sample
from brew.core.population import Population


class PETH: # Peri-Event Trace Histogram
    def __init__(
        self,
        sample: Sample,
        event_dict: dict,
        event_names: Sequence[str],
        event_bins: Sequence[int],
        event_hist_type: Literal["count", "density"] = "count",
        event_hist_norm: Literal["none", "percent", "density"] = "none",
        event_hist_log: bool = False,
    ):
        self._sample = sample
        self._event_dict = event_dict
        self._event_names = event_names
        self._event_bins = event_bins
        self._event_hist_type = event_hist_type
        self._event_hist_norm = event_hist_norm
        self._event_hist_log = event_hist_log
