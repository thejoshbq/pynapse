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
from brew.core.sample import Sample


class PeriEventTraces:
    def __init__(
        self,
        sample: Sample,
    ):
        self._sample = sample

    def _sec_to_frames(self, s: int) -> int:
        return int(s * self._sample.fps)

    def get_event_windows(self, event_id: int, pre_event: int, post_event: int) -> NDArray[np.float32]:
        df = self._sample.get_dataframe()
        signals = self._sample.get_signals()
        event_frame_indices = df["frame_index"][df["code"] == event_id]
        pre_frames = self._sec_to_frames(pre_event)
        post_frames = self._sec_to_frames(post_event)
        windows = []
        for frame_index in event_frame_indices:
            window = signals[:, frame_index - pre_frames:frame_index + post_frames + 1]
            windows.append(window)
        windows = np.array(windows)
        return windows
