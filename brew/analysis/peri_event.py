# peri_event.py
# Joshua Boquiren (@thejoshbq)
# boquiren@musc.edu
"""
A module to extract peri-event signal windows from sample data.

This module provides functionality for handling peri-event traces based
on a given list of samples. The key features include calculating windows
of signal data centered around specified events and converting time in
seconds to frame indices based on the sampling frequency.

Classes:
- PeriEventTraces: Handles peri-event trace extraction utilities.
"""

import numpy as np
from numpy.typing import NDArray
from brew.core.sample import Sample
from typing import List


class PeriEventTraces:
    def __init__(
        self,
        samples: Sample | List[Sample],
    ):
        if isinstance(samples, Sample):
            samples = [samples]
        self._samples = samples

    @staticmethod
    def _sec_to_frames(s: int, fps: int) -> int:
        return int(s * fps)

    def get_event_windows(self, event_id: int, pre_event: int, post_event: int) -> NDArray[np.float32]:
        windows = []
        for sample in self._samples:
            df = sample.get_dataframe()
            signals = sample.get_signals()
            fps = sample.fps
            event_frame_indices = df["frame_index"][df["code"] == event_id]
            pre_frames = self._sec_to_frames(pre_event, fps)
            post_frames = self._sec_to_frames(post_event, fps)
            window_size = pre_frames + post_frames
            for frame_index in event_frame_indices:
                window = np.array(signals[:, frame_index - pre_frames:frame_index + post_frames])
                if window.shape[1] != window_size:
                    continue
                windows.append(window)
        return windows
