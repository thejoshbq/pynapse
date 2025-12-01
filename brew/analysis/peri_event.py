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
- EventMatrix: Handles peri-event trace extraction utilities.
"""

import numpy as np
from numpy.typing import NDArray

from brew.core.sample import Sample


class EventMatrix:
    def __init__(
        self,
        sample: Sample,
        event_id: int,
        pre_event: float,
        post_event: float,
        downsample: bool = False,
        min_events: int = 1,
        overlap: int = 1
    ):
        self._sample = sample
        self._signals = sample.get_signals().copy()
        self._event_id = event_id
        self._pre_event = pre_event
        self._post_event = post_event
        self._downsample = downsample
        self._min_events = min_events
        self._overlap = overlap
        self._matrix = self.__extract_event_windows__()

    @staticmethod
    def _sec_to_frames(s: float, fps: float) -> int:
        return int(s * fps)

    def __extract_event_windows__(self) -> NDArray:
        windows = []
        df = self._sample.get_dataframe()
        signals = self._sample.get_signals()
        if self._downsample:
            fps = self._sample.effective_fps
        else:
            fps = self._sample.fps
        event_frame_indices = df["frame_index"][df["code"] == self._event_id]
        if len(event_frame_indices) < self._min_events:
            intended_window_size = self._sec_to_frames(self._pre_event + self._post_event, fps)
            return np.empty((0, intended_window_size))
        pre_frames = self._sec_to_frames(self._pre_event, fps)
        post_frames = self._sec_to_frames(self._post_event, fps)
        window_size = pre_frames + post_frames
        for frame_index in event_frame_indices:
            window = np.array(signals[:, frame_index - pre_frames:frame_index + post_frames])
            if window.shape[1] != window_size:
                continue
            windows.append(window)
        return np.stack(windows)

    def get_event_windows(self) -> NDArray:
        return self._matrix