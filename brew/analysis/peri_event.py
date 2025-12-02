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
from typing import List
from brew.core.sample import Sample
from brew.core.population import Population


class EventMatrix:
    def __init__(
        self,
        data: Sample,
        event_id: int,
        pre_event: float,
        post_event: float,
        buffer_ms: int = 0,
        min_trials: int = 1,
    ):
        self._event_id = event_id
        self._pre_event = pre_event
        self._post_event = post_event
        self._buffer_ms = buffer_ms
        self._min_trials = min_trials
        self._data = data
        self._signals = data.get_signals().copy()
        self._matrix = None

    @staticmethod
    def _sec_to_frames(s: float, fps: float) -> int:
        return int(s * fps)

    def _filter_event_indices(self, indices: NDArray) -> NDArray:
        if self._buffer_ms <= 0:
            return indices
        buffer_frames = self._buffer_ms / self._data.interframe_interval  # Assumes _data set in subclass
        diffs = np.diff(indices)
        valid_mask = np.append(diffs > buffer_frames, True)
        return indices[valid_mask]

    def _extract_from_sample(self, sample: Sample) -> NDArray:
        df = sample.get_dataframe()
        signals = sample.get_signals()  # neurons x frames
        fps = sample.effective_fps
        event_indices = df["frame_index"][df["code"] == self._event_id].values
        if len(event_indices) < self._min_trials:
            raise ValueError(f"Insufficient events ({len(event_indices)} < {self._min_trials})")
        event_indices = self._filter_event_indices(event_indices)
        windows = []
        pre = self._sec_to_frames(self._pre_event, fps)
        post = self._sec_to_frames(self._post_event, fps)
        for idx in event_indices:
            start, end = idx - pre, idx + post
            if 0 <= start and end <= signals.shape[1]:
                win = signals[:, start:end]
                windows.append(win)
        if len(windows) < self._min_trials:
            raise ValueError(f"Only {len(windows)} valid windows.")
        return np.stack(windows)

class SampleEventTensor(EventMatrix):
    def __init__(
            self,
            sample: Sample,
            event_id: int,
            pre_event: float,
            post_event: float,
            buffer_ms: int = 0,
            min_trials: int = 1,
    ):
        self._data = sample
        super().__init__(event_id, pre_event, post_event, buffer_ms, min_trials)

    def _compute_windows(self) -> None:
        self._windows = self._extract_from_sample(self._data)

    def get_event_windows(self) -> NDArray:
        if self._windows is None:
            self._compute_windows()
        return self._windows

class PopulationEventTensor(EventMatrix): # FIXME: incomplete
    def __init__(
            self,
            population: Population,
            event_id: int,
            pre_event: float,
            post_event: float,
            buffer_ms: int = 0,
            min_trials: int = 1,
    ):
        self._data = population
        super().__init__(event_id, pre_event, post_event, buffer_ms, min_trials)

    def _compute_windows(self) -> None:
        self._windows = [self._extract_from_sample(s) for s in self._data.get_samples()]

    def get_event_windows(self) -> List[NDArray]:
        if self._windows is None:
            self._compute_windows()
        return self._windows