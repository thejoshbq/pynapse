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
from typing import List, Optional
from pynapse.core import Sample, Population
from pynapse.analysis.preprocessing import Preprocessor, Pipeline


class EventTensor:
    def __init__(
        self,
        data: Sample | Population,
        event_id: int | List[int],
        pre_event: float,
        post_event: float,
        buffer_ms: int = 0,
        min_trials: int = 1,
        trace_preprocess: Optional[Preprocessor] | Pipeline = None,
        window_preprocess: Optional[Preprocessor] | Pipeline = None,
    ):
        self._data = data
        self._event_id = event_id if isinstance(event_id, list) else [event_id]
        self._pre_event = pre_event
        self._post_event = post_event
        self._event_frame = self._sec_to_frames(self._pre_event, self._data.effective_fps)
        self._buffer_ms = buffer_ms
        self._min_trials = min_trials
        self._trace_preprocess = trace_preprocess
        self._window_preprocess = window_preprocess
        self._tensor = None

    @staticmethod
    def _sec_to_frames(s: float, fps: float) -> int:
        return int(s * fps)

    def _filter_event_indices(self, indices: NDArray) -> NDArray:
        if self._buffer_ms <= 0:
            return indices
        buffer_frames = self._buffer_ms / self._data.interframe_interval
        diffs = np.diff(indices)
        valid_mask = np.append(diffs > buffer_frames, True)
        return indices[valid_mask]

class SampleEventTensor(EventTensor):
    def __init__(
            self,
            sample: Sample,
            event_id: int | List[int],
            pre_event: float,
            post_event: float,
            buffer_ms: int = 0,
            min_trials: int = 1,
            trace_preprocess: Optional[Preprocessor] | Pipeline = None,
            window_preprocess: Optional[Preprocessor] | Pipeline = None,
    ):
        self._data = sample
        self._trace_preprocess = trace_preprocess
        self._window_preprocess = window_preprocess
        super().__init__(
            sample,
            event_id,
            pre_event,
            post_event,
            buffer_ms,
            min_trials,
            trace_preprocess,
            window_preprocess
        )

    def _extract_event_windows(self) -> NDArray[np.floating]:
        df = self._data.get_dataframe()
        signals = self._data.get_signals()
        if self._trace_preprocess is not None:
            signals = self._trace_preprocess.apply(signals)
        fps_eff = self._data.effective_fps
        event_ts_ms = []
        for eid in self._event_id:
            event_ts_ms.extend(df[df["code"] == eid]["t1"].values)
        event_ts_ms = np.sort(np.array(event_ts_ms, dtype=np.float64))
        if self._buffer_ms > 0:
            diffs_ms = np.diff(event_ts_ms)
            close_later_idx = np.where(diffs_ms < self._buffer_ms)[0] + 1
            event_ts_ms = np.delete(event_ts_ms, close_later_idx)
        if len(event_ts_ms) < self._min_trials:
            return np.empty((0, self._data.num_neurons, 0))
        frame_ts = self._data._get_frame_timestamps() # FIXME: access private method
        indices = np.searchsorted(frame_ts, event_ts_ms, side="right") - 1
        valid = frame_ts[indices] <= event_ts_ms
        indices[~valid] = 0
        pre_frames = self._sec_to_frames(self._pre_event, fps_eff)
        post_frames = self._sec_to_frames(self._post_event, fps_eff)
        window_size = pre_frames + post_frames
        windows = []
        for idx in indices:
            if idx == 0:
                continue
            start = idx - pre_frames
            end = idx + post_frames
            if start < 0 or end > signals.shape[1]:
                continue
            win = signals[:, start:end]
            windows.append(win)
        if not windows:
            return np.empty((0, self._data.num_neurons, window_size))
        stacked = np.stack(windows, axis=0)

        if self._window_preprocess is not None:
            stacked = self._window_preprocess.apply(stacked)

        return stacked

    def get_event_windows(self) -> NDArray:
        if self._tensor is None:
            self._tensor = self._extract_event_windows()
        return self._tensor

class PopulationEventTensor:
    def __init__(
            self,
            population: Population,
            event_id: int | List[int],
            pre_event: float,
            post_event: float,
            buffer_ms: int = 0,
            min_trials: int = 1,
            trace_preprocess: Optional[Preprocessor] | Pipeline = None,
            window_preprocess: Optional[Preprocessor] | Pipeline = None,
    ):
        self._data = population
        self._event_id = event_id if isinstance(event_id, list) else [event_id]
        self._pre_event = pre_event
        self._post_event = post_event
        self._buffer_ms = buffer_ms
        self._min_trials = min_trials
        self._trace_preprocess = trace_preprocess
        self._window_preprocess = window_preprocess
        first_sample = population.get_samples()[0]
        self._event_frame = int(pre_event * first_sample.effective_fps)
        self._tensor = self._extract_event_windows()

    def _extract_event_windows(self) -> List[NDArray]:
        event_windows = []
        for sample in self._data.get_samples():
            total_events = sum(sample.get_num_events(eid) for eid in self._event_id)
            if total_events < self._min_trials:
                continue
            sample_tensor = SampleEventTensor(
                sample,
                self._event_id,
                self._pre_event,
                self._post_event,
                self._buffer_ms,
                self._min_trials,
                self._trace_preprocess,
                self._window_preprocess,
            )
            event_windows.append(sample_tensor.get_event_windows())
        return event_windows

    @property
    def name(self):
        return self._data.name
    
    @property
    def num_neurons(self) -> int:
        return self._data.num_neurons

    @property
    def num_samples(self) -> int:
        return self._data.num_samples

    def get_event_windows(self) -> List[NDArray]:
        if self._tensor is None:
            self._tensor = self._extract_event_windows()
        return self._tensor

    def get_grand_average(self) -> NDArray:
        trial_averages = []
        for tensor in self._tensor:
            if tensor.size > 0:
                trial_averages.append(np.nanmean(tensor, axis=0))
        if not trial_averages:
            return np.array([])
        trial_average = np.vstack(trial_averages)
        return np.nanmean(trial_average, axis=0)

    def get_trial_average(
        self, epoch: str = "full_trace", method: str = "unsorted"
    ) -> NDArray:
        trial_averages = []
        for tensor in self._tensor:
            if tensor.size > 0:
                trial_averages.append(np.nanmean(tensor, axis=0))
        if not trial_averages:
            return np.array([])
        trial_average = np.vstack(trial_averages)
        if epoch == "post_event":
            sort_slice = slice(self._event_frame, None)
        elif epoch == "pre_event":
            sort_slice = slice(None, self._event_frame)
        elif epoch == "full_trace":
            sort_slice = slice(None, None)
        else:
            raise ValueError(f"Invalid epoch: {epoch}")
        if method == "excitatory":
            sort_idx = np.argsort(np.argmax(trial_average[:, sort_slice], axis=1))
        elif method == "inhibitory":
            sort_idx = np.argsort(np.argmin(trial_average[:, sort_slice], axis=1))
        elif method == "unsorted":
            sort_idx = np.arange(len(trial_average))
        else:
            raise ValueError(f"Invalid sort method: {method}")
        return trial_average[sort_idx]