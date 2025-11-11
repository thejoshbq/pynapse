# bean.py
# Joshua Boquiren (@thejoshbq)
# boquiren@musc.edu
# Encapsulates an aligned sample of event logs and neural fluorescence signals.

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
from ground import EventLog
from roast import SignalRecording

class Sample:
    def __init__(
        self,
        event_data: List[Union[str, Path]] | Union[str, Path] | str | EventLog,
        signal_data: List[Union[str, Path]] | Union[str, Path] | str | SignalRecording,
        name: str = "Brew Sample",
        event_dict: Optional[Dict[int, str]] = None,
        fps: float = 30.0,
        frame_averaging: int = 1,
        frame_correction: bool = False,
        correction_file: Union[str, Path] = None,
        start_time: float = 0.0,
    ):
        self._name = name
        self._fps = fps
        self._frame_averaging = frame_averaging
        self._effective_fps = fps / frame_averaging
        self._frame_duration = 1.0 / self._effective_fps
        self._interframe_interval = 1000.0 / self._fps
        self._start_time = start_time
        self._frame_correction = frame_correction
        self._correction_file = correction_file

        if isinstance(event_data, EventLog):
            self._event_log = event_data
        else:
            self._event_log = EventLog(data=event_data, name=f"{name} Events", event_dict=event_dict)

        if isinstance(signal_data, SignalRecording):
            self._signals = signal_data
        else:
            self._signals = SignalRecording(data=signal_data, name=f"{name} Signals")

        if self._signals.num_frames == 0 or self._event_log.count_events() == 0:
            raise ValueError("Event log or signal recording is empty.")

    @property
    def num_neurons(self) -> int:
        """Returns the number of neurons in the signal recording."""
        return self._signals.num_neurons

    @property
    def num_frames(self) -> int:
        """Returns the number of timepoints in the signal recording."""
        return self._signals.num_frames

    def count_events(self, target: int | str = None) -> int:
        """Returns the number of events in the event log."""
        return self._event_log.count_events(target)

    def __load_mat_file__(self, path: Union[str, Path]):
        return self._event_log.__load_mat_file__(path)

    def __fill_assumed_frames__(self):
        assumed_data = self.__load_mat_file__(self._correction_file)
        max_timestamp = np.max(assumed_data[:, 1])
        length = len(assumed_data)
        temp_stack = np.vstack((assumed_data, assumed_data, assumed_data))
        temp_stack[length:, 1] += max_timestamp
        temp_stack[2 * length:, 1] += max_timestamp
        assumed_data = temp_stack

        frame_timestamps = assumed_data[assumed_data[:, 0] == 9, 1]
        dropped_timestamps = []
        diff_timestamps = np.diff(frame_timestamps)
        missed_timestamps_indices = np.where(diff_timestamps > 1.5 * self._interframe_interval)[0]
        for i in missed_timestamps_indices:
            num_frames_dropped = int(np.round((frame_timestamps[i + 1] - frame_timestamps[i]) / (self._interframe_interval - 1)))
            temp = [frame_timestamps[i] + a * self._interframe_interval for a in range(1, num_frames_dropped + 1)]
            dropped_timestamps.extend(temp)
        frame_timestamps = np.sort(np.concatenate((frame_timestamps, dropped_timestamps)))
        return frame_timestamps

    def __fill_missed_frames__(self, frames):
        first_frame = np.array(frames[0])
        last_frame = np.array([int(np.max(frames) + (500 * self._interframe_interval))])
        temp_frame_indices = np.concatenate((first_frame, frames, last_frame))
        missed_frame_indices = []
        for i in range(len(temp_frame_indices) - 1):
            num_missed = int(np.round((temp_frame_indices[i + 1] - temp_frame_indices[i]) / self._interframe_interval) - 1)
            if num_missed > 0:
                for missed_frame in range(num_missed):
                    missed = temp_frame_indices[i] + int(self._interframe_interval * (missed_frame - 1))
                    missed_frame_indices.append(missed)
        missed_frames_filled = np.sort(np.concatenate((temp_frame_indices, missed_frame_indices)))
        return missed_frames_filled

    def __downsample_frames__ (self):
        raw_log = self._event_log.get_data()
        raw_frame_timestamps = raw_log[raw_log[:, 0] == 9, 1]
        if self._frame_correction:
            corrected_frame_timestamps = self.__fill_assumed_frames__()
        else:
            corrected_frame_timestamps = self.__fill_missed_frames__(raw_frame_timestamps)
        downsampled_frame_timestamps = corrected_frame_timestamps[::self._frame_averaging]
        return downsampled_frame_timestamps

    def __get_event_frame_indices__(self):
        event_ts = self._event_log.get_data()[:, 1]
        frame_ts = self.__downsample_frames__()

        indices = np.zeros(len(event_ts), dtype=np.uint64)
        for i, ev in enumerate(event_ts):
            if np.isnan(ev):
                continue
            cand = np.nonzero(frame_ts <= ev)[0]
            indices[i] = cand[-1] if cand.size else 0
        return indices

    def get_dataframe(self) -> pd.DataFrame:
        """Returns a pandas DataFrame containing the aligned event log and signal recording."""
        df = self._event_log.get_dataframe().copy()
        df["frame_index"] = self.__get_event_frame_indices__()
        return df

if __name__ == "__main__":
    event_dict = {
        22: "active_lever",
        222: "active_lever_timeout",
        21: "inactive_lever",
        212: "inactive_lever_timeout",
        7: "cue",
        4: "infusion",
    }
    sample = Sample(
        event_data=[r"../../data/0 EarlyAcq/CTL1/FOV1/HH-CTL1_HER_HI_D1_0_6000_191028-144741_part1.mat", r"../../data/0 EarlyAcq/CTL1/FOV1/HH-CTL1_HER_HI_D1_0_6000_191028-163758_part2.mat"],
        signal_data=[r"../../data/0 EarlyAcq/CTL1/FOV1/T2_HH-CTL1_HER_HI_D1_behavior-001_extractedsignals_raw_part1.npy", r"../../data/0 EarlyAcq/CTL1/FOV1/T2_HH-CTL1_HER_HI_D1_behavior-000_extractedsignals_raw_part2.npy"],
        event_dict=event_dict,
        fps=30,
        frame_averaging=4,
        frame_correction=True,
        correction_file=r"../../data/empty.mat"
    )
    df = sample.get_dataframe()
    print(df[df["label"] == "active_lever"])