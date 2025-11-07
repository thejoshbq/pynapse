# sample.py
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
        start_time: float = 0.0,
    ):
        self._name = name
        self._effective_fps = fps / frame_averaging
        self._frame_duration = 1.0 / self._effective_fps
        self._start_time = start_time

        if isinstance(event_data, EventLog):
            self._event_log = event_data
        else:
            self._event_log = EventLog(data=event_data, name=f"{name} Events", event_dict=event_dict)

        if isinstance(signal_data, SignalRecording):
            self._signals = signal_data
        else:
            self._signals = SignalRecording(data=signal_data, name=f"{name} Signals")

        if self._signals.num_timepoints == 0 or self._event_log.count_events() == 0:
            raise ValueError("Event log or signal recording is empty.")

        self._aligned_dataframe = self.__align_data__()

    @property
    def num_neurons(self) -> int:
        """Returns the number of neurons in the signal recording."""
        return self._signals.num_neurons

    @property
    def num_timepoints(self) -> int:
        """Returns the number of timepoints in the signal recording."""
        return self._signals.num_timepoints

    def count_events(self, target: int | str = None) -> int:
        """Returns the number of events in the event log."""
        return self._event_log.count_events(target)

    def __align_data__(self) -> pd.DataFrame:
        """Aligns the event log and signal recording to a common timebase."""
        df = self._event_log.get_dataframe().copy()
        event_sec = df["t1"].astype(float) / 1_000_000
        event_sec = event_sec - self._start_time
        frame_idx = np.floor(event_sec * self._effective_fps).astype(int)
        max_frame = self.num_timepoints - 1
        frame_idx = np.clip(frame_idx, 0, max_frame)
        df["frame_index"] = frame_idx
        return df

    def get_dataframe(self) -> pd.DataFrame:
        """Returns a pandas DataFrame containing the aligned event log and signal recording."""
        return self._aligned_dataframe

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
        event_data=[r"/home/thejoshbq/Desktop/Projects/brew/data/0 EarlyAcq/CTL1/FOV1/HH-CTL1_HER_HI_D1_0_6000_191028-144741_part1.mat", r"/home/thejoshbq/Desktop/Projects/brew/data/0 EarlyAcq/CTL1/FOV1/HH-CTL1_HER_HI_D1_0_6000_191028-163758_part2.mat"],
        signal_data=[r"/home/thejoshbq/Desktop/Projects/brew/data/0 EarlyAcq/CTL1/FOV1/T2_HH-CTL1_HER_HI_D1_behavior-001_extractedsignals_raw_part1.npy", r"/home/thejoshbq/Desktop/Projects/brew/data/0 EarlyAcq/CTL1/FOV1/T2_HH-CTL1_HER_HI_D1_behavior-000_extractedsignals_raw_part2.npy"],
        event_dict=event_dict,
        fps=30,
        frame_averaging=4
    )
    print(sample.get_dataframe())