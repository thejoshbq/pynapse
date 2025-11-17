# bean.py
# Joshua Boquiren (@thejoshbq)
# boquiren@musc.edu
# Encapsulates an aligned sample of event logs and neural fluorescence signals.

import numpy as np
import pandas as pd
import scipy.io as sio
from typing import Dict, List, Optional, Union
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
        correction_file: Union[str, Path, None] = None,
        start_time: float = 0.0,
    ):
        self._name = name
        self._fps = fps
        self._frame_averaging = frame_averaging
        self._start_time = start_time
        self._frame_correction = frame_correction
        self._correction_file = correction_file

        if isinstance(event_data, EventLog):
            self._event_log = event_data
        else:
            self._event_log = EventLog(source=event_data, name=f"{name} Events", event_dict=event_dict)

        if isinstance(signal_data, SignalRecording):
            self._signals = signal_data
        else:
            self._signals = SignalRecording(source=signal_data, name=f"{name} Signals")

        if self._signals.num_frames == 0 or self._event_log.count_events() == 0:
            raise ValueError("Event log or signal recording is empty.")

        self._averaged_frame_ts: Optional[np.ndarray] = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def num_neurons(self) -> int:
        return self._signals.num_neurons

    @property
    def num_frames(self) -> int:
        return self._signals.num_frames

    @property
    def num_events(self) -> int:
        return self._event_log.num_events

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def frame_averaging(self) -> int:
        return self._frame_averaging

    @property
    def effective_fps(self) -> float:
        return self.fps / self.frame_averaging

    @property
    def frame_duration(self) -> float:
        return 1.0 / self.effective_fps

    @property
    def interframe_interval(self) -> float:
        return 1000.0 / self.fps

    @property
    def start_time(self) -> float:
        return self._start_time

    def count_events(self, target: int | str = None) -> int:
        return self._event_log.count_events(target)

    def _handle_missed_frames(self, frame_ts_raw: np.ndarray) -> np.ndarray:
        """Insert missing frames when we have real frame triggers (code 9)."""
        frame_ts = frame_ts_raw.copy()
        if frame_ts.size == 0:
            n = self.num_frames * self._frame_averaging
            return np.arange(n) * self.interframe_interval

        first = np.array([0.0])
        last = np.array([np.max(frame_ts) + 500 * self.interframe_interval])
        temp = np.concatenate((first, frame_ts, last))

        inserted = []
        for i in range(len(temp) - 1):
            gap = temp[i + 1] - temp[i]
            n_miss = int(round(gap / self.interframe_interval) - 1)
            if n_miss > 0:
                for j in range(1, n_miss + 1):
                    inserted.append(temp[i] + j * self.interframe_interval)

        if inserted:
            frame_ts = np.sort(np.concatenate((temp, np.array(inserted))))
        else:
            frame_ts = temp
        return frame_ts

    def _handle_assumed_frames(self, mat_file: Optional[Union[str, Path]]) -> np.ndarray:
        """For unreliable animals (e.g. CTL1) – use empty.mat template or fall back to regular grid."""
        if mat_file is None:
            n = self.num_frames * self._frame_averaging
            return np.arange(n) * self.interframe_interval

        try:
            data = sio.loadmat(str(mat_file))
            log = np.squeeze(data["eventlog"])
            if log.ndim == 1:
                log = log[:, np.newaxis]

            max_t = np.max(log[:, 1])
            n_rows = log.shape[0]
            tri = np.vstack((log, log, log))
            tri[n_rows:, 1] += max_t
            tri[2 * n_rows:, 1] += 2 * max_t

            frame_ts = tri[tri[:, 0] == 9, 1]

            diffs = np.diff(frame_ts)
            drop_idx = np.where(diffs > 1.5 * self.interframe_interval)[0]
            inserted = []
            for idx in drop_idx:
                gap = frame_ts[idx + 1] - frame_ts[idx]
                n_drop = int(round(gap / self.interframe_interval) - 1)
                if n_drop > 0:
                    for a in range(1, n_drop + 1):
                        inserted.append(frame_ts[idx] + a * self.interframe_interval)

            if inserted:
                frame_ts = np.sort(np.concatenate((frame_ts, np.array(inserted))))
        except Exception as e:
            print(f"Correction file failed ({e}); falling back to regular grid")
            n = self.num_frames * self._frame_averaging
            frame_ts = np.arange(n) * self.interframe_interval

        return frame_ts

    def _get_frame_timestamps(self) -> np.ndarray:
        """Return timestamps (ms) for every *averaged* signal frame."""
        if self._averaged_frame_ts is not None:
            return self._averaged_frame_ts

        raw = self._event_log.get_raw_data()
        frame_ts_raw = raw[raw[:, 0] == 9, 1]

        if self._frame_correction:
            full_ts = self._handle_assumed_frames(self._correction_file)
        else:
            full_ts = self._handle_missed_frames(frame_ts_raw)

        averaged_ts = full_ts[::self._frame_averaging]

        if self.num_frames != len(averaged_ts):
            print(f"Warning: signal frames ({self.num_frames}) ≠ timestamp frames ({len(averaged_ts)}). "
                  "Trimmed signals in this case.")

        self._averaged_frame_ts = averaged_ts
        return averaged_ts

    def get_dataframe(self) -> pd.DataFrame:
        df = self._event_log.get_dataframe().copy()

        frame_ts = self._get_frame_timestamps()
        event_ts = df["t1"].values

        indices = np.searchsorted(frame_ts, event_ts, side="right") - 1
        indices = np.clip(indices, 0, len(frame_ts) - 1)

        df["frame_index"] = indices.astype(np.int64)
        return df

    def __str__(self):
        name = f"Name: {self.name}"
        if isinstance(self._event_log.source, str):
            event_source = f"Event log source: {self._event_log.source.split('/')[-1]}"
        else:
            event_source = f"Event log source:"
            for s in self._event_log.source:
                event_source += f"\n - {s.split('/')[-1]}"
        if isinstance(self._signals.source, str):
            signal_source = f"Signal recording source: {self._signals.source.split('/')[-1]}"
        else:
            signal_source = "Signal recording source:"
            for s in self._signals.source:
                signal_source += f"\n - {s.split('/')[-1]}"
        fps = f"FPS: {self.fps}"
        averaging = f"Frame averaging: {self.frame_averaging}"
        n_neurons = f"Number of Neurons: {self.num_neurons}"
        n_frames = f"Number of Frames: {self.num_frames}"
        n_events = f"Number of Events: {self.num_events}"
        return f"{name}\n{event_source}\n{signal_source}\n{fps}\n{averaging}\n{n_neurons}\n{n_frames}\n{n_events}"

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
        event_data=[r"../../data/0 EarlyAcq/CTL1/FOV1/HH-CTL1_HER_HI_D1_0_6000_191028-144741_part1.mat",
                    r"../../data/0 EarlyAcq/CTL1/FOV1/HH-CTL1_HER_HI_D1_0_6000_191028-163758_part2.mat"],
        signal_data=[r"../../data/0 EarlyAcq/CTL1/FOV1/T2_HH-CTL1_HER_HI_D1_behavior-001_extractedsignals_raw_part1.npy",
                     r"../../data/0 EarlyAcq/CTL1/FOV1/T2_HH-CTL1_HER_HI_D1_behavior-000_extractedsignals_raw_part2.npy"],
        event_dict=event_dict,
        fps=30,
        frame_averaging=4,
        frame_correction=True,
        correction_file=r"../../data/empty.mat"
    )
    df = sample.get_dataframe()
    print(df[df["label"] == "active_lever"])
    print(sample)