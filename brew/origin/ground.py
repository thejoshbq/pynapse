# ground.py
# Joshua Boquiren (@thejoshbq)
# boquiren@musc.edu
# Structures .mat files produced from custom MATLAB scripts (Otis Lab, MUSC) and REACHER event logs.

import os
from pathlib import Path
from typing_extensions import deprecated

import scipy.io as sio
import warnings
from tqdm import tqdm
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from typing import Any, Dict, List, Union, Optional

class EventLog:
    def __init__(
        self,
        data: List[Union[str, Path]] | Union[str, Path] | str,
        name: str = "Brew Event Log",
        event_dict: Optional[Dict[int, str]] = None,
    ):
        self._name = name
        self._event_log = None
        self._event_dict = event_dict

        if isinstance(data, str) or isinstance(data, Path):
            if os.path.isfile(data):
                if data.endswith(".mat"):
                    warnings.warn("Use of MATLAB-produced event logs for the 'event_log' parameter is deprecated and will be removed in a future version.", DeprecationWarning)
                    self._event_log = self.__load_mat_file__(data)
                elif data.endswith(".csv"):
                    pass # FIXME: implement CSV reader
        elif isinstance(data, List):
            self._event_log = self.__compile_mat_files__(data)
        self._event_log = self.__create_event_log__()

    @staticmethod
    @deprecated("DEPRECATED: Use of MATLAB-produced event logs for the 'event_log' parameter is deprecated and will be removed in a future version.")
    def __load_mat_file__(path: Union[str, Path]) -> NDArray[Any]:
        """Loads a MATLAB file and returns the event log as a numpy array."""
        if os.path.exists(path) and os.path.isfile(path):
            if path.endswith(".mat"):
                mat_file = sio.loadmat(path)
                event_log = np.squeeze(mat_file['eventlog'])
                event_log = event_log[:, 0:2]
            else:
                raise ValueError("Unsupported file format.")
        else:
            raise ValueError("File does not exist.")
        return event_log

    @deprecated("DEPRECATED: Use only for compatibility with older versions of the library.")
    def __compile_mat_files__(self, paths: List[Union[str, Path]]) -> NDArray[Any]:
        """Concatenates multiple MATLAB files into a single event log."""
        paths = sorted([str(f) for f in paths])
        stack = []
        last_timestamp = 0
        if isinstance(paths, list) and len(paths) > 0:
            for file in paths:
                event_log = self.__load_mat_file__(file)
                event_log[:, 1] = event_log[:, 1] + last_timestamp  # offset timestamps by the last timestamp
                last_timestamp = np.max(event_log[:, 1])
                stack.append(event_log[:, 0:2])
            stack = np.vstack(stack).squeeze() if len(stack) > 0 else np.squeeze(stack)
            if len(stack) > 0:
                stack = np.vstack(stack)
                stack = stack[stack[:, 0] != 0]  # ← FILTER BY EVENT CODE, NOT TIMESTAMP
            else:
                stack = np.empty((0, 2))
        return stack

    def __create_event_log__(self) -> pd.DataFrame:
        """Creates a human-readable table from the event log. The table contains the following columns: code, t1, t2, label."""
        log = np.asarray(self._event_log)
        n_rows, n_cols = log.shape

        data = {
            "code": log[:, 0].astype(int),
            "t1": log[:, 1],  # µs
        }

        if n_cols > 2:
            data["t2"] = log[:, 2]  # µs
        else:
            data["t2"] = np.nan

        df = pd.DataFrame(data, index=range(n_rows))

        if self._event_dict is not None:
            df["label"] = df["code"].map(self._event_dict)
            df = df.dropna(subset=["label"])
        else:
            df["label"] = "unknown"
        return df

    def get_dataframe(self) -> pd.DataFrame:
        """Returns the event log as a pandas DataFrame."""
        return self._event_log

    def count_events(self, target: int | str = None) -> int:
        """Returns the number of events in the event log."""
        if target is None:
            return len(self._event_log)
        else:
            if isinstance(target, str):
                return self._event_log.loc[self._event_log["label"] == target, "code"].count()
            elif isinstance(target, int):
                return self._event_log.loc[self._event_log["code"] == target, "code"].count()

    def __str__(self):
        return f"{self._name} ({self.count_events()} events)"

if __name__ == "__main__":
    t1 = EventLog(
        data=r"../../data/0 EarlyAcq/CTL1/FOV1/HH-CTL1_HER_HI_D1_0_6000_191028-144741_part1.mat")
    print(t1.get_dataframe())

    event_dict = {
        22: "active_lever",
        222: "active_lever_timeout",
        21: "inactive_lever",
        212: "inactive_lever_timeout",
        7: "cue",
        4: "infusion",
    }
    t2 = EventLog(
        data=[r"../../data/0 EarlyAcq/CTL1/FOV1/HH-CTL1_HER_HI_D1_0_6000_191028-144741_part1.mat", r"../../data/0 EarlyAcq/CTL1/FOV1/HH-CTL1_HER_HI_D1_0_6000_191028-163758_part2.mat"], event_dict=event_dict)
    print(t2.get_dataframe().sort_values(by="label"))
    print(t2)