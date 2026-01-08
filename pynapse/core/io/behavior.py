# behavior.py
# Joshua Boquiren (@thejoshbq)
# boquiren@musc.edu
"""
Module to handle event logs derived from MATLAB files or CSVs. Provides functionalities
to parse, process, and export the event log into human-readable formats using pandas.

The EventLog class supports loading individual or multiple event log files, creating a
human-readable event log in the form of a pandas DataFrame, and mapping event codes to
labels using an optional event dictionary.

Classes:
    - EventLog: Represents an event log and provides various utilities to process and analyze it.
"""

import os
from pathlib import Path
from typing_extensions import deprecated
import scipy.io as sio
import warnings
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from typing import Any, Dict, List, Union, Optional


class EventLog:
    def __init__(
        self,
        source: List[Union[str, Path]] | Union[str, Path] | str,
        name: str | None = None,
        event_dict: Optional[Dict[int, str]] = None,
    ):
        self._name = name or self.__class__.__name__
        self._source = source
        self._event_log = None
        self._event_dict = event_dict
        if isinstance(source, str) or isinstance(source, Path):
            if os.path.isfile(source):
                if source.endswith(".mat"):
                    warnings.warn("Use of MATLAB-produced event logs for the 'event_log' parameter is deprecated and will be removed in a future version.", DeprecationWarning)
                    self._raw_data = self._load_mat_file(source)
                elif source.endswith(".csv"):
                    pass # FIXME: CSV reader to be implemented in future version
                else:
                    raise ValueError("Unsupported file format.")
            else:
                raise ValueError("File does not exist.")
        elif isinstance(source, List):
            self._raw_data = self._compile_mat_files(source)
        self._event_log = self._create_event_log()

    @property
    def name(self) -> str:
        return self._name

    @property
    def source(self) -> Union[str, Path] | List[Union[str, Path]] | str:
        return self._source

    @property
    def num_events(self) -> int:
        return len(self._event_log)

    @staticmethod
    @deprecated("DEPRECATED: Use of MATLAB-produced event logs for the 'event_log' parameter is deprecated and will be removed in a future version.")
    def _load_mat_file(path: Union[str, Path]) -> NDArray[Any]:
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
    def _compile_mat_files(self, paths: List[Union[str, Path]]) -> NDArray[Any]:
        paths = [str(f) for f in paths]
        stack = []
        last_timestamp = 0
        if isinstance(paths, list) and len(paths) > 0:
            for file in sorted(paths):
                event_log = self._load_mat_file(file)
                event_log[:, 1] = event_log[:, 1] + last_timestamp  # offset timestamps by the last timestamp
                last_timestamp = np.max(event_log[:, 1])
                stack.append(event_log[:, 0:2])
            stack = np.vstack(stack).squeeze() if len(stack) > 0 else np.squeeze(stack)
            if len(stack) > 0:
                stack = np.vstack(stack)
                stack = stack[stack[:, 0] != 0]
            else:
                stack = np.empty((0, 2))
        return stack

    def _create_event_log(self) -> pd.DataFrame:
        log = np.asarray(self._raw_data)
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
        return self._event_log

    def get_code_dict(self) -> Dict[int, str]:
        return self._event_dict

    def get_raw_data(self) -> NDArray[Any]:
        return self._raw_data

    def count_events(self, target: int | List[int] | str = None) -> int:
        if target is None:
            return len(self._event_log)
        else:
            if isinstance(target, str):
                return self._event_log.loc[self._event_log["label"] == target, "code"].count()
            elif isinstance(target, int):
                return self._event_log.loc[self._event_log["code"] == target, "code"].count()
            elif isinstance(target, list):
                return self._event_log.loc[self._event_log["code"].isin(target), "code"].count()

    def __str__(self):
        name = f"Name: {self.name}"
        if isinstance(self._source, str):
            source = f"Source: {self._source.split('/')[-1]}"
        else:
            source = "Source:"
            for s in self._source:
                source += f"\n - {s.split('/')[-1]}"
        n_neurons = f"Number of Events: {self.count_events()}"
        return f"{name}\n{source}\n{n_neurons}"

if __name__ == "__main__":
    t1 = EventLog(
        source=r"../../data/0 EarlyAcq/CTL1/FOV1/HH-CTL1_HER_HI_D1_0_6000_191028-144741_part1.mat")
    print(t1)

    event_dict = {
        22: "active_lever",
        222: "active_lever_timeout",
        21: "inactive_lever",
        212: "inactive_lever_timeout",
        7: "cue",
        4: "infusion",
    }
    t2 = EventLog(
        source=[r"../../data/0 EarlyAcq/CTL1/FOV1/HH-CTL1_HER_HI_D1_0_6000_191028-144741_part1.mat", r"../../data/0 EarlyAcq/CTL1/FOV1/HH-CTL1_HER_HI_D1_0_6000_191028-163758_part2.mat"],
        event_dict=event_dict
    )
    print(t2.get_dataframe().sort_values(by="label"))