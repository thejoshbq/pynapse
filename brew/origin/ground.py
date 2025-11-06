# ground.py

import os
from pathlib import Path
import scipy.io as sio
import warnings
from tqdm import tqdm
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from typing import Any, Dict, List, Union

class EventLog:
    def __init__(
        self,
        name: str = "Event Log",
        event_log: List[Union[str, Path]] = None | Dict[str, Any] | Union[str, Path],
        event_dict: Dict[str, Any] = None,
    ):
        self._name = name
        self._event_log = None
        self._event_dict = event_dict
        self._table = None

        if isinstance(event_log, str):
            if os.path.isfile(event_log):
                if event_log.endswith(".mat"):
                    warnings.warn("Use of MATLAB-produced event logs for the 'event_log' parameter is deprecated and will be removed in a future version.", DeprecationWarning)
                    self._event_log = self.__load_mat_file__(event_log)
                elif event_log.endswith(".csv"):
                    pass # FIXME: implement CSV reader
                else:
                    raise ValueError("Unsupported file format.")
        elif isinstance(event_log, List):
            self._event_log = self.__concat_mat_files__(event_log)

        self._table = self.__create_table__()


    # ====================
    # Private Methods
    # ====================
    @staticmethod
    def __load_mat_file__(path: Union[str, Path]) -> NDArray[Any]:
        """
        Deprecated: use only for compatibility with older versions of the library.
        Loads a MATLAB file and returns the event log as a numpy array.
        """
        mat_file = sio.loadmat(path)
        event_log = np.squeeze(mat_file['eventlog'])
        event_log = event_log[:, 0:2]

        return event_log

    def __concat_mat_files__(self, paths: List[Union[str, Path]]) -> NDArray[Any]:
        """
        Deprecated: use only for compatibility with older versions of the library.
        Concatenates multiple MATLAB files into a single event log.
        """
        paths = sorted([str(f) for f in paths])
        stack = []
        last_timestamp = 0
        if isinstance(paths, list) and len(paths) > 0:
            for _, file in enumerate(tqdm(paths, desc=f"Compiling MATLAB files, n={len(paths)}", total=len(paths))):
                try:
                    event_log = self.__load_mat_file__(file)
                    event_log[:, 1] = event_log[:, 1] + last_timestamp  # offset timestamps by the last timestamp
                    last_timestamp = np.max(event_log[:, 1])
                    stack.append(event_log)
                except Exception as e:
                    print(f"Error loading {file}: {e}")
            stack = np.vstack(stack).squeeze() if len(stack) > 0 else np.squeeze(stack)
            stack = stack[stack[:, 1] != 0]  # only return valid data

        return stack

    def __create_table__(self) -> pd.DataFrame:
        """
        Creates a human-readable table from the event log. The table contains the following columns: code, t1, t2, label.
        """
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

    # ====================
    # Public Methods
    # ====================
    def table(self) -> pd.DataFrame:
        return self._table

    def event_count(self, target: int = None | str) -> int:
        if target is None:
            return len(self._table)
        else:
            if isinstance(target, str):
                return self._table.loc[self._table["label"] == target, "code"].count()
            elif isinstance(target, int):
                return self._table.loc[self._table["code"] == target, "code"].count()

if __name__ == "__main__":
    # Testing EventLog class creation from filepath
    t1 = EventLog(event_log=r"C:\Users\boqui\OneDrive\Desktop\Projects\brew\data\0 EarlyAcq\CTL1\FOV1\HH-CTL1_HER_HI_D1_0_6000_191028-144741_part1.mat")
    print(t1.table())

    # Testing EventLog class creation from a list of filepaths
    t2 = EventLog(event_log=[r"C:\Users\boqui\OneDrive\Desktop\Projects\brew\data\0 EarlyAcq\CTL1\FOV1\HH-CTL1_HER_HI_D1_0_6000_191028-144741_part1.mat", r"C:\Users\boqui\OneDrive\Desktop\Projects\sink2p\data\0 EarlyAcq\CTL1\FOV1\HH-CTL1_HER_HI_D1_0_6000_191028-163758_part2.mat"])
    print(t2.table())

    # Testing EventLog class creation with a dictionary of event codes and labels
    event_dict = {
        22: "active_lever",
        222: "active_lever_timeout",
        21: "inactive_lever",
        212: "inactive_lever_timeout",
        7: "cue",
        4: "infusion",
    }
    t3 = EventLog(event_log=[r"C:\Users\boqui\OneDrive\Desktop\Projects\brew\data\0 EarlyAcq\CTL1\FOV1\HH-CTL1_HER_HI_D1_0_6000_191028-144741_part1.mat", r"C:\Users\boqui\OneDrive\Desktop\Projects\sink2p\data\0 EarlyAcq\CTL1\FOV1\HH-CTL1_HER_HI_D1_0_6000_191028-163758_part2.mat"], event_dict=event_dict)
    print(t3.table().sort_values(by="label"))

    print(f"Number of events: {t3.event_count()}, Number of active_press events: {t3.event_count(22)}, Number of active_press_timeout events: {t3.event_count("active_lever_timeout")}")
