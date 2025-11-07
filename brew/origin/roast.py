# roast.py
# Joshua Boquiren (@thejoshbq)
# boquiren@musc.edu
# Structures 2-photon imaging recordings of neural fluorescence.

import os
from pathlib import Path
import warnings
warnings.filterwarnings('always', category=UserWarning)
warnings.filterwarnings('always', category=DeprecationWarning)
from tqdm import tqdm
import numpy as np
from numpy.typing import NDArray
from typing import Any, List, Union

class SignalRecording:
    def __init__(
        self,
        data: List[Union[str, Path]] | Union[str, Path] | str,
        name: str = "Brew Signal Recording",
    ):
        self._name = name
        self._signals = None

        if isinstance(data, list):
            self._signals = self.__compile_npy_files__(data)
        elif isinstance(data, str) or isinstance(data, Path):
            self._signals = self.__load_npy_file__(data)

    @property
    def num_neurons(self) -> int:
        """Returns the number of neurons in the signal recording."""
        return self._signals.shape[0]

    @property
    def num_timepoints(self) -> int:
        """Returns the number of timepoints in the signal recording."""
        return self._signals.shape[1]

    @staticmethod
    def __load_npy_file__(path: Union[str, Path]) -> NDArray[Any]:
        """Loads a numpy file and returns the signals as a numpy array."""
        if os.path.exists(path) and os.path.isfile(path):
            if path.endswith(".npy"):
                with warnings.catch_warnings(record=True) as captured_warnings:
                    npy_file = np.load(path).squeeze()
                    for warning in captured_warnings:
                        if "Python 2" in str(warning.message): # reformat old files
                            np.save(str(path), npy_file)
            else:
                raise ValueError("Unsupported file format.")
        else:
            raise ValueError("File does not exist.")
        return npy_file

    def __compile_npy_files__(self, paths: List[Union[str, Path]]) -> NDArray[Any]:
        """Concatenates multiple numpy files into a single signal recording."""
        stack = []
        if isinstance(paths, list) and len(paths) > 0:
            for file in paths:
                stack.append(self.__load_npy_file__(file))
        return np.hstack(stack).squeeze() if len(stack) > 0 else np.array(stack)

    def get_array(self) -> NDArray[Any]:
        """Returns the signals as a numpy array."""
        return self._signals

    def __str__(self):
        return f"{self._name}"

if __name__ == "__main__":
    t1 = SignalRecording(name="Test Signal Recording 1",
        data=r"/home/thejoshbq/Desktop/Projects/brew/data/0 EarlyAcq/CTL1/FOV1/T2_HH-CTL1_HER_HI_D1_behavior-001_extractedsignals_raw_part1.npy")
    print(t1)

    t2 = SignalRecording(name="Test Signal Recording 2", data=[r"/home/thejoshbq/Desktop/Projects/brew/data/0 EarlyAcq/CTL1/FOV1/T2_HH-CTL1_HER_HI_D1_behavior-001_extractedsignals_raw_part1.npy", r"/home/thejoshbq/Desktop/Projects/brew/data/0 EarlyAcq/CTL1/FOV1/T2_HH-CTL1_HER_HI_D1_behavior-000_extractedsignals_raw_part2.npy"])
    print(t2)
    print(t2.get_array())