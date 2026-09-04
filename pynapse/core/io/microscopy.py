# microscopy.py
# Joshua Boquiren (@thejoshbq)
# boquiren@musc.edu
"""
Provides the SignalRecording class for handling neural fluorescence recordings.

This module is focused on structuring `.npy` 2-photon imaging recording files
of neural fluorescence, as well as roigbiv's `.h5` trace exports (written via
`pandas.HDFStore`). The primary class, SignalRecording, enables loading,
processing, and analyzing these recordings.

Classes:
    - SignalRecording: Handles neural signal recordings, including loading
      and concatenating `.npy` files, or reading a roigbiv `.h5` trace export.
"""

import os
from pathlib import Path
import warnings
warnings.filterwarnings('always', category=UserWarning)
warnings.filterwarnings('always', category=DeprecationWarning)
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from typing import Any, List, Optional, Union

# Trace kinds a roigbiv `.h5` export may store. There is no default — callers
# must say which one they want (see SignalRecording.__init__).
H5_KINDS = ("f", "dff", "raw", "neuropil")


class SignalRecording:
    def __init__(
        self,
        source: List[Union[str, Path]] | Union[str, Path] | str,
        name: str | None = None,
        kind: Optional[str] = None,
    ):
        self._name = name or self.__class__.__name__
        self._source = source
        self._kind = kind
        self._fs: Optional[float] = None
        self._neuron_ids: Optional[List[str]] = None
        self._meta: Optional[pd.DataFrame] = None

        if isinstance(source, list):
            self._signals = self._compile_files(source, kind)
        elif isinstance(source, str) or isinstance(source, Path):
            self._signals = self._load_file(source, kind)

    @property
    def name(self) -> str:
        return self._name

    @property
    def source(self) -> Union[str, Path] | List[Union[str, Path]] | str:
        return self._source

    @property
    def kind(self) -> Optional[str]:
        """Trace kind selected for a `.h5` source (``None`` for `.npy`)."""
        return self._kind

    @property
    def fs(self) -> Optional[float]:
        """Authoritative sampling rate from a `.h5` source's `/meta` (``None`` for `.npy`)."""
        return self._fs

    @property
    def neuron_ids(self) -> Optional[List[str]]:
        """Neuron identifier columns from a `.h5` source (``None`` for `.npy`)."""
        return self._neuron_ids

    @property
    def meta(self) -> Optional[pd.DataFrame]:
        """Raw `/meta` DataFrame from a `.h5` source (``None`` for `.npy`)."""
        return self._meta

    @property
    def num_neurons(self) -> int:
        return self._signals.shape[0]

    @property
    def num_frames(self) -> int:
        return self._signals.shape[1]

    @staticmethod
    def _is_h5_path(path: Union[str, Path]) -> bool:
        return str(path).lower().endswith((".h5", ".hdf5"))

    def _load_file(self, path: Union[str, Path], kind: Optional[str]) -> NDArray[Any]:
        if self._is_h5_path(path):
            return self._load_h5_file(path, kind)
        return self._load_npy_file(path)

    @staticmethod
    def _load_npy_file(path: Union[str, Path]) -> NDArray[Any]:
        if os.path.exists(path) and os.path.isfile(path):
            if str(path).endswith(".npy"):
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

    def _load_h5_file(self, path: Union[str, Path], kind: Optional[str]) -> NDArray[np.float32]:
        """Load one trace kind from a roigbiv `.h5` export.

        roigbiv writes traces via ``pandas.HDFStore`` as DataFrames shaped
        ``(n_frames, n_rois)`` (index ``time_s``, columns = neuron ids) under
        keys ``/f``, ``/dff``, ``/raw``, ``/neuropil``, plus a `/meta` table
        with an authoritative ``fs``. This transposes to pynapse's
        ``(neurons, frames)`` convention.
        """
        if not os.path.exists(path) or not os.path.isfile(path):
            raise ValueError("File does not exist.")

        if kind is None:
            raise ValueError(
                "kind is required for .h5 sources (no default trace kind). "
                f"Choose one of {H5_KINDS}."
            )
        if kind not in H5_KINDS:
            raise ValueError(f"Unknown kind '{kind}'. Expected one of {H5_KINDS}.")

        key = f"/{kind}"
        with pd.HDFStore(str(path), mode="r") as store:
            available = [k for k in store.keys() if k != "/meta"]
            if key not in store:
                raise ValueError(
                    f"Kind '{kind}' not found in {path}. Available kinds: "
                    f"{[k.lstrip('/') for k in available]}"
                )
            df = store[key]
            meta = store["/meta"] if "/meta" in store else None

        self._neuron_ids = list(df.columns)
        self._meta = meta
        if meta is not None and "fs" in meta.columns and len(meta) > 0:
            self._fs = float(meta["fs"].iloc[0])

        return df.to_numpy(dtype=np.float32).T

    def _compile_files(self, paths: List[Union[str, Path]], kind: Optional[str]) -> NDArray[Any]:
        if not isinstance(paths, list) or len(paths) == 0:
            return np.array([])

        is_h5 = [self._is_h5_path(p) for p in paths]
        if any(is_h5) and not all(is_h5):
            raise ValueError("Cannot mix .npy and .h5 files in a single SignalRecording.")

        if all(is_h5):
            return self._compile_h5_files(paths, kind)
        return self._compile_npy_files(paths)

    def _compile_npy_files(self, paths: List[Union[str, Path]]) -> NDArray[Any]:
        stack = []
        for file in sorted(paths):
            stack.append(self._load_npy_file(file))
        return np.hstack(stack).squeeze() if len(stack) > 0 else np.array(stack)

    def _compile_h5_files(self, paths: List[Union[str, Path]], kind: Optional[str]) -> NDArray[np.float32]:
        stack = []
        fs_values = []
        neuron_ids_per_part = []
        for file in sorted(paths):
            part = self._load_h5_file(file, kind)
            stack.append(part)
            fs_values.append(self._fs)
            neuron_ids_per_part.append(self._neuron_ids)

        if len(set(fs_values)) > 1:
            raise ValueError(f"All .h5 parts must share the same fs. Got: {fs_values}")
        if any(ids != neuron_ids_per_part[0] for ids in neuron_ids_per_part):
            raise ValueError("All .h5 parts must share the same neuron columns.")

        return np.hstack(stack)

    def get_signals(self) -> NDArray[Any]:
        return self._signals

    def __str__(self):
        name = f"Name: {self.name}"
        if isinstance(self._source, str):
            source = f"Source: {self._source.split('/')[-1]}"
        else:
            source = "Source:"
            for s in self._source:
                source += f"\n - {s.split('/')[-1]}"
        n_neurons = f"Number of Neurons: {self.num_neurons}"
        n_frames = f"Number of Frames: {self.num_frames}"
        return f"{name}\n{source}\n{n_neurons}\n{n_frames}"

if __name__ == "__main__":
    t1 = SignalRecording(name="Test Signal Recording 1",
                         source=r"./../../data/0 EarlyAcq/CTL1/FOV1/T2_HH-CTL1_HER_HI_D1_behavior-001_extractedsignals_raw_part1.npy")
    print(t1)

    t2 = SignalRecording(name="Test Signal Recording 2", source=[r"./../../data/0 EarlyAcq/CTL1/FOV1/T2_HH-CTL1_HER_HI_D1_behavior-001_extractedsignals_raw_part1.npy", r"./../../data/0 EarlyAcq/CTL1/FOV1/T2_HH-CTL1_HER_HI_D1_behavior-000_extractedsignals_raw_part2.npy"])
    print(t2)
    print(t2.get_signals())
