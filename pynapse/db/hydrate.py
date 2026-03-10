# hydrate.py
# DBSample — Sample-compatible object backed by the pynapse database.

import json
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from pynapse.core.mixins import TensorConfigMixin
from . import engine
from . import query as _query


class DBSample(TensorConfigMixin):
    """A lightweight Sample-compatible object that reads from the database.

    Implements the same interface consumed by ``SampleEventTensor`` and the
    rest of the analysis pipeline, but loads data lazily from DuckDB rather
    than from raw files.
    """

    def __init__(
        self,
        fov_id: int,
        conn=None,
        default_event_id: Optional[Union[int, List[int]]] = None,
        default_pre_event: Optional[float] = None,
        default_post_event: Optional[float] = None,
        default_buffer_ms: int = 0,
        default_min_trials: int = 1,
    ):
        self._fov_id = fov_id
        self._conn = conn or engine.get_connection()

        # Load FOV metadata (small, always needed)
        fov = _query.get_fov(fov_id=fov_id, conn=self._conn)
        if fov is None:
            raise KeyError(f"No FOV with id={fov_id}")

        self._name = fov["name"]
        self._fps = float(fov["fps"])
        self._frame_averaging = int(fov["frame_averaging"])
        self._num_neurons = int(fov["num_neurons"])
        self._num_frames = int(fov["num_frames"])
        self._paradigm_id = int(fov["paradigm_id"])

        # Lazy caches
        self._signals_cache = None
        self._dataframe_cache = None
        self._frame_ts_cache = None
        self._event_dict_cache = None

        self._init_tensor_config(
            default_event_id=default_event_id,
            default_pre_event=default_pre_event,
            default_post_event=default_post_event,
            default_buffer_ms=default_buffer_ms,
            default_min_trials=default_min_trials,
        )

    # ------------------------------------------------------------------
    # Properties (match Sample interface exactly)
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return self._name

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def frame_averaging(self) -> int:
        return self._frame_averaging

    @property
    def effective_fps(self) -> float:
        return self._fps / self._frame_averaging

    @property
    def interframe_interval(self) -> float:
        return 1000.0 / self._fps

    @property
    def num_neurons(self) -> int:
        return self._num_neurons

    @property
    def num_frames(self) -> int:
        return self._num_frames

    @property
    def num_events(self) -> int:
        df = self.get_dataframe()
        return len(df)

    # ------------------------------------------------------------------
    # Data access (lazy-loaded with caching)
    # ------------------------------------------------------------------

    def get_signals(self) -> np.ndarray:
        if self._signals_cache is None:
            self._signals_cache = _query.get_traces(self._fov_id, conn=self._conn)
        return self._signals_cache

    def get_dataframe(self) -> pd.DataFrame:
        if self._dataframe_cache is None:
            self._dataframe_cache = _query.get_events(self._fov_id, conn=self._conn)
        return self._dataframe_cache

    def _get_frame_timestamps(self) -> np.ndarray:
        if self._frame_ts_cache is None:
            self._frame_ts_cache = _query.get_frame_timestamps(
                self._fov_id, conn=self._conn
            )
        return self._frame_ts_cache

    def get_event_dict(self) -> Dict[int, str]:
        if self._event_dict_cache is None:
            row = self._conn.execute(
                "SELECT event_dict FROM paradigms WHERE id = ?",
                [self._paradigm_id],
            ).fetchone()
            if row and row[0]:
                self._event_dict_cache = {int(k): v for k, v in json.loads(row[0]).items()}
            else:
                self._event_dict_cache = {}
        return self._event_dict_cache

    def get_num_events(self, event_id: int) -> int:
        df = self.get_dataframe()
        return int((df["code"] == event_id).sum())

    def count_events(self, target=None) -> int:
        df = self.get_dataframe()
        if target is None:
            return len(df)
        if isinstance(target, str):
            return int((df["label"] == target).sum())
        if isinstance(target, int):
            return int((df["code"] == target).sum())
        if isinstance(target, list):
            return int(df["code"].isin(target).sum())
        return 0

    def get_event_timestamps(self, event_id: int) -> np.ndarray:
        df = self.get_dataframe()
        return df[df["code"] == event_id]["t1"].values

    # ------------------------------------------------------------------
    # Tensor creation (TensorConfigMixin contract)
    # ------------------------------------------------------------------

    def create_tensor(
        self,
        event_id: Union[int, List[int]],
        pre_event: float,
        post_event: float,
        buffer_ms: int,
        min_trials: int,
        trace_preprocess: Any,
        window_preprocess: Any,
    ) -> Any:
        from pynapse.analysis.peri_event import SampleEventTensor
        return SampleEventTensor(
            sample=self,
            event_id=event_id,
            pre_event=pre_event,
            post_event=post_event,
            buffer_ms=buffer_ms,
            min_trials=min_trials,
            trace_preprocess=trace_preprocess,
            window_preprocess=window_preprocess,
        )

    def __str__(self):
        return (
            f"DBSample(fov_id={self._fov_id}, name={self._name}, "
            f"neurons={self._num_neurons}, frames={self._num_frames})"
        )
