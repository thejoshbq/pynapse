# test_db_integration.py
# Integration test: ingest real REACHER data, load as DBSample,
# and compare SampleEventTensor output against the original Sample.

import pytest
import numpy as np
import pandas as pd
import duckdb
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants (same dataset as test_reacher_real_data.py)
# ---------------------------------------------------------------------------
DATA_DIR = Path("/home/thejoshbq/Otis-Lab/Analysis/REACHER-Trial/data/FOV")
XLSX_PATH = DATA_DIR / "D1CONFON5F_REACHERTEST_1HR.xlsx"
SIGNAL_PATH = DATA_DIR / "082025_D1CONFON5F_REACHERTEST_60MIN_BEH-000_extractedsignals_raw.npy"

FPS = 7.46
FRAME_AVERAGING = 1
NUM_NEURONS = 39
NUM_FRAMES = 26887

_data_available = XLSX_PATH.exists() and SIGNAL_PATH.exists()
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not _data_available, reason="Real REACHER data files not found"),
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def shared_tmp(tmp_path_factory):
    return tmp_path_factory.mktemp("db_integration")


@pytest.fixture(scope="module")
def behavior_csv(shared_tmp):
    df = pd.read_excel(XLSX_PATH, sheet_name="Behavior Data")
    df = df.drop(columns=[c for c in df.columns if "Unnamed" in str(c)])
    out = shared_tmp / "behavior_events.csv"
    df.to_csv(out, index=False)
    return str(out)


@pytest.fixture(scope="module")
def frame_csv(shared_tmp):
    df = pd.read_excel(XLSX_PATH, sheet_name="Frame Timestamps")
    df = df.drop(columns=[c for c in df.columns if "Unnamed" in str(c)])
    df = df.iloc[1:].reset_index(drop=True)
    df.columns = ["timestamp_ms"]
    df.insert(0, "frame_index", range(len(df)))
    out = shared_tmp / "frame_timestamps.csv"
    df.to_csv(out, index=False)
    return str(out)


@pytest.fixture(scope="module")
def original_sample(behavior_csv, frame_csv):
    from pynapse.core.sample import Sample
    return Sample(
        event_data=behavior_csv,
        signal_data=str(SIGNAL_PATH),
        name="D1CONFON5F_REACHERTEST",
        fps=FPS,
        frame_averaging=FRAME_AVERAGING,
        frame_timestamps=frame_csv,
    )


@pytest.fixture(scope="module")
def db_sample(behavior_csv, frame_csv, original_sample):
    """Ingest the real REACHER sample into an in-memory DB and return a DBSample."""
    from pynapse.db import schema, ingest, query
    from pynapse.db.hydrate import DBSample

    conn = duckdb.connect(":memory:")
    schema.initialize(conn)

    project_id = ingest._get_or_create_project("ReacherTrial", conn)
    pop_id = ingest._get_or_create_population("TestPop", project_id, conn)
    subject_id = ingest._get_or_create_subject("D1CONFON5F", pop_id, conn)
    fov_id = ingest.from_sample(
        original_sample, subject_id, "reacher",
        fov_name="D1CONFON5F_REACHERTEST",
        conn=conn, copy_raw=False,
    )
    return DBSample(fov_id, conn=conn)


# ===================================================================
# Tests
# ===================================================================

class TestDBIngestion:
    def test_num_neurons(self, db_sample):
        assert db_sample.num_neurons == NUM_NEURONS

    def test_num_frames(self, db_sample):
        assert db_sample.num_frames == NUM_FRAMES

    def test_fps(self, db_sample):
        assert db_sample.fps == FPS

    def test_effective_fps(self, db_sample):
        assert db_sample.effective_fps == FPS / FRAME_AVERAGING

    def test_signals_shape(self, db_sample, original_sample):
        db_sig = db_sample.get_signals()
        orig_sig = original_sample.get_signals()
        assert db_sig.shape == orig_sig.shape

    def test_signals_match(self, db_sample, original_sample):
        db_sig = db_sample.get_signals()
        orig_sig = original_sample.get_signals().astype(np.float32)
        np.testing.assert_array_almost_equal(db_sig, orig_sig, decimal=5)

    def test_events_match(self, db_sample, original_sample):
        db_df = db_sample.get_dataframe()
        orig_df = original_sample.get_dataframe()
        assert len(db_df) == len(orig_df)
        # Codes match
        np.testing.assert_array_equal(
            db_df["code"].values, orig_df["code"].values
        )
        # Labels match
        assert list(db_df["label"].values) == list(orig_df["label"].values)

    def test_frame_timestamps_match(self, db_sample, original_sample):
        db_ts = db_sample._get_frame_timestamps()
        orig_ts = original_sample._get_frame_timestamps()
        assert len(db_ts) == len(orig_ts)
        np.testing.assert_array_almost_equal(db_ts, orig_ts)


class TestDBSampleTensor:
    """Compare SampleEventTensor outputs: original Sample vs DBSample."""

    def test_active_press_tensor_identical(self, db_sample, original_sample):
        from pynapse.analysis.peri_event import SampleEventTensor

        kwargs = dict(event_id=101, pre_event=5.0, post_event=10.0)

        orig_tensor = SampleEventTensor(sample=original_sample, **kwargs)
        db_tensor = SampleEventTensor(sample=db_sample, **kwargs)

        orig_win = orig_tensor.get_event_windows()
        db_win = db_tensor.get_event_windows()

        assert orig_win.shape == db_win.shape
        np.testing.assert_array_almost_equal(db_win, orig_win.astype(np.float32), decimal=4)

    def test_timeout_press_with_buffer(self, db_sample, original_sample):
        from pynapse.analysis.peri_event import SampleEventTensor

        kwargs = dict(event_id=102, pre_event=5.0, post_event=10.0, buffer_ms=500)

        orig_tensor = SampleEventTensor(sample=original_sample, **kwargs)
        db_tensor = SampleEventTensor(sample=db_sample, **kwargs)

        orig_win = orig_tensor.get_event_windows()
        db_win = db_tensor.get_event_windows()

        assert orig_win.shape == db_win.shape

    def test_multi_event(self, db_sample, original_sample):
        from pynapse.analysis.peri_event import SampleEventTensor

        kwargs = dict(event_id=[101, 113], pre_event=3.0, post_event=5.0)

        orig_tensor = SampleEventTensor(sample=original_sample, **kwargs)
        db_tensor = SampleEventTensor(sample=db_sample, **kwargs)

        orig_win = orig_tensor.get_event_windows()
        db_win = db_tensor.get_event_windows()

        assert orig_win.shape == db_win.shape
