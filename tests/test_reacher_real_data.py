# test_reacher_real_data.py
# End-to-end integration test using real REACHER experimental data.
#
# Dataset: D1CONFON5F_REACHERTEST — 39 neurons, 26887 frames, 7.46 Hz
# Source:  /home/thejoshbq/Otis-Lab/Analysis/REACHER-Trial/data/FOV/

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_DIR = Path("/home/thejoshbq/Otis-Lab/Analysis/REACHER-Trial/data/FOV")
XLSX_PATH = DATA_DIR / "D1CONFON5F_REACHERTEST_1HR.xlsx"
SIGNAL_PATH = DATA_DIR / "082025_D1CONFON5F_REACHERTEST_60MIN_BEH-000_extractedsignals_raw.npy"

FPS = 7.46
FRAME_AVERAGING = 1
NUM_NEURONS = 39
NUM_FRAMES = 26887
TOTAL_EVENTS = 131

_data_available = XLSX_PATH.exists() and SIGNAL_PATH.exists()
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not _data_available, reason="Real REACHER data files not found"),
]


# ---------------------------------------------------------------------------
# Fixtures — extract CSVs from xlsx
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def shared_tmp(tmp_path_factory):
    return tmp_path_factory.mktemp("reacher_real")


@pytest.fixture(scope="module")
def behavior_csv(shared_tmp):
    """Extract behavior_events.csv from the xlsx Behavior Data sheet."""
    df = pd.read_excel(XLSX_PATH, sheet_name="Behavior Data")
    # Drop the unnamed index column
    df = df.drop(columns=[c for c in df.columns if "Unnamed" in str(c)])
    out = shared_tmp / "behavior_events.csv"
    df.to_csv(out, index=False)
    return str(out)


@pytest.fixture(scope="module")
def frame_csv(shared_tmp):
    """Extract frame_timestamps.csv from the xlsx Frame Timestamps sheet."""
    df = pd.read_excel(XLSX_PATH, sheet_name="Frame Timestamps")
    # Drop unnamed index column
    df = df.drop(columns=[c for c in df.columns if "Unnamed" in str(c)])
    # Row 0 is the system clock offset (586975), not a frame timestamp — skip it
    df = df.iloc[1:].reset_index(drop=True)
    df.columns = ["timestamp_ms"]
    df.insert(0, "frame_index", range(len(df)))
    out = shared_tmp / "frame_timestamps.csv"
    df.to_csv(out, index=False)
    return str(out)


@pytest.fixture(scope="module")
def sample(behavior_csv, frame_csv):
    """Create a Sample from the extracted CSVs and the real signal .npy."""
    from pynapse.core.sample import Sample

    return Sample(
        event_data=behavior_csv,
        signal_data=str(SIGNAL_PATH),
        name="D1CONFON5F_REACHERTEST",
        fps=FPS,
        frame_averaging=FRAME_AVERAGING,
        frame_timestamps=frame_csv,
    )


# ===================================================================
# 1. CSV Extraction Validation
# ===================================================================
class TestXlsxExtraction:
    def test_behavior_csv_exists(self, behavior_csv):
        assert Path(behavior_csv).exists()

    def test_behavior_csv_columns(self, behavior_csv):
        df = pd.read_csv(behavior_csv)
        assert set(df.columns) == {"device", "event", "start_timestamp", "end_timestamp"}

    def test_behavior_csv_row_count(self, behavior_csv):
        df = pd.read_csv(behavior_csv)
        assert len(df) == TOTAL_EVENTS

    def test_behavior_csv_devices(self, behavior_csv):
        df = pd.read_csv(behavior_csv)
        assert set(df["device"].unique()) == {"CONTROLLER", "RH_LEVER", "LH_LEVER"}

    def test_frame_csv_exists(self, frame_csv):
        assert Path(frame_csv).exists()

    def test_frame_csv_columns(self, frame_csv):
        df = pd.read_csv(frame_csv)
        assert list(df.columns) == ["frame_index", "timestamp_ms"]

    def test_frame_csv_row_count(self, frame_csv):
        df = pd.read_csv(frame_csv)
        assert len(df) == NUM_FRAMES


# ===================================================================
# 2. Sample Creation & Properties
# ===================================================================
class TestReacherSampleCreation:
    def test_num_neurons(self, sample):
        assert sample.num_neurons == NUM_NEURONS

    def test_num_frames(self, sample):
        assert sample.num_frames == NUM_FRAMES

    def test_fps(self, sample):
        assert sample.fps == FPS

    def test_effective_fps(self, sample):
        assert sample.effective_fps == FPS / FRAME_AVERAGING

    def test_total_events(self, sample):
        assert sample.count_events() == TOTAL_EVENTS

    def test_active_press_count(self, sample):
        assert sample.count_events(101) == 22

    def test_timeout_press_count(self, sample):
        assert sample.count_events(102) == 94

    def test_inactive_press_count(self, sample):
        assert sample.count_events(113) == 13

    def test_controller_start(self, sample):
        assert sample.count_events(701) == 1

    def test_controller_end(self, sample):
        assert sample.count_events(702) == 1


# ===================================================================
# 3. DataFrame & Alignment
# ===================================================================
class TestReacherAlignment:
    def test_dataframe_columns(self, sample):
        df = sample.get_dataframe()
        assert list(df.columns) == ["code", "t1", "t2", "label", "frame_index"]

    def test_frame_index_range(self, sample):
        df = sample.get_dataframe()
        assert df["frame_index"].min() >= 0
        assert df["frame_index"].max() <= NUM_FRAMES - 1

    def test_timestamps_non_negative(self, sample):
        df = sample.get_dataframe()
        assert (df["t1"] >= 0).all()
        assert (df["t2"] >= df["t1"]).all()

    def test_labels_present(self, sample):
        df = sample.get_dataframe()
        labels = set(df["label"].unique())
        expected = {
            "rh_lever_active_press",
            "rh_lever_timeout_press",
            "lh_lever_inactive_press",
            "controller_start",
            "controller_end",
        }
        assert expected.issubset(labels)

    def test_frame_timestamps_properties(self, sample):
        ts = sample._get_frame_timestamps()
        assert len(ts) == NUM_FRAMES
        # Monotonically increasing
        assert (np.diff(ts) > 0).all()
        # First timestamp ~134 ms, last ~3602987 ms
        assert 100 < ts[0] < 200
        assert 3_500_000 < ts[-1] < 3_700_000


# ===================================================================
# 4. Peri-Event Tensor Extraction
# ===================================================================
class TestReacherPeriEvent:
    def test_active_press_tensor(self, sample):
        from pynapse.analysis.peri_event import SampleEventTensor

        tensor = SampleEventTensor(
            sample=sample, event_id=101, pre_event=5.0, post_event=10.0,
        )
        windows = tensor.get_event_windows()
        pre_frames = int(5.0 * FPS)   # 37
        post_frames = int(10.0 * FPS)  # 74
        window_size = pre_frames + post_frames  # 111
        assert windows.ndim == 3
        assert windows.shape[1] == NUM_NEURONS
        assert windows.shape[2] == window_size
        assert 1 <= windows.shape[0] <= 22
        assert not np.isnan(windows).any()
        assert (windows >= 0).all()  # raw fluorescence

    def test_timeout_press_tensor_with_buffer(self, sample):
        from pynapse.analysis.peri_event import SampleEventTensor

        tensor = SampleEventTensor(
            sample=sample, event_id=102, pre_event=5.0, post_event=10.0,
            buffer_ms=500,
        )
        windows = tensor.get_event_windows()
        assert windows.ndim == 3
        assert windows.shape[1] == NUM_NEURONS
        assert windows.shape[2] == int(5.0 * FPS) + int(10.0 * FPS)
        # Buffer filtering should reduce trial count below 94
        assert windows.shape[0] <= 94

    def test_multi_event_tensor(self, sample):
        from pynapse.analysis.peri_event import SampleEventTensor

        tensor = SampleEventTensor(
            sample=sample, event_id=[101, 113], pre_event=3.0, post_event=5.0,
        )
        windows = tensor.get_event_windows()
        window_size = int(3.0 * FPS) + int(5.0 * FPS)  # 22 + 37 = 59
        assert windows.shape[1] == NUM_NEURONS
        assert windows.shape[2] == window_size
        assert windows.shape[0] <= 22 + 13

    def test_insufficient_trials_returns_empty(self, sample):
        from pynapse.analysis.peri_event import SampleEventTensor

        tensor = SampleEventTensor(
            sample=sample, event_id=113, pre_event=5.0, post_event=10.0,
            min_trials=100,
        )
        windows = tensor.get_event_windows()
        assert windows.shape[0] == 0

    def test_at_least_one_active_trial(self, sample):
        from pynapse.analysis.peri_event import SampleEventTensor

        tensor = SampleEventTensor(
            sample=sample, event_id=101, pre_event=5.0, post_event=10.0,
        )
        windows = tensor.get_event_windows()
        assert windows.shape[0] >= 1


# ===================================================================
# 5. Preprocessing Integration
# ===================================================================
class TestReacherPreprocessing:
    def test_df_over_f(self, sample):
        from pynapse.analysis.peri_event import SampleEventTensor
        from pynapse.analysis.preprocessing.epoch.relative_change import DFOverF

        tensor = SampleEventTensor(
            sample=sample, event_id=101, pre_event=5.0, post_event=10.0,
            trace_preprocess=DFOverF(percentile=8),
        )
        windows = tensor.get_event_windows()
        assert windows.ndim == 3
        assert windows.shape[1] == NUM_NEURONS
        assert windows.shape[2] == int(5.0 * FPS) + int(10.0 * FPS)
        assert windows.dtype == np.float32
        # DF/F should include negative values (below-baseline activity)
        assert (windows < 0).any()

    def test_otis_pipe(self, sample):
        from pynapse.analysis.peri_event import SampleEventTensor
        from pynapse.analysis.preprocessing.epoch.pipelines import OTIS_PIPE

        tensor = SampleEventTensor(
            sample=sample, event_id=101, pre_event=5.0, post_event=10.0,
            window_preprocess=OTIS_PIPE,
        )
        windows = tensor.get_event_windows()
        assert windows.ndim == 3
        assert windows.shape[1] == NUM_NEURONS
        assert windows.shape[2] == int(5.0 * FPS) + int(10.0 * FPS)
        assert windows.dtype == np.float32


# ===================================================================
# 6. Tensor Caching via get_tensor()
# ===================================================================
class TestReacherCaching:
    def test_get_tensor_with_defaults(self, behavior_csv, frame_csv):
        from pynapse.core.sample import Sample

        s = Sample(
            event_data=behavior_csv,
            signal_data=str(SIGNAL_PATH),
            name="D1CONFON5F_REACHERTEST_cached",
            fps=FPS,
            frame_averaging=FRAME_AVERAGING,
            frame_timestamps=frame_csv,
            default_event_id=101,
            default_pre_event=5.0,
            default_post_event=10.0,
        )
        t1 = s.get_tensor()
        t2 = s.get_tensor()
        assert t1 is t2  # cache hit — same object

    def test_cache_miss_on_different_params(self, behavior_csv, frame_csv):
        from pynapse.core.sample import Sample

        s = Sample(
            event_data=behavior_csv,
            signal_data=str(SIGNAL_PATH),
            name="D1CONFON5F_REACHERTEST_cached2",
            fps=FPS,
            frame_averaging=FRAME_AVERAGING,
            frame_timestamps=frame_csv,
            default_event_id=101,
            default_pre_event=5.0,
            default_post_event=10.0,
        )
        t1 = s.get_tensor()
        t2 = s.get_tensor(event_id=102)
        assert t1 is not t2  # different params — different tensor
