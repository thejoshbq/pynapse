# test_h5_signal.py
# Tests for SignalRecording's roigbiv `.h5` trace-export support.

import numpy as np
import pandas as pd
import pytest

from pynapse.core.io.behavior import EventLog
from pynapse.core.io.microscopy import SignalRecording
from pynapse.core.sample import Sample

from conftest import create_mock_roigbiv_h5


class TestSignalRecordingH5:
    """Test suite for SignalRecording's `.h5` loading path."""

    def test_load_default_kind(self, mock_roigbiv_h5_file):
        """Loading with kind='f' returns a (neurons, frames) array."""
        rec = SignalRecording(source=mock_roigbiv_h5_file, kind="f")
        assert rec.num_neurons == 5
        assert rec.num_frames == 100
        assert rec.get_signals().shape == (5, 100)
        assert rec.kind == "f"

    def test_load_dff_kind(self, mock_roigbiv_h5_file):
        """Loading with kind='dff' reads the /dff key."""
        rec = SignalRecording(source=mock_roigbiv_h5_file, kind="dff")
        assert rec.get_signals().shape == (5, 100)
        assert rec.kind == "dff"

    def test_missing_kind_raises(self, mock_roigbiv_h5_file):
        """kind=None must raise -- there is no default trace kind."""
        with pytest.raises(ValueError, match="kind is required"):
            SignalRecording(source=mock_roigbiv_h5_file)

    def test_unknown_kind_raises(self, mock_roigbiv_h5_file):
        """An unrecognized kind string raises with the valid set listed."""
        with pytest.raises(ValueError, match="Unknown kind"):
            SignalRecording(source=mock_roigbiv_h5_file, kind="bogus")

    def test_unavailable_kind_raises_with_available_list(self, mock_roigbiv_h5_file):
        """Requesting a valid-but-absent kind lists what IS available."""
        # Fixture only writes "f" and "dff" kinds.
        with pytest.raises(ValueError, match="Available kinds"):
            SignalRecording(source=mock_roigbiv_h5_file, kind="raw")

    def test_fs_and_neuron_ids_populated(self, mock_roigbiv_h5_file):
        """The /meta-derived fs and neuron_ids properties are populated."""
        rec = SignalRecording(source=mock_roigbiv_h5_file, kind="f")
        assert rec.fs == pytest.approx(7.5)
        assert rec.neuron_ids == [f"lcl:{i}" for i in range(5)]
        assert rec.meta is not None
        assert len(rec.meta) == 5

    def test_transpose_matches_source_dataframe(self, temp_data_dir):
        """The loaded array is the transpose of the on-disk (frames, neurons) table."""
        h5_path = create_mock_roigbiv_h5(temp_data_dir, n_neurons=3, n_frames=10, fs=10.0)
        with pd.HDFStore(h5_path, mode="r") as store:
            df = store["/f"]
        rec = SignalRecording(source=h5_path, kind="f")
        np.testing.assert_allclose(rec.get_signals(), df.to_numpy(dtype=np.float32).T)

    def test_npy_path_unaffected_by_kind_param(self, temp_data_dir):
        """Legacy .npy loading ignores `kind` entirely and works as before."""
        from conftest import create_mock_signal_file

        npy_path = create_mock_signal_file(temp_data_dir, n_neurons=4, n_frames=50)
        rec = SignalRecording(source=npy_path)
        assert rec.get_signals().shape == (4, 50)
        assert rec.kind is None
        assert rec.fs is None
        assert rec.neuron_ids is None

    def test_mixed_npy_and_h5_list_raises(self, temp_data_dir, mock_roigbiv_h5_file):
        """A list mixing .npy and .h5 sources is rejected."""
        from conftest import create_mock_signal_file

        npy_path = create_mock_signal_file(temp_data_dir, n_neurons=5, n_frames=100)
        with pytest.raises(ValueError, match="Cannot mix"):
            SignalRecording(source=[npy_path, mock_roigbiv_h5_file], kind="f")

    def test_compile_multiple_h5_parts(self, temp_data_dir):
        """Multiple .h5 parts concatenate along the frames axis, like .npy parts."""
        part1 = create_mock_roigbiv_h5(
            temp_data_dir, n_neurons=3, n_frames=10, fs=10.0, filename="part1.h5"
        )
        part2 = create_mock_roigbiv_h5(
            temp_data_dir, n_neurons=3, n_frames=10, fs=10.0, filename="part2.h5"
        )
        rec = SignalRecording(source=[part1, part2], kind="f")
        assert rec.get_signals().shape == (3, 20)

    def test_mismatched_fs_across_parts_raises(self, temp_data_dir):
        """Concatenating .h5 parts with different fs is rejected."""
        part1 = create_mock_roigbiv_h5(
            temp_data_dir, n_neurons=3, n_frames=10, fs=10.0, filename="part1.h5"
        )
        part2 = create_mock_roigbiv_h5(
            temp_data_dir, n_neurons=3, n_frames=10, fs=20.0, filename="part2.h5"
        )
        with pytest.raises(ValueError, match="same fs"):
            SignalRecording(source=[part1, part2], kind="f")


@pytest.fixture
def deterministic_event_file(temp_data_dir):
    """A .mat event file with guaranteed code-22 ("active_lever") occurrences,
    timestamped (ms) to fall within a 100-frame / 7.5 Hz h5 recording (~13.3 s)."""
    import scipy.io as sio

    event_data = [[22, t] for t in (1000.0, 3000.0, 5000.0, 7000.0, 9000.0)]
    mat_path = temp_data_dir / "deterministic_events.mat"
    sio.savemat(str(mat_path), {"eventlog": np.array(event_data)})
    return str(mat_path)


class TestSampleWithH5Signal:
    """Integration: a .h5-backed SignalRecording flows through Sample unchanged."""

    def test_sample_with_h5_signal_and_external_frame_timestamps(
        self, mock_roigbiv_h5_file, deterministic_event_file, sample_event_dict
    ):
        """Sample accepts a pre-built h5 SignalRecording + explicit frame_timestamps,
        with zero special-casing inside Sample itself (mirrors the REACHER path)."""
        signal = SignalRecording(source=mock_roigbiv_h5_file, kind="f")
        frame_timestamps = np.arange(signal.num_frames) / signal.fs * 1000.0

        sample = Sample(
            event_data=deterministic_event_file,
            signal_data=signal,
            name="h5 test sample",
            event_dict=sample_event_dict,
            fps=signal.fs,
            frame_averaging=1,
            frame_timestamps=frame_timestamps,
        )

        assert sample.num_neurons == 5
        assert sample.num_frames == 100
        assert sample.effective_fps == pytest.approx(signal.fs)

        df = sample.get_dataframe()
        assert "frame_index" in df.columns
        assert sample.get_num_events(22) == 5

    def test_sample_event_tensor_extracts_windows_from_h5_signal(
        self, mock_roigbiv_h5_file, deterministic_event_file, sample_event_dict
    ):
        """SampleEventTensor extraction works unchanged against an h5-backed Sample."""
        from pynapse.analysis.peri_event import SampleEventTensor

        signal = SignalRecording(source=mock_roigbiv_h5_file, kind="f")
        frame_timestamps = np.arange(signal.num_frames) / signal.fs * 1000.0

        sample = Sample(
            event_data=deterministic_event_file,
            signal_data=signal,
            name="h5 test sample",
            event_dict=sample_event_dict,
            fps=signal.fs,
            frame_averaging=1,
            frame_timestamps=frame_timestamps,
        )

        tensor = SampleEventTensor(
            sample=sample,
            event_id=22,
            pre_event=1.0,
            post_event=1.0,
            min_trials=1,
        )
        windows = tensor.get_event_windows()
        assert windows.ndim == 3
        assert windows.shape[0] == 5  # 5 active_lever events
        assert windows.shape[1] == sample.num_neurons
