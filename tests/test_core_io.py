# test_core_io.py
# Comprehensive tests for pynapse.core.io modules

import pytest
import numpy as np
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch
import scipy.io as sio

from pynapse.core.io.behavior import EventLog
from pynapse.core.io.microscopy import SignalRecording


class TestEventLog:
    """Test suite for EventLog class."""
    
    @pytest.fixture
    def sample_event_data(self):
        """Create sample event log data."""
        return np.array([
            [22, 1000.0],
            [7, 2000.0],
            [4, 3000.0],
            [21, 4000.0],
            [22, 5000.0],
        ])
    
    @pytest.fixture
    def sample_event_dict(self):
        """Create sample event dictionary."""
        return {
            22: "active_lever",
            21: "inactive_lever",
            7: "cue",
            4: "infusion",
        }
    
    @pytest.fixture
    def temp_mat_file(self, sample_event_data, tmp_path):
        """Create a temporary .mat file with event data."""
        mat_path = tmp_path / "test_events.mat"
        sio.savemat(str(mat_path), {"eventlog": sample_event_data})
        return str(mat_path)
    
    def test_init_with_mat_file(self, temp_mat_file, sample_event_dict):
        """Test EventLog initialization with a .mat file."""
        event_log = EventLog(
            source=temp_mat_file,
            name="Test EventLog",
            event_dict=sample_event_dict
        )
        assert event_log.name == "Test EventLog"
        assert event_log.num_events == 4  # Frame trigger excluded
    
    def test_init_with_multiple_mat_files(self, sample_event_data, sample_event_dict, tmp_path):
        """Test EventLog initialization with multiple .mat files."""
        # Create two mat files
        mat_path1 = tmp_path / "test_events1.mat"
        mat_path2 = tmp_path / "test_events2.mat"
        sio.savemat(str(mat_path1), {"eventlog": sample_event_data})
        sio.savemat(str(mat_path2), {"eventlog": sample_event_data})
        
        event_log = EventLog(
            source=[str(mat_path1), str(mat_path2)],
            name="Multi EventLog",
            event_dict=sample_event_dict
        )
        # Should combine both files
        assert event_log.num_events == 8  # 4 events per file
    
    def test_count_events_no_target(self, temp_mat_file, sample_event_dict):
        """Test counting all events."""
        event_log = EventLog(source=temp_mat_file, event_dict=sample_event_dict)
        assert event_log.count_events() == 4
    
    def test_count_events_by_code(self, temp_mat_file, sample_event_dict):
        """Test counting events by code."""
        event_log = EventLog(source=temp_mat_file, event_dict=sample_event_dict)
        assert event_log.count_events(22) == 2  # Two active lever presses
        assert event_log.count_events(7) == 1   # One cue
    
    def test_count_events_by_label(self, temp_mat_file, sample_event_dict):
        """Test counting events by label."""
        event_log = EventLog(source=temp_mat_file, event_dict=sample_event_dict)
        assert event_log.count_events("active_lever") == 2
        assert event_log.count_events("cue") == 1
    
    def test_count_events_by_code_list(self, temp_mat_file, sample_event_dict):
        """Test counting events with a list of codes."""
        event_log = EventLog(source=temp_mat_file, event_dict=sample_event_dict)
        assert event_log.count_events([22, 21]) == 3  # active + inactive levers
    
    def test_get_dataframe(self, temp_mat_file, sample_event_dict):
        """Test getting DataFrame representation."""
        event_log = EventLog(source=temp_mat_file, event_dict=sample_event_dict)
        df = event_log.get_dataframe()
        
        assert "code" in df.columns
        assert "t1" in df.columns
        assert "label" in df.columns
        assert len(df) == 4
    
    def test_get_raw_data(self, temp_mat_file, sample_event_dict):
        """Test getting raw data."""
        event_log = EventLog(source=temp_mat_file, event_dict=sample_event_dict)
        raw = event_log.get_raw_data()
        assert isinstance(raw, np.ndarray)
        assert raw.shape[1] == 2  # code and timestamp columns
    
    def test_invalid_file_format(self):
        """Test that invalid file formats raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported file format"):
            EventLog(source="test.txt")
    
    def test_nonexistent_file(self):
        """Test that nonexistent files raise ValueError."""
        with pytest.raises(ValueError, match="File does not exist"):
            EventLog(source="nonexistent.mat")
    
    def test_str_representation(self, temp_mat_file):
        """Test string representation."""
        event_log = EventLog(source=temp_mat_file, name="Test")
        str_repr = str(event_log)
        assert "Test" in str_repr
        assert "Number of Events" in str_repr


class TestSignalRecording:
    """Test suite for SignalRecording class."""
    
    @pytest.fixture
    def sample_signal_data(self):
        """Create sample signal data (neurons x frames)."""
        return np.random.rand(10, 100).astype(np.float32)
    
    @pytest.fixture
    def temp_npy_file(self, sample_signal_data, tmp_path):
        """Create a temporary .npy file with signal data."""
        npy_path = tmp_path / "test_signals.npy"
        np.save(str(npy_path), sample_signal_data)
        return str(npy_path)
    
    def test_init_with_single_file(self, temp_npy_file):
        """Test SignalRecording initialization with a single file."""
        recording = SignalRecording(
            source=temp_npy_file,
            name="Test Recording"
        )
        assert recording.name == "Test Recording"
        assert recording.num_neurons == 10
        assert recording.num_frames == 100
    
    def test_init_with_multiple_files(self, sample_signal_data, tmp_path):
        """Test SignalRecording initialization with multiple files."""
        # Create two npy files
        npy_path1 = tmp_path / "test_signals1.npy"
        npy_path2 = tmp_path / "test_signals2.npy"
        np.save(str(npy_path1), sample_signal_data)
        np.save(str(npy_path2), sample_signal_data)
        
        recording = SignalRecording(
            source=[str(npy_path1), str(npy_path2)],
            name="Multi Recording"
        )
        # Should concatenate horizontally (along frames axis)
        assert recording.num_neurons == 10
        assert recording.num_frames == 200  # 100 frames per file
    
    def test_get_signals(self, temp_npy_file):
        """Test getting signal data."""
        recording = SignalRecording(source=temp_npy_file)
        signals = recording.get_signals()
        assert isinstance(signals, np.ndarray)
        assert signals.shape == (10, 100)
    
    def test_source_property(self, temp_npy_file):
        """Test source property."""
        recording = SignalRecording(source=temp_npy_file)
        assert recording.source == temp_npy_file
    
    def test_invalid_file_format(self, tmp_path):
        """Test that invalid file formats raise ValueError."""
        txt_path = tmp_path / "test.txt"
        txt_path.write_text("invalid")
        with pytest.raises(ValueError, match="Unsupported file format"):
            SignalRecording(source=str(txt_path))
    
    def test_nonexistent_file(self):
        """Test that nonexistent files raise ValueError."""
        with pytest.raises(ValueError, match="File does not exist"):
            SignalRecording(source="nonexistent.npy")
    
    def test_str_representation(self, temp_npy_file):
        """Test string representation."""
        recording = SignalRecording(source=temp_npy_file, name="Test")
        str_repr = str(recording)
        assert "Test" in str_repr
        assert "Number of Neurons" in str_repr
        assert "Number of Frames" in str_repr
    
    def test_path_object_support(self, temp_npy_file):
        """Test that Path objects are supported."""
        recording = SignalRecording(source=Path(temp_npy_file))
        assert recording.num_neurons == 10
        assert recording.num_frames == 100


# Integration tests
class TestEventLogSignalRecordingIntegration:
    """Integration tests for EventLog and SignalRecording together."""
    
    @pytest.fixture
    def create_test_data(self, tmp_path):
        """Create test event log and signal data."""
        # Create event log
        event_data = np.array([
            [9, 100.0],    # Frame trigger
            [22, 200.0],   # Active lever
            [9, 200.0],    # Frame trigger
            [7, 300.0],    # Cue
            [9, 300.0],    # Frame trigger
        ])
        mat_path = tmp_path / "test.mat"
        sio.savemat(str(mat_path), {"eventlog": event_data})
        
        # Create signal data
        signal_data = np.random.rand(5, 10).astype(np.float32)
        npy_path = tmp_path / "test.npy"
        np.save(str(npy_path), signal_data)
        
        event_dict = {22: "active_lever", 7: "cue", 9: "frame_trigger"}
        
        return str(mat_path), str(npy_path), event_dict
    
    def test_matching_dimensions(self, create_test_data):
        """Test that event log and signals can be loaded together."""
        mat_path, npy_path, event_dict = create_test_data
        
        event_log = EventLog(source=mat_path, event_dict=event_dict)
        recording = SignalRecording(source=npy_path)
        
        # Both should be loadable
        assert event_log.num_events > 0
        assert recording.num_frames > 0
