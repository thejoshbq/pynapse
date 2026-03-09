# conftest.py
# Shared pytest fixtures and configuration for pynapse tests

import pytest
import numpy as np
import scipy.io as sio
import tempfile
from pathlib import Path


@pytest.fixture
def sample_event_dict():
    """
    Standard event dictionary for testing.
    
    Returns a dictionary mapping event codes to labels
    compatible with LEGACY_HER format.
    """
    return {
        22: "active_lever",
        222: "active_lever_timeout",
        21: "inactive_lever",
        212: "inactive_lever_timeout",
        7: "cue",
        4: "infusion",
        9: "frame_trigger",
    }


@pytest.fixture
def random_seed():
    """
    Set random seed for reproducible tests.
    """
    np.random.seed(42)


@pytest.fixture
def temp_data_dir(tmp_path):
    """
    Create a temporary directory for test data.
    
    Returns a Path object to the temporary directory.
    """
    data_dir = tmp_path / "test_data"
    data_dir.mkdir(exist_ok=True)
    return data_dir


def create_mock_event_file(path: Path, n_events: int = 10) -> str:
    """
    Create a mock MATLAB .mat file with event data.
    
    Args:
        path: Path where to save the file
        n_events: Number of events to generate
    
    Returns:
        String path to the created file
    """
    event_codes = [22, 7, 4, 21]
    event_data = []
    
    # Generate frame triggers and events
    time = 0.0
    for i in range(n_events):
        # Frame trigger
        event_data.append([9, time])
        time += 33.33
        
        # Random event
        if np.random.rand() > 0.5:
            code = np.random.choice(event_codes)
            event_data.append([code, time])
            time += 10.0
    
    event_array = np.array(event_data)
    mat_path = path / "events.mat"
    sio.savemat(str(mat_path), {"eventlog": event_array})
    return str(mat_path)


def create_mock_signal_file(path: Path, n_neurons: int = 10, n_frames: int = 100) -> str:
    """
    Create a mock .npy file with signal data.
    
    Args:
        path: Path where to save the file
        n_neurons: Number of neurons
        n_frames: Number of frames
    
    Returns:
        String path to the created file
    """
    signal_data = np.random.rand(n_neurons, n_frames).astype(np.float32)
    npy_path = path / "signals.npy"
    np.save(str(npy_path), signal_data)
    return str(npy_path)


@pytest.fixture
def mock_data_files(temp_data_dir):
    """
    Create mock event and signal data files for testing.
    
    Returns a tuple of (event_file_path, signal_file_path, event_dict)
    """
    event_dict = {
        22: "active_lever",
        21: "inactive_lever",
        7: "cue",
        4: "infusion",
        9: "frame_trigger",
    }
    
    event_file = create_mock_event_file(temp_data_dir, n_events=20)
    signal_file = create_mock_signal_file(temp_data_dir, n_neurons=15, n_frames=100)
    
    return event_file, signal_file, event_dict


def create_mock_reacher_behavior_csv(path: Path, n_events: int = 20) -> str:
    """Create a mock REACHER behavior_events.csv file.

    Generates a CSV in the format produced by the REACHER backend:
    device,event,start_timestamp,end_timestamp,start_frame_index,end_frame_index

    Timestamps are in milliseconds (session-relative).

    Args:
        path: Directory where the file is created.
        n_events: Approximate number of behavioral events to generate.

    Returns:
        String path to the created CSV file.
    """
    import csv as _csv

    devices_events = [
        ("RH_LEVER", "ACTIVE_PRESS"),
        ("RH_LEVER", "TIMEOUT_PRESS"),
        ("RH_LEVER", "INACTIVE_PRESS"),
        ("PUMP", "INFUSION"),
        ("CUE", "TONE"),
        ("LICK", "LICK"),
    ]

    csv_path = path / "behavior_events.csv"
    ts = 500  # start 500 ms into the session
    with open(csv_path, "w", newline="") as f:
        writer = _csv.DictWriter(
            f,
            fieldnames=["device", "event", "start_timestamp", "end_timestamp",
                        "start_frame_index", "end_frame_index"],
        )
        writer.writeheader()
        for i in range(n_events):
            dev, evt = devices_events[i % len(devices_events)]
            duration = 200 if dev != "PUMP" else 2000
            writer.writerow({
                "device": dev,
                "event": evt,
                "start_timestamp": ts,
                "end_timestamp": ts + duration,
                "start_frame_index": ts // 33,
                "end_frame_index": (ts + duration) // 33,
            })
            ts += duration + np.random.randint(300, 2000)

    return str(csv_path)


def create_mock_reacher_frame_csv(path: Path, n_frames: int = 100) -> str:
    """Create a mock REACHER frame_timestamps.csv file.

    Args:
        path: Directory where the file is created.
        n_frames: Number of frames.

    Returns:
        String path to the created CSV file.
    """
    import csv as _csv

    csv_path = path / "frame_timestamps.csv"
    with open(csv_path, "w", newline="") as f:
        writer = _csv.DictWriter(f, fieldnames=["frame_index", "timestamp_ms"])
        writer.writeheader()
        for i in range(n_frames):
            writer.writerow({"frame_index": i, "timestamp_ms": i * 33})

    return str(csv_path)


@pytest.fixture
def mock_reacher_data_files(temp_data_dir):
    """Create mock REACHER event CSV, frame CSV, and signal data files.

    Returns a tuple of (behavior_csv_path, frame_csv_path, signal_npy_path).
    """
    behavior_csv = create_mock_reacher_behavior_csv(temp_data_dir, n_events=20)
    frame_csv = create_mock_reacher_frame_csv(temp_data_dir, n_frames=100)
    signal_file = create_mock_signal_file(temp_data_dir, n_neurons=15, n_frames=100)
    return behavior_csv, frame_csv, signal_file


# Configure pytest to show more detailed output
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )


# Custom assertion helpers
class Helpers:
    """Custom helper functions for tests."""
    
    @staticmethod
    def assert_shape_equals(array, expected_shape):
        """Assert that array has expected shape."""
        assert array.shape == expected_shape, \
            f"Expected shape {expected_shape}, got {array.shape}"
    
    @staticmethod
    def assert_array_positive(array):
        """Assert that all array values are positive."""
        assert np.all(array >= 0), "Array contains negative values"
    
    @staticmethod
    def assert_valid_signal_data(signals):
        """Assert that signal data has valid shape and properties."""
        assert signals.ndim == 2, f"Expected 2D array, got {signals.ndim}D"
        assert signals.shape[0] > 0, "No neurons in signal data"
        assert signals.shape[1] > 0, "No frames in signal data"
        assert not np.isnan(signals).any(), "Signal data contains NaN values"


@pytest.fixture
def helpers():
    """Provide helper functions to tests."""
    return Helpers()
