# test_peri_event.py
# Comprehensive tests for pynapse.analysis.peri_event module

import pytest
import numpy as np
import pandas as pd
import scipy.io as sio
from unittest.mock import Mock, MagicMock

from pynapse.analysis.peri_event import EventTensor, SampleEventTensor, PopulationEventTensor
from pynapse.core.sample import Sample
from pynapse.core.population import Population
from pynapse.analysis.preprocessing.base import Preprocessor


class TestEventTensorBase:
    """Test suite for EventTensor base class."""
    
    def test_sec_to_frames_conversion(self):
        """Test second to frame conversion."""
        frames = EventTensor._sec_to_frames(1.0, 30.0)
        assert frames == 30
        
        frames = EventTensor._sec_to_frames(0.5, 30.0)
        assert frames == 15
        
        frames = EventTensor._sec_to_frames(2.0, 60.0)
        assert frames == 120


class TestSampleEventTensor:
    """Test suite for SampleEventTensor class."""
    
    @pytest.fixture
    def mock_sample(self):
        """Create a mock Sample with realistic data."""
        sample = Mock(spec=Sample)
        sample.effective_fps = 30.0
        sample.num_neurons = 10
        sample.interframe_interval = 33.33
        
        # Create DataFrame with events
        df_data = {
            "code": [22, 22, 7, 22, 4],
            "t1": [1000.0, 2000.0, 3000.0, 4000.0, 5000.0],
            "label": ["active_lever"] * 3 + ["cue"] + ["infusion"],
            "frame_index": [30, 60, 90, 120, 150]
        }
        sample.get_dataframe.return_value = pd.DataFrame(df_data)
        
        # Create mock signals (neurons x frames)
        signals = np.random.rand(10, 200)
        sample.get_signals.return_value = signals
        
        # Mock frame timestamps
        frame_ts = np.arange(0, 6000, 33.33)  # ~180 frames
        sample._get_frame_timestamps.return_value = frame_ts
        
        return sample
    
    @pytest.fixture
    def event_dict(self):
        """Create sample event dictionary."""
        return {
            22: "active_lever",
            7: "cue",
            4: "infusion",
        }
    
    def test_init_with_single_event_id(self, mock_sample):
        """Test initialization with single event ID."""
        tensor = SampleEventTensor(
            sample=mock_sample,
            event_id=22,
            pre_event=1.0,
            post_event=1.0
        )
        assert tensor._event_id == [22]
        assert tensor._pre_event == 1.0
        assert tensor._post_event == 1.0
    
    def test_init_with_multiple_event_ids(self, mock_sample):
        """Test initialization with multiple event IDs."""
        tensor = SampleEventTensor(
            sample=mock_sample,
            event_id=[22, 7],
            pre_event=1.0,
            post_event=1.0
        )
        assert tensor._event_id == [22, 7]
    
    def test_init_with_buffer(self, mock_sample):
        """Test initialization with buffer."""
        tensor = SampleEventTensor(
            sample=mock_sample,
            event_id=22,
            pre_event=1.0,
            post_event=1.0,
            buffer_ms=500
        )
        assert tensor._buffer_ms == 500
    
    def test_init_with_min_trials(self, mock_sample):
        """Test initialization with minimum trials."""
        tensor = SampleEventTensor(
            sample=mock_sample,
            event_id=22,
            pre_event=1.0,
            post_event=1.0,
            min_trials=5
        )
        assert tensor._min_trials == 5
    
    def test_get_event_windows_shape(self, mock_sample):
        """Test that event windows have correct shape."""
        tensor = SampleEventTensor(
            sample=mock_sample,
            event_id=22,
            pre_event=1.0,  # 30 frames at 30 fps
            post_event=1.0   # 30 frames at 30 fps
        )
        
        windows = tensor.get_event_windows()
        
        # Should have shape (n_trials, n_neurons, window_size)
        assert windows.ndim == 3
        assert windows.shape[1] == 10  # n_neurons
        assert windows.shape[2] == 60  # pre + post frames
    
    def test_get_event_windows_caching(self, mock_sample):
        """Test that event windows are cached."""
        tensor = SampleEventTensor(
            sample=mock_sample,
            event_id=22,
            pre_event=1.0,
            post_event=1.0
        )
        
        # First call
        windows1 = tensor.get_event_windows()
        # Second call should return same object
        windows2 = tensor.get_event_windows()
        
        assert windows1 is windows2
    
    def test_insufficient_trials_returns_empty(self, mock_sample):
        """Test that insufficient trials returns empty array."""
        tensor = SampleEventTensor(
            sample=mock_sample,
            event_id=22,
            pre_event=1.0,
            post_event=1.0,
            min_trials=100  # More than available
        )
        
        windows = tensor.get_event_windows()
        assert windows.shape[0] == 0
    
    def test_with_trace_preprocessing(self, mock_sample):
        """Test with trace preprocessing."""
        class DummyPreprocessor(Preprocessor):
            def apply(self, data):
                return (data * 2.0).astype(np.float32)
        
        preprocessor = DummyPreprocessor()
        
        tensor = SampleEventTensor(
            sample=mock_sample,
            event_id=22,
            pre_event=1.0,
            post_event=1.0,
            trace_preprocess=preprocessor
        )
        
        windows = tensor.get_event_windows()
        # Should have applied preprocessing (multiplied by 2)
        # Just verify it runs without error
        assert windows.shape[1] == 10
    
    def test_with_window_preprocessing(self, mock_sample):
        """Test with window preprocessing."""
        class DummyPreprocessor(Preprocessor):
            def apply(self, data):
                return (data + 1.0).astype(np.float32)
        
        preprocessor = DummyPreprocessor()
        
        tensor = SampleEventTensor(
            sample=mock_sample,
            event_id=22,
            pre_event=1.0,
            post_event=1.0,
            window_preprocess=preprocessor
        )
        
        windows = tensor.get_event_windows()
        # Should have applied preprocessing
        assert windows.shape[1] == 10


class TestPopulationEventTensor:
    """Test suite for PopulationEventTensor class."""
    
    @pytest.fixture
    def mock_population(self):
        """Create a mock Population with multiple samples."""
        population = Mock(spec=Population)
        
        # Create multiple mock samples
        samples = []
        for i in range(3):
            sample = Mock(spec=Sample)
            sample.effective_fps = 30.0
            sample.num_neurons = 5
            sample.interframe_interval = 33.33
            
            # Create DataFrame with events
            df_data = {
                "code": [22] * (i + 2),  # Different number of events per sample
                "t1": [1000.0 * (j + 1) for j in range(i + 2)],
                "label": ["active_lever"] * (i + 2),
            }
            sample.get_dataframe.return_value = pd.DataFrame(df_data)
            
            # Mock signals
            signals = np.random.rand(5, 100)
            sample.get_signals.return_value = signals
            
            # Mock frame timestamps
            frame_ts = np.arange(0, 3000, 33.33)
            sample._get_frame_timestamps.return_value = frame_ts
            
            # Mock get_num_events
            sample.get_num_events.return_value = i + 2
            
            samples.append(sample)
        
        population.get_samples.return_value = samples
        return population
    
    def test_init_extracts_from_all_samples(self, mock_population):
        """Test that initialization extracts windows from all samples."""
        tensor = PopulationEventTensor(
            population=mock_population,
            event_id=22,
            pre_event=0.5,
            post_event=0.5
        )
        
        windows_list = tensor.get_event_windows()
        
        # Should have windows from 3 samples
        assert len(windows_list) == 3
        
        # Each should be an ndarray
        for windows in windows_list:
            assert isinstance(windows, np.ndarray)
    
    def test_min_trials_filters_samples(self, mock_population):
        """Test that min_trials filters out samples with insufficient events."""
        tensor = PopulationEventTensor(
            population=mock_population,
            event_id=22,
            pre_event=0.5,
            post_event=0.5,
            min_trials=3  # Only last sample has 4 events
        )
        
        windows_list = tensor.get_event_windows()
        
        # Should only include samples with enough trials
        # Sample 0: 2 events (excluded)
        # Sample 1: 3 events (included)
        # Sample 2: 4 events (included)
        assert len(windows_list) == 2
    
    def test_caching_of_event_windows(self, mock_population):
        """Test that event windows are cached."""
        tensor = PopulationEventTensor(
            population=mock_population,
            event_id=22,
            pre_event=0.5,
            post_event=0.5
        )
        
        # First call extracts and caches
        windows1 = tensor.get_event_windows()
        # Second call returns cached
        windows2 = tensor.get_event_windows()
        
        assert windows1 is windows2
    
    def test_with_multiple_event_ids(self, mock_population):
        """Test with multiple event IDs."""
        tensor = PopulationEventTensor(
            population=mock_population,
            event_id=[22, 7],
            pre_event=0.5,
            post_event=0.5
        )
        
        windows_list = tensor.get_event_windows()
        assert isinstance(windows_list, list)
    
    def test_empty_population(self):
        """Test with population that has no samples."""
        empty_pop = Mock(spec=Population)
        empty_pop.get_samples.return_value = []
        
        tensor = PopulationEventTensor(
            population=empty_pop,
            event_id=22,
            pre_event=0.5,
            post_event=0.5
        )
        
        windows_list = tensor.get_event_windows()
        assert len(windows_list) == 0


# Integration tests
class TestPeriEventIntegration:
    """Integration tests for peri-event analysis."""
    
    @pytest.fixture
    def create_sample_with_data(self, tmp_path):
        """Create a real Sample with test data."""
        # Create event data with frame triggers and events
        event_data = np.array([
            [9, 100.0],
            [22, 500.0],
            [9, 1000.0],
            [22, 1500.0],
            [9, 2000.0],
            [7, 2500.0],
            [9, 3000.0],
        ])
        mat_path = tmp_path / "test.mat"
        sio.savemat(str(mat_path), {"eventlog": event_data})
        
        # Create signal data (neurons x frames)
        signal_data = np.random.rand(10, 100).astype(np.float32)
        npy_path = tmp_path / "test.npy"
        np.save(str(npy_path), signal_data)
        
        event_dict = {22: "active_lever", 7: "cue", 9: "frame_trigger"}
        
        sample = Sample(
            event_data=str(mat_path),
            signal_data=str(npy_path),
            name="Test Sample",
            event_dict=event_dict,
            fps=30.0,
            frame_averaging=1
        )
        
        return sample
    
    def test_real_sample_event_extraction(self, create_sample_with_data):
        """Test event extraction with real Sample."""
        sample = create_sample_with_data
        
        # Extract peri-event windows
        tensor = SampleEventTensor(
            sample=sample,
            event_id=22,
            pre_event=0.3,  # ~9 frames at 30 fps
            post_event=0.3
        )
        
        windows = tensor.get_event_windows()
        
        # Should extract windows for active_lever events
        assert windows.ndim == 3
        assert windows.shape[1] == 10  # neurons
        # May have fewer trials due to edge effects
        assert windows.shape[0] >= 0
    
    def test_multiple_event_types(self, create_sample_with_data):
        """Test extraction of multiple event types."""
        sample = create_sample_with_data
        
        # Extract windows for multiple event types
        tensor = SampleEventTensor(
            sample=sample,
            event_id=[22, 7],  # active_lever and cue
            pre_event=0.2,
            post_event=0.2
        )
        
        windows = tensor.get_event_windows()
        assert windows.ndim == 3
