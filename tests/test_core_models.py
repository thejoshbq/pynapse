# test_core_models.py
# Comprehensive tests for pynapse.core models

import pytest
import numpy as np
import pandas as pd
import scipy.io as sio
from unittest.mock import Mock, MagicMock, patch

from pynapse.core.sample import Sample
from pynapse.core.population import Population
from pynapse.core.project import Project
from pynapse.core.io.behavior import EventLog
from pynapse.core.io.microscopy import SignalRecording


class TestSample:
    """Test suite for Sample class."""
    
    @pytest.fixture
    def mock_event_log(self):
        """Create a mock EventLog."""
        mock_log = Mock(spec=EventLog)
        mock_log.count_events.return_value = 10
        mock_log.num_events = 10
        mock_log.source = "test_events.mat"
        
        # Mock dataframe with frame triggers and events
        df_data = {
            "code": [9, 22, 9, 7, 9, 4, 9],
            "t1": [100.0, 150.0, 200.0, 250.0, 300.0, 350.0, 400.0],
            "label": ["frame_trigger", "active_lever", "frame_trigger", "cue", 
                     "frame_trigger", "infusion", "frame_trigger"]
        }
        mock_log.get_dataframe.return_value = pd.DataFrame(df_data)
        
        # Mock raw data with frame triggers
        raw_data = np.array([
            [9, 100.0],
            [22, 150.0],
            [9, 200.0],
            [7, 250.0],
            [9, 300.0],
            [4, 350.0],
            [9, 400.0],
        ])
        mock_log.get_raw_data.return_value = raw_data
        
        return mock_log
    
    @pytest.fixture
    def mock_signal_recording(self):
        """Create a mock SignalRecording."""
        mock_rec = Mock(spec=SignalRecording)
        mock_rec.num_neurons = 20
        mock_rec.num_frames = 100
        mock_rec.source = "test_signals.npy"
        mock_rec.get_signals.return_value = np.random.rand(20, 100)
        return mock_rec
    
    @pytest.fixture
    def sample_event_dict(self):
        """Create sample event dictionary."""
        return {
            22: "active_lever",
            21: "inactive_lever",
            7: "cue",
            4: "infusion",
            9: "frame_trigger",
        }
    
    def test_init_with_event_log_and_signal_recording_objects(
        self, mock_event_log, mock_signal_recording, sample_event_dict
    ):
        """Test Sample initialization with EventLog and SignalRecording objects."""
        sample = Sample(
            event_data=mock_event_log,
            signal_data=mock_signal_recording,
            name="Test Sample",
            event_dict=sample_event_dict,
            fps=30.0,
            frame_averaging=4
        )
        
        assert sample.name == "Test Sample"
        assert sample.num_neurons == 20
        assert sample.num_frames == 100
        assert sample.fps == 30.0
        assert sample.frame_averaging == 4
        assert sample.effective_fps == 7.5
    
    def test_fps_calculations(self, mock_event_log, mock_signal_recording):
        """Test FPS-related calculations."""
        sample = Sample(
            event_data=mock_event_log,
            signal_data=mock_signal_recording,
            fps=30.0,
            frame_averaging=4
        )
        
        assert sample.fps == 30.0
        assert sample.effective_fps == 7.5  # 30 / 4
        assert sample.frame_duration == pytest.approx(0.1333, rel=1e-3)  # 1 / 7.5
        assert sample.interframe_interval == pytest.approx(33.3333, rel=1e-3)  # 1000 / 30
    
    def test_count_events(self, mock_event_log, mock_signal_recording):
        """Test event counting."""
        sample = Sample(
            event_data=mock_event_log,
            signal_data=mock_signal_recording
        )
        
        mock_event_log.count_events.return_value = 5
        assert sample.count_events() == 5
    
    def test_get_signals(self, mock_event_log, mock_signal_recording):
        """Test getting signal data."""
        sample = Sample(
            event_data=mock_event_log,
            signal_data=mock_signal_recording
        )
        
        signals = sample.get_signals()
        assert signals.shape == (20, 100)
    
    def test_get_event_dict(self, mock_event_log, mock_signal_recording, sample_event_dict):
        """Test getting event dictionary."""
        sample = Sample(
            event_data=mock_event_log,
            signal_data=mock_signal_recording,
            event_dict=sample_event_dict
        )
        
        assert sample.get_event_dict() == sample_event_dict
    
    def test_empty_event_log_raises_error(self, mock_signal_recording):
        """Test that empty event log raises ValueError."""
        empty_log = Mock(spec=EventLog)
        empty_log.count_events.return_value = 0
        empty_log.num_events = 0
        
        with pytest.raises(ValueError, match="Event log or signal recording is empty"):
            Sample(
                event_data=empty_log,
                signal_data=mock_signal_recording
            )
    
    def test_empty_signal_recording_raises_error(self, mock_event_log):
        """Test that empty signal recording raises ValueError."""
        empty_rec = Mock(spec=SignalRecording)
        empty_rec.num_frames = 0
        
        with pytest.raises(ValueError, match="Event log or signal recording is empty"):
            Sample(
                event_data=mock_event_log,
                signal_data=empty_rec
            )
    
    def test_str_representation(self, mock_event_log, mock_signal_recording):
        """Test string representation."""
        sample = Sample(
            event_data=mock_event_log,
            signal_data=mock_signal_recording,
            name="Test Sample",
            fps=30.0
        )
        
        str_repr = str(sample)
        assert "Test Sample" in str_repr
        assert "FPS: 30.0" in str_repr
        assert "Number of Neurons: 20" in str_repr
        assert "Number of Frames: 100" in str_repr
    
    @pytest.mark.parametrize("fps,expected_frames", [
        (30.0, 30),
        (60.0, 60),
        (15.0, 15),
    ])
    def test_sec_to_frames(self, fps, expected_frames):
        """Test second to frame conversion."""
        frames = Sample._sec_to_frames(1.0, fps)
        assert frames == expected_frames


class TestPopulation:
    """Test suite for Population class."""
    
    @pytest.fixture
    def mock_samples(self):
        """Create mock samples."""
        samples = []
        for i in range(3):
            sample = Mock(spec=Sample)
            sample.name = f"Sample {i+1}"
            sample.num_neurons = 10 * (i + 1)
            sample.num_events = 5 * (i + 1)
            samples.append(sample)
        return samples
    
    def test_init(self, mock_samples):
        """Test Population initialization."""
        population = Population(
            samples=mock_samples,
            name="Test Population",
            description="A test population"
        )
        
        assert population.name == "Test Population"
        assert population.description == "A test population"
        assert population.num_samples == 3
    
    def test_num_neurons_aggregation(self, mock_samples):
        """Test aggregation of neurons across samples."""
        population = Population(samples=mock_samples, name="Test")
        # 10 + 20 + 30 = 60
        assert population.num_neurons == 60
    
    def test_num_events_aggregation(self, mock_samples):
        """Test aggregation of events across samples."""
        population = Population(samples=mock_samples, name="Test")
        # 5 + 10 + 15 = 30
        assert population.num_events == 30
    
    def test_get_samples(self, mock_samples):
        """Test getting samples."""
        population = Population(samples=mock_samples, name="Test")
        samples = population.get_samples()
        assert len(samples) == 3
        assert samples == mock_samples
    
    def test_default_name(self, mock_samples):
        """Test default name when none provided."""
        population = Population(samples=mock_samples)
        assert population.name == "Population"
    
    def test_str_representation(self, mock_samples):
        """Test string representation."""
        population = Population(
            samples=mock_samples,
            name="Test Population",
            description="Test description"
        )
        
        str_repr = str(population)
        assert "Test Population" in str_repr
        assert "Test description" in str_repr
        assert "Sample 1" in str_repr
        assert "Sample 2" in str_repr
        assert "Sample 3" in str_repr
    
    def test_empty_population(self):
        """Test population with no samples."""
        population = Population(samples=[], name="Empty")
        assert population.num_samples == 0
        assert population.num_neurons == 0
        assert population.num_events == 0


class TestProject:
    """Test suite for Project class."""
    
    @pytest.fixture
    def mock_populations(self):
        """Create mock populations."""
        populations = []
        for i in range(2):
            pop = Mock(spec=Population)
            pop.name = f"Population {i+1}"
            pop.num_samples = 3
            pop.num_neurons = 50 * (i + 1)
            populations.append(pop)
        return populations
    
    def test_init(self, mock_populations):
        """Test Project initialization."""
        project = Project(
            name="Test Project",
            populations=mock_populations,
            authors=["Author 1", "Author 2"],
            description="A test project"
        )
        
        assert project.name == "Test Project"
        assert project.description == "A test project"
        assert len(project.get_authors()) == 2
        assert project.num_populations == 2
    
    def test_get_populations(self, mock_populations):
        """Test getting populations."""
        project = Project(
            name="Test Project",
            populations=mock_populations
        )
        
        pops = project.get_populations()
        assert len(pops) == 2
        assert pops == mock_populations
    
    def test_get_authors(self):
        """Test getting authors."""
        authors = ["Alice", "Bob", "Charlie"]
        project = Project(
            name="Test",
            populations=[],
            authors=authors
        )
        
        assert project.get_authors() == authors
    
    def test_default_name(self, mock_populations):
        """Test default name when none provided."""
        project = Project(populations=mock_populations)
        assert project.name == "Project"
    
    def test_str_representation(self, mock_populations):
        """Test string representation."""
        project = Project(
            name="Test Project",
            populations=mock_populations,
            description="Test description"
        )
        
        str_repr = str(project)
        assert "Test Project" in str_repr
        assert "Test description" in str_repr
        assert "Population 1" in str_repr
        assert "Population 2" in str_repr
    
    def test_empty_project(self):
        """Test project with no populations."""
        project = Project(
            name="Empty Project",
            populations=[]
        )
        assert project.num_populations == 0


class TestReacherSample:
    """Test suite for Sample with REACHER data."""

    def test_init_with_reacher_csv_and_frame_csv(self, mock_reacher_data_files):
        """Test Sample creation using REACHER CSV files."""
        behavior_csv, frame_csv, signal_npy = mock_reacher_data_files
        sample = Sample(
            event_data=behavior_csv,
            signal_data=signal_npy,
            name="REACHER Sample",
            fps=30.0,
            frame_timestamps=frame_csv,
        )
        assert sample.name == "REACHER Sample"
        assert sample.num_neurons == 15
        assert sample.num_frames == 100
        assert sample.num_events > 0

    def test_frame_timestamps_from_csv(self, mock_reacher_data_files):
        """Test that frame timestamps from CSV are used for alignment."""
        behavior_csv, frame_csv, signal_npy = mock_reacher_data_files
        sample = Sample(
            event_data=behavior_csv,
            signal_data=signal_npy,
            fps=30.0,
            frame_timestamps=frame_csv,
        )
        df = sample.get_dataframe()
        assert "frame_index" in df.columns
        assert (df["frame_index"] >= 0).all()

    def test_frame_timestamps_from_array(self, mock_reacher_data_files):
        """Test that frame timestamps from a numpy array work."""
        behavior_csv, _, signal_npy = mock_reacher_data_files
        frame_ts = np.arange(100) * 33.0
        sample = Sample(
            event_data=behavior_csv,
            signal_data=signal_npy,
            fps=30.0,
            frame_timestamps=frame_ts,
        )
        assert sample.num_frames == 100
        df = sample.get_dataframe()
        assert "frame_index" in df.columns

    def test_get_event_timestamps(self, mock_reacher_data_files):
        """Test retrieving event timestamps by REACHER code."""
        behavior_csv, frame_csv, signal_npy = mock_reacher_data_files
        sample = Sample(
            event_data=behavior_csv,
            signal_data=signal_npy,
            fps=30.0,
            frame_timestamps=frame_csv,
        )
        # 101 = rh_lever_active_press
        ts = sample.get_event_timestamps(101)
        assert len(ts) >= 1
        assert all(t > 0 for t in ts)

    def test_count_events_reacher(self, mock_reacher_data_files):
        """Test event counting with REACHER data."""
        behavior_csv, frame_csv, signal_npy = mock_reacher_data_files
        sample = Sample(
            event_data=behavior_csv,
            signal_data=signal_npy,
            fps=30.0,
            frame_timestamps=frame_csv,
        )
        total = sample.count_events()
        assert total > 0
        # Count by label
        active = sample.count_events("rh_lever_active_press")
        assert active >= 1

    def test_dataframe_columns(self, mock_reacher_data_files):
        """Test that the DataFrame has all expected columns."""
        behavior_csv, frame_csv, signal_npy = mock_reacher_data_files
        sample = Sample(
            event_data=behavior_csv,
            signal_data=signal_npy,
            fps=30.0,
            frame_timestamps=frame_csv,
        )
        df = sample.get_dataframe()
        for col in ("code", "t1", "t2", "label", "frame_index"):
            assert col in df.columns

    def test_frame_timestamps_invalid_type_raises(self, mock_reacher_data_files):
        """Test that invalid frame_timestamps type raises TypeError."""
        behavior_csv, _, signal_npy = mock_reacher_data_files
        with pytest.raises(TypeError, match="frame_timestamps must be"):
            Sample(
                event_data=behavior_csv,
                signal_data=signal_npy,
                fps=30.0,
                frame_timestamps=12345,
            )


# Integration tests
class TestSamplePopulationProjectIntegration:
    """Integration tests for Sample, Population, and Project together."""
    
    @pytest.fixture
    def create_full_dataset(self, tmp_path):
        """Create a full test dataset with files."""
        # Create event data
        event_data = np.array([
            [9, 100.0],
            [22, 200.0],
            [9, 300.0],
            [7, 400.0],
            [9, 500.0],
        ])
        mat_path = tmp_path / "test.mat"
        sio.savemat(str(mat_path), {"eventlog": event_data})
        
        # Create signal data
        signal_data = np.random.rand(10, 50).astype(np.float32)
        npy_path = tmp_path / "test.npy"
        np.save(str(npy_path), signal_data)
        
        event_dict = {22: "active_lever", 7: "cue", 9: "frame_trigger"}
        
        return str(mat_path), str(npy_path), event_dict
    
    def test_full_hierarchy(self, create_full_dataset):
        """Test creating full Project > Population > Sample hierarchy."""
        mat_path, npy_path, event_dict = create_full_dataset
        
        # Create samples
        sample1 = Sample(
            event_data=mat_path,
            signal_data=npy_path,
            name="Sample 1",
            event_dict=event_dict,
            fps=30.0,
            frame_averaging=1
        )
        
        sample2 = Sample(
            event_data=mat_path,
            signal_data=npy_path,
            name="Sample 2",
            event_dict=event_dict,
            fps=30.0,
            frame_averaging=1
        )
        
        # Create population
        population = Population(
            samples=[sample1, sample2],
            name="Test Population",
            description="Test description"
        )
        
        # Create project
        project = Project(
            name="Test Project",
            populations=[population],
            authors=["Test Author"],
            description="Test project description"
        )
        
        # Verify hierarchy
        assert project.num_populations == 1
        assert population.num_samples == 2
        assert sample1.num_neurons == 10
        assert sample2.num_neurons == 10
        
        # Verify aggregations
        assert population.num_neurons == 20  # 10 + 10
