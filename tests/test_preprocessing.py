# test_preprocessing.py
# Comprehensive tests for pynapse.analysis.preprocessing modules

import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
import warnings

from pynapse.analysis.preprocessing.base import Preprocessor
from pynapse.analysis.preprocessing.continuous.baseline_subtraction import BaselineSubtraction
from pynapse.analysis.preprocessing.pipeline import Pipeline


class TestPreprocessorBase:
    """Test suite for Preprocessor abstract base class."""
    
    def test_cannot_instantiate_abstract_class(self):
        """Test that Preprocessor cannot be instantiated directly."""
        with pytest.raises(TypeError):
            Preprocessor()
    
    def test_subclass_must_implement_apply(self):
        """Test that subclasses must implement apply method."""
        class IncompletePreprocessor(Preprocessor):
            pass
        
        with pytest.raises(TypeError):
            IncompletePreprocessor()
    
    def test_repr_method(self):
        """Test __repr__ method on a concrete implementation."""
        class TestPreprocessor(Preprocessor):
            def __init__(self, param1=10, param2="test"):
                self.param1 = param1
                self.param2 = param2
            
            def apply(self, data):
                return data
        
        proc = TestPreprocessor(param1=20, param2="custom")
        repr_str = repr(proc)
        assert "TestPreprocessor" in repr_str
        assert "param1=20" in repr_str
        assert "param2='custom'" in repr_str


class TestBaselineSubtraction:
    """Test suite for BaselineSubtraction preprocessor."""
    
    @pytest.fixture
    def sample_2d_data(self):
        """Create sample 2D data (neurons x frames)."""
        np.random.seed(42)
        # Create data with known baseline
        data = np.random.rand(5, 100) + 10.0  # Add baseline of 10
        return data.astype(np.float64)
    
    @pytest.fixture
    def sample_3d_data(self):
        """Create sample 3D data (trials x neurons x frames)."""
        np.random.seed(42)
        # Create data with known baseline
        data = np.random.rand(10, 5, 100) + 5.0  # Add baseline of 5
        return data.astype(np.float64)
    
    def test_init_with_mean_method(self):
        """Test initialization with mean method."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            proc = BaselineSubtraction(method="mean")
            assert proc.method == "mean"
            assert proc.window_ms is None
    
    def test_init_with_median_method(self):
        """Test initialization with median method."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            proc = BaselineSubtraction(method="median")
            assert proc.method == "median"
    
    def test_init_with_window(self):
        """Test initialization with time window."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            proc = BaselineSubtraction(
                method="mean",
                window_ms=(0.0, 1000.0),
                frame_duration_ms=10.0
            )
            assert proc.window_ms == (0.0, 1000.0)
            assert proc.frame_duration_ms == 10.0
    
    def test_invalid_method_raises_error(self):
        """Test that invalid method raises ValueError."""
        with pytest.raises(ValueError, match="method must be 'mean' or 'median'"):
            BaselineSubtraction(method="invalid")
    
    def test_future_warning_issued(self):
        """Test that FutureWarning is issued on initialization."""
        with pytest.warns(FutureWarning, match="BaselineSubtraction is legacy"):
            BaselineSubtraction(method="mean")
    
    def test_mean_baseline_2d(self, sample_2d_data):
        """Test mean baseline subtraction on 2D data."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            proc = BaselineSubtraction(method="mean")
            result = proc.apply(sample_2d_data)
        
        # Result should be centered around 0
        assert result.shape == sample_2d_data.shape
        assert np.abs(result.mean(axis=1)).max() < 1e-6  # Mean should be ~0 for each neuron
    
    def test_median_baseline_2d(self, sample_2d_data):
        """Test median baseline subtraction on 2D data."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            proc = BaselineSubtraction(method="median")
            result = proc.apply(sample_2d_data)
        
        assert result.shape == sample_2d_data.shape
        # Median should be ~0 for each neuron
        assert np.abs(np.median(result, axis=1)).max() < 1e-6
    
    def test_mean_baseline_3d(self, sample_3d_data):
        """Test mean baseline subtraction on 3D data."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            proc = BaselineSubtraction(method="mean")
            result = proc.apply(sample_3d_data)
        
        assert result.shape == sample_3d_data.shape
        # Mean should be ~0 along time axis
        assert np.abs(result.mean(axis=2)).max() < 1e-6
    
    def test_window_baseline_2d(self, sample_2d_data):
        """Test baseline subtraction with time window on 2D data."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            # Use first 10 frames as baseline (10ms per frame)
            proc = BaselineSubtraction(
                method="mean",
                window_ms=(0.0, 100.0),
                frame_duration_ms=10.0
            )
            result = proc.apply(sample_2d_data)
        
        assert result.shape == sample_2d_data.shape
        # Baseline window should be centered around 0
        baseline_window = result[:, 0:10]
        assert np.abs(baseline_window.mean(axis=1)).max() < 1e-6
    
    def test_window_baseline_3d(self, sample_3d_data):
        """Test baseline subtraction with time window on 3D data."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            proc = BaselineSubtraction(
                method="mean",
                window_ms=(0.0, 200.0),
                frame_duration_ms=10.0
            )
            result = proc.apply(sample_3d_data)
        
        assert result.shape == sample_3d_data.shape
        # Baseline window (first 20 frames) should be centered around 0
        baseline_window = result[:, :, 0:20]
        assert np.abs(baseline_window.mean(axis=2)).max() < 1e-6
    
    def test_window_without_frame_duration_raises_error(self, sample_2d_data):
        """Test that using window without frame_duration raises ValueError."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            proc = BaselineSubtraction(method="mean", window_ms=(0.0, 100.0))
        
        with pytest.raises(ValueError, match="frame_duration_ms required"):
            proc.apply(sample_2d_data)
    
    def test_invalid_ndim_raises_error(self):
        """Test that invalid array dimensions raise ValueError."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            proc = BaselineSubtraction(method="mean")
        
        # 1D array should fail
        with pytest.raises(ValueError, match="Expected 2D or 3D array"):
            proc.apply(np.random.rand(100))
        
        # 4D array should fail
        with pytest.raises(ValueError, match="Expected 2D or 3D array"):
            proc.apply(np.random.rand(10, 5, 100, 2))
    
    def test_callable_interface(self, sample_2d_data):
        """Test that preprocessor is callable."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            proc = BaselineSubtraction(method="mean")
            result = proc(sample_2d_data)
        
        assert result.shape == sample_2d_data.shape
    
    def test_output_dtype(self, sample_2d_data):
        """Test that output is float32."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            proc = BaselineSubtraction(method="mean")
            result = proc.apply(sample_2d_data)
        
        assert result.dtype == np.float32


class TestPipeline:
    """Test suite for Pipeline class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data."""
        np.random.seed(42)
        return np.random.rand(5, 100).astype(np.float64)
    
    @pytest.fixture
    def mock_preprocessors(self):
        """Create mock preprocessors."""
        class AddOne(Preprocessor):
            def apply(self, data):
                return (data + 1.0).astype(np.float32)
        
        class MultiplyTwo(Preprocessor):
            def apply(self, data):
                return (data * 2.0).astype(np.float32)
        
        return AddOne(), MultiplyTwo()
    
    def test_pipeline_initialization(self, mock_preprocessors):
        """Test Pipeline initialization."""
        pipeline = Pipeline(steps=list(mock_preprocessors))
        assert len(pipeline.steps) == 2
    
    def test_pipeline_apply_sequential(self, sample_data, mock_preprocessors):
        """Test that pipeline applies steps sequentially."""
        pipeline = Pipeline(steps=list(mock_preprocessors))
        result = pipeline.apply(sample_data)
        
        # Should first add 1, then multiply by 2: (x + 1) * 2
        expected = (sample_data + 1.0) * 2.0
        assert_array_almost_equal(result, expected, decimal=5)
    
    def test_pipeline_with_single_step(self, sample_data, mock_preprocessors):
        """Test pipeline with single preprocessor."""
        pipeline = Pipeline(steps=[mock_preprocessors[0]])
        result = pipeline.apply(sample_data)
        
        expected = sample_data + 1.0
        assert_array_almost_equal(result, expected, decimal=5)
    
    def test_empty_pipeline(self, sample_data):
        """Test empty pipeline returns data unchanged."""
        pipeline = Pipeline(steps=[])
        result = pipeline.apply(sample_data)
        
        # Should return the same data
        assert_array_almost_equal(result, sample_data.astype(np.float32), decimal=5)
    
    def test_pipeline_with_baseline_subtraction(self, sample_data):
        """Test pipeline with actual BaselineSubtraction."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            baseline_sub = BaselineSubtraction(method="mean")
        
        pipeline = Pipeline(steps=[baseline_sub])
        result = pipeline.apply(sample_data)
        
        # Result should be centered around 0
        assert np.abs(result.mean(axis=1)).max() < 1e-5
    
    def test_pipeline_repr(self, mock_preprocessors):
        """Test pipeline string representation."""
        pipeline = Pipeline(steps=list(mock_preprocessors))
        repr_str = repr(pipeline)
        assert "Pipeline" in repr_str


# Integration tests
class TestPreprocessingIntegration:
    """Integration tests for preprocessing functionality."""
    
    @pytest.fixture
    def realistic_neural_data(self):
        """Create realistic-looking neural data."""
        np.random.seed(42)
        # Simulate neurons x frames with baseline + activity
        baseline = 100.0
        n_neurons = 10
        n_frames = 500
        
        # Add baseline, noise, and some activity
        data = np.random.randn(n_neurons, n_frames) * 5 + baseline
        
        # Add some event-related activity
        for i in range(n_neurons):
            # Add a "response" in middle section
            data[i, 200:250] += np.random.rand() * 20
        
        return data.astype(np.float64)
    
    def test_full_preprocessing_workflow(self, realistic_neural_data):
        """Test a complete preprocessing workflow."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            # Create preprocessing steps
            baseline_sub = BaselineSubtraction(
                method="mean",
                window_ms=(0.0, 1000.0),
                frame_duration_ms=10.0
            )
        
        # Create pipeline
        pipeline = Pipeline(steps=[baseline_sub])
        
        # Apply preprocessing
        processed = pipeline.apply(realistic_neural_data)
        
        # Verify baseline window is centered
        baseline_window = processed[:, 0:100]
        assert np.abs(baseline_window.mean(axis=1)).max() < 1e-4
        
        # Verify shape is preserved
        assert processed.shape == realistic_neural_data.shape
        
        # Verify activity is preserved (should still see elevated activity in middle section)
        activity_window = processed[:, 200:250]
        assert activity_window.mean() > 0  # Should be elevated relative to baseline
