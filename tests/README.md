# Pynapse Test Suite

This directory contains comprehensive unit and integration tests for the pynapse library.

## Test Structure

The test suite is organized by module:

### Core Modules
- **`test_core_io.py`**: Tests for IO modules (`EventLog`, `SignalRecording`)
- **`test_core_models.py`**: Tests for core data models (`Sample`, `Population`, `Project`)

### Analysis Modules
- **`test_preprocessing.py`**: Tests for preprocessing pipeline and transformations
- **`test_peri_event.py`**: Tests for peri-event time histogram analysis

### Configuration
- **`test_config.py`**: Tests for event dictionaries and configuration

### Shared Resources
- **`conftest.py`**: Shared pytest fixtures and test configuration
- **`test_core.py`**: Legacy integration test (kept for compatibility)

## Running Tests

### Run All Tests
```bash
pytest tests/
```

### Run Specific Test Module
```bash
pytest tests/test_core_io.py
```

### Run Specific Test Class
```bash
pytest tests/test_core_io.py::TestEventLog
```

### Run Specific Test
```bash
pytest tests/test_core_io.py::TestEventLog::test_count_events_by_code
```

### Run with Verbose Output
```bash
pytest tests/ -v
```

### Run with Coverage Report
```bash
pytest tests/ --cov=pynapse --cov-report=html
```

### Run Only Unit Tests
```bash
pytest tests/ -m unit
```

### Run Only Integration Tests
```bash
pytest tests/ -m integration
```

### Skip Slow Tests
```bash
pytest tests/ -m "not slow"
```

## Test Coverage

The test suite covers:

### Core IO (`test_core_io.py`)
- ✅ EventLog initialization with single/multiple files
- ✅ Event counting by code, label, or list
- ✅ DataFrame and raw data extraction
- ✅ Error handling for invalid files
- ✅ SignalRecording initialization and concatenation
- ✅ Signal data extraction and validation

### Core Models (`test_core_models.py`)
- ✅ Sample initialization and properties
- ✅ FPS calculations and frame conversions
- ✅ Population aggregation (neurons, events)
- ✅ Project hierarchy and metadata
- ✅ Full data hierarchy integration tests

### Preprocessing (`test_preprocessing.py`)
- ✅ Abstract Preprocessor base class
- ✅ BaselineSubtraction with mean/median methods
- ✅ Time-window baseline subtraction
- ✅ Pipeline sequential processing
- ✅ 2D and 3D data handling

### Peri-Event Analysis (`test_peri_event.py`)
- ✅ SampleEventTensor window extraction
- ✅ PopulationEventTensor across samples
- ✅ Event buffering and filtering
- ✅ Preprocessing integration
- ✅ Real data integration tests

### Configuration (`test_config.py`)
- ✅ Event dictionary definitions
- ✅ Color scheme definitions
- ✅ Task-to-dictionary mappings
- ✅ Naming conventions and consistency

## Writing New Tests

### Test Class Structure
```python
class TestMyFeature:
    """Test suite for MyFeature class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        return create_test_data()
    
    def test_feature_behavior(self, sample_data):
        """Test specific feature behavior."""
        result = my_feature(sample_data)
        assert result.is_valid()
```

### Using Shared Fixtures
```python
def test_with_mock_data(mock_data_files):
    """Test using shared fixture from conftest.py."""
    event_file, signal_file, event_dict = mock_data_files
    # Use the mock data...
```

### Parametrized Tests
```python
@pytest.mark.parametrize("fps,expected_frames", [
    (30.0, 30),
    (60.0, 60),
    (15.0, 15),
])
def test_fps_conversion(fps, expected_frames):
    """Test FPS conversion with multiple values."""
    frames = convert_to_frames(1.0, fps)
    assert frames == expected_frames
```

### Test Markers
Use markers to categorize tests:
```python
@pytest.mark.unit
def test_unit_feature():
    """Unit test for isolated functionality."""
    pass

@pytest.mark.integration
def test_integration_feature():
    """Integration test for combined functionality."""
    pass

@pytest.mark.slow
def test_slow_feature():
    """Test that takes significant time."""
    pass
```

## Continuous Integration

These tests are designed to run in CI/CD pipelines:
- All tests should be deterministic
- Use fixtures for random seeds
- Clean up temporary files automatically
- Mock external dependencies when possible

## Best Practices

1. **Test Naming**: Use descriptive names starting with `test_`
2. **Fixtures**: Use fixtures for setup/teardown
3. **Assertions**: Use specific assertions with clear messages
4. **Coverage**: Aim for >80% code coverage
5. **Documentation**: Include docstrings for test classes and methods
6. **Independence**: Tests should not depend on each other
7. **Speed**: Keep unit tests fast (<1s each)

## Troubleshooting

### Import Errors
Ensure pynapse is installed in development mode:
```bash
pip install -e .
```

### Missing Dependencies
Install test dependencies:
```bash
pip install -e ".[dev]"
```

### Fixture Not Found
Check that `conftest.py` is in the tests directory and fixtures are properly defined.

### Tests Not Discovered
Ensure test files start with `test_` and test functions/methods start with `test_`.

## Contributing

When adding new features to pynapse:
1. Write tests first (TDD approach)
2. Ensure all tests pass before submitting PR
3. Maintain or improve code coverage
4. Follow existing test structure and naming conventions
5. Add integration tests for complex features
