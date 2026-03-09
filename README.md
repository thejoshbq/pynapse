# Pynapse — Neural Data Engine

**Data loading, alignment, preprocessing, and tensor extraction for calcium imaging experiments**

[![Version](https://img.shields.io/badge/version-0.1.0-blue)](https://github.com/thejoshbq/pynapse)
[![Python](https://img.shields.io/badge/python-3.8%E2%80%933.12-blue)](https://www.python.org)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![REACHER Suite](https://img.shields.io/badge/REACHER_Suite-member-orange)](https://github.com/Otis-Lab-MUSC)

*Written by*: Joshua Boquiren

[![](https://img.shields.io/badge/@thejoshbq-grey?style=flat&logo=github)](https://github.com/thejoshbq)

---

## Overview

Pynapse is the data engine for the REACHER analysis ecosystem. It handles loading raw neural fluorescence recordings and behavioral event logs, aligning timescales between imaging and behavior, applying configurable preprocessing pipelines, and extracting peri-event tensors for downstream analysis.

Higher-level tools like [Axplorer](https://github.com/thejoshbq/neural-eda) wrap Pynapse's core objects — they never reimplement the validated low-level logic. Pynapse provides:

- **Data I/O** — `EventLog` for MATLAB/CSV behavioral events, `SignalRecording` for `.npy` fluorescence traces
- **Alignment** — Frame-timestamp alignment with frame averaging and optional correction
- **Preprocessing** — Epoch-based (DF/F, Z-score, Gaussian smoothing) and continuous (baseline subtraction, normalization) pipelines
- **Peri-event tensors** — `SampleEventTensor` and `PopulationEventTensor` for windowed signal extraction locked to behavioral events
- **Event configuration** — Canonical code→label dictionaries for all lab paradigms

---

## Role in the Ecosystem

Pynapse sits between raw data and analysis tools:

```
Raw Data (.npy, .mat, .xlsx)
        │
        ▼
   ┌─────────┐
   │ Pynapse │  ← loading, alignment, preprocessing, tensor extraction
   └────┬────┘
        │
        ▼
   ┌──────────┐
   │ Axplorer │  ← PETH computation, response metrics, visualization, export
   └──────────┘
```

---

## Architecture

### Project Structure

```
pynapse/
├── pyproject.toml                      # Package metadata and dependencies
├── pynapse/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py                 # Exports: Sample, Population, Project
│   │   ├── sample.py                   # Sample — single experiment session
│   │   ├── population.py               # Population — collection of Samples
│   │   ├── project.py                  # Project — collection of Populations
│   │   ├── mixins.py                   # TensorConfigMixin — shared tensor defaults
│   │   └── io/
│   │       ├── __init__.py
│   │       ├── behavior.py             # EventLog — MATLAB/CSV event loading
│   │       └── microscopy.py           # SignalRecording — .npy signal loading
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── peri_event.py               # EventTensor, SampleEventTensor, PopulationEventTensor
│   │   ├── cluster.py                  # Clustering utilities
│   │   ├── forest_clustering.py        # Random forest-based clustering
│   │   └── preprocessing/
│   │       ├── __init__.py             # Exports: Pipeline, Preprocessor, DFOverF, ZScore, etc.
│   │       ├── base.py                 # Preprocessor abstract base class
│   │       ├── pipeline.py             # Pipeline — ordered sequence of preprocessors
│   │       ├── epoch/                  # Per-window preprocessors
│   │       │   ├── relative_change.py  # DFOverF
│   │       │   ├── standardization.py  # ZScore
│   │       │   ├── gaussian_smoothing.py # GaussianSmoothing
│   │       │   └── pipelines.py        # OTIS_PIPE — lab standard pipeline
│   │       └── continuous/             # Full-trace preprocessors
│   │           ├── baseline_subtraction.py # BaselineSubtraction (legacy)
│   │           ├── normalization.py    # Normalize (legacy)
│   │           ├── pipeline.py         # Continuous pipeline
│   │           └── pipelines.py        # Pre-built continuous pipelines
│   ├── config/
│   │   ├── __init__.py
│   │   └── events.py                   # LEGACY_HER, LEGACY_ETH, REACHER, TASK_TO_DICT, COLORS
│   ├── viz/
│   │   ├── __init__.py
│   │   ├── colors.py                   # Color utilities
│   │   ├── ensemble.py                 # Ensemble plots
│   │   ├── heatmaps.py                # Heatmap visualizations
│   │   ├── peth.py                     # PETH plotting
│   │   ├── style.py                    # Plot styling
│   │   └── trace.py                    # Trace plots
│   ├── db/
│   │   └── __init__.py                 # Database integration (placeholder)
│   └── llm/
│       └── __init__.py                 # LLM integration (placeholder)
└── tests/
    ├── conftest.py                     # Shared fixtures
    ├── test_core.py                    # Sample integration tests
    ├── test_core_models.py             # Population, Project tests
    ├── test_core_io.py                 # EventLog, SignalRecording tests
    ├── test_config.py                  # Event dictionary tests
    ├── test_peri_event.py              # SampleEventTensor, PopulationEventTensor tests
    └── test_preprocessing.py           # DFOverF, ZScore, GaussianSmoothing tests
```

---

## Data Models

### Sample

The central class. Encapsulates a single experimental session with aligned neural signals and behavioral events.

**Constructor parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `event_data` | path(s) or `EventLog` | *required* | MATLAB/CSV event log file(s) |
| `signal_data` | path(s) or `SignalRecording` | *required* | `.npy` fluorescence file(s) |
| `name` | `str` | `None` | Session name |
| `event_dict` | `dict[int, str]` | `None` | Event code → label mapping |
| `fps` | `float` | `30.0` | Raw imaging frame rate (Hz) |
| `frame_averaging` | `int` | `1` | Number of raw frames binned per signal frame |
| `frame_correction` | `bool` | `False` | Enable frame timestamp correction |
| `correction_file` | path | `None` | Path to correction file |
| `start_time` | `float` | `0.0` | Session start time offset |
| `default_event_id` | `int` or `list[int]` | `None` | Default event(s) for tensor extraction |
| `default_pre_event` | `float` | `None` | Default pre-event window (seconds) |
| `default_post_event` | `float` | `None` | Default post-event window (seconds) |
| `default_buffer_ms` | `int` | `0` | Minimum inter-event interval (ms) |
| `default_min_trials` | `int` | `1` | Minimum valid trials required |

**Key properties:** `name`, `effective_fps`, `num_neurons`, `num_frames`, `interframe_interval`

### Population

Aggregates a collection of `Sample` objects with computed properties.

```python
pop = Population(
    samples=[sample_1, sample_2, sample_3],
    name="Treatment Group A",
)
print(pop.num_samples, pop.num_neurons)
```

### Project

Top-level organizer grouping `Population` objects with metadata.

```python
proj = Project(
    name="Heroin SA Study",
    populations=[saline_group, treatment_group],
    authors=["Boquiren, J."],
)
```

### EventLog

Loads and parses behavioral event logs from MATLAB `.mat` files (or lists of files). Maps integer event codes to human-readable labels via an optional event dictionary.

### SignalRecording

Loads `.npy` fluorescence trace files (single or concatenated). Provides `num_neurons` and `num_frames` properties.

---

## Preprocessing

Pynapse provides two categories of preprocessors:

### Epoch Preprocessors (per peri-event window)

| Class | Description | Key Parameters |
|---|---|---|
| `DFOverF` | ΔF/F normalization — baseline percentile-based | `percentile` (default: 8.0) |
| `ZScore` | Z-score normalization — baseline window mean/SD | `window_ms` (default: (-3000, -500)), `frame_duration_ms` |
| `GaussianSmoothing` | Temporal Gaussian smoothing | `sigma_frames` (default: 2.0) |

### Continuous Preprocessors (full trace)

| Class | Description | Status |
|---|---|---|
| `BaselineSubtraction` | Mean/median baseline subtraction | Legacy — prefer DFOverF + ZScore |
| `Normalize` | Mean normalization with optional full-trace Z-score | Legacy |

### OTIS_PIPE — Lab Standard Pipeline

The canonical preprocessing pipeline used across all lab publications:

```python
from pynapse.analysis.preprocessing.epoch.pipelines import OTIS_PIPE

# Equivalent to:
# Pipeline([DFOverF(percentile=8), ZScore(window_ms=(-3000, -500)), GaussianSmoothing()])
```

---

## Peri-Event Tensor Extraction

### SampleEventTensor

Extracts windowed signal data locked to behavioral events from a single `Sample`:

```python
from pynapse.analysis.peri_event import SampleEventTensor
from pynapse.analysis.preprocessing.epoch.pipelines import OTIS_PIPE

tensor = SampleEventTensor(
    sample=my_sample,
    event_id=7,            # event code for behavioral events
    pre_event=5.0,         # 5 seconds before event
    post_event=10.0,       # 10 seconds after event
    buffer_ms=2000,        # exclude events closer than 2s apart
    min_trials=3,          # require at least 3 valid trials
    window_preprocess=OTIS_PIPE,
)
```

### PopulationEventTensor

Extracts and aggregates tensors across all samples in a `Population`, sharing the same event and window parameters.

---

## Event Configuration

Canonical event code → label dictionaries are defined in `pynapse/config/events.py`. All analysis code, plotting, and database schemas import from this single source of truth.

### LEGACY_HER (Heroin Self-Administration)

| Code | Label |
|---|---|
| 22 | `active_lever` |
| 222 | `active_lever_timeout` |
| 21 | `inactive_lever` |
| 212 | `inactive_lever_timeout` |
| 7 | `cue` |
| 4 | `infusion` |
| 9 | `frame_trigger` |

### LEGACY_ETH (Ethanol Self-Administration)

| Code | Label |
|---|---|
| 22 | `active_lick` |
| 222 | `active_lick_timeout` |
| 21 | `inactive_lick` |
| 7 | `cue_onset` |
| 50 | `ethanol_delivery` |
| 51 | `water_delivery` |
| 9 | `frame_trigger` |

### REACHER

Placeholder — will be populated when the REACHER task event codes are finalized.

### Convenience Lookup

```python
from pynapse.config.events import TASK_TO_DICT

event_dict = TASK_TO_DICT["legacy_her"]  # → LEGACY_HER dictionary
```

### Standard Colors

`COLORS` provides consistent plot colors across all figures (e.g., `active_lever_press` → red, `infusion` → green).

---

## Installation

Pynapse is not published on PyPI. Install from a local clone:

```bash
git clone https://github.com/thejoshbq/pynapse.git
cd pynapse
pip install -e .
```

For development:

```bash
pip install -e ".[dev]"
```

---

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov

# Run a specific test module
pytest tests/test_preprocessing.py
```

### Test Module Mapping

| Test File | Covers |
|---|---|
| `test_core.py` | `pynapse/core/sample.py` — Sample integration |
| `test_core_models.py` | `pynapse/core/population.py`, `pynapse/core/project.py` |
| `test_core_io.py` | `pynapse/core/io/behavior.py`, `pynapse/core/io/microscopy.py` |
| `test_config.py` | `pynapse/config/events.py` |
| `test_peri_event.py` | `pynapse/analysis/peri_event.py` |
| `test_preprocessing.py` | `pynapse/analysis/preprocessing/epoch/`, `pynapse/analysis/preprocessing/continuous/` |

---

## Dependencies

### Runtime

| Package | Purpose |
|---|---|
| numpy | Array operations, signal processing |
| pandas | Event data manipulation, DataFrames |
| scipy | Gaussian smoothing, MATLAB file loading |
| scikit-learn | Clustering and classification |
| matplotlib | Static figure generation |
| plotly | Interactive figures |
| seaborn | Statistical visualizations |
| narwhals | DataFrame compatibility layer |
| ipython | Interactive shell integration |

### Development

| Package | Purpose |
|---|---|
| pytest | Test runner |
| pytest-cov | Coverage reporting |

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Contact

Joshua Boquiren — [thejoshbq@proton.me](mailto:thejoshbq@proton.me)

[GitHub: thejoshbq/pynapse](https://github.com/thejoshbq/pynapse)
