# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

**brew** is a Python library for analyzing behavioral and neural fluorescence data from 2-photon imaging experiments. It is designed for neuroscience research involving head-fixed self-administration paradigms (heroin, ethanol).

## Commands

### Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run tests
Tests use the `if __name__ == "__main__"` pattern rather than pytest:
```bash
python -m brew.core.test_core
python -m brew.analysis.test_analysis
```

### Run individual modules
```bash
python -m brew.core.sample
python -m brew.core.population
python -m brew.core.project
```

## Architecture

### Data Flow
```
EventLog + SignalRecording → Sample → Population → Project
                                ↓
                         EventMatrix (peri-event extraction)
                                ↓
                      ProcessingPipeline (preprocessing)
```

### Core Modules (`brew/core/`)

- **Sample**: Aligns event logs with neural signals. Handles frame timestamp alignment, frame averaging, and missed frame correction. Key parameters: `fps`, `frame_averaging`, `event_dict`.
- **Population**: Aggregates multiple Samples with computed properties (total neurons, events).
- **Project**: Top-level container grouping Populations with metadata.
- **io/behavior.EventLog**: Loads event logs from MATLAB `.mat` files (legacy, deprecated) or lists of files. Creates pandas DataFrames with columns: `code`, `t1`, `t2`, `label`.
- **io/microscopy.SignalRecording**: Loads neural signals from `.npy` files. Shape: `(neurons, frames)`.

### Analysis Modules (`brew/analysis/`)

- **peri_event.SampleEventTensor**: Extracts time windows around events for a single Sample. Returns tensor of shape `(trials, neurons, frames)`.
- **peri_event.PopulationEventTensor**: Extracts windows across all Samples in a Population.
- **preprocessing/**: Composable preprocessors that operate on 2D `(neurons, frames)` or 3D `(trials, neurons, frames)` arrays:
  - `DFOverF`: ΔF/F calculation using percentile baseline
  - `ZScore`: Z-scoring with configurable baseline window
  - `GaussianSmoothing`: Temporal smoothing
  - `BaselineSubtraction`: Legacy, use DFOverF + ZScore instead
  - `ProcessingPipeline`: Chains preprocessors sequentially

### Configuration (`brew/config/`)

- **events.py**: Single source of truth for event code → label mappings. Use `LEGACY_HER` for heroin paradigm, `LEGACY_ETH` for ethanol. Never define event dictionaries elsewhere.
- **pipelines.py**: Standard preprocessing pipelines. Use `OTIS_PIPE` (DFOverF → ZScore → GaussianSmoothing) or `LEGACY_PIPE` for consistency.

### Data Formats

- **Event logs**: MATLAB `.mat` files containing `eventlog` array with columns `[code, timestamp_ms]`. Event code 9 = frame trigger.
- **Neural signals**: NumPy `.npy` files with shape `(neurons, frames)`.
- **Event codes**: Integers mapped to labels via dictionaries in `config/events.py`.

## Key Patterns

1. **Always import event dictionaries from `brew.config.events`** - never define them inline.
2. **Use standard pipelines from `brew.config.pipelines`** for reproducibility.
3. **Preprocessors are callable** - they implement both `__call__` and `apply` methods.
4. **Frame averaging**: Raw microscopy runs at 30fps, typically averaged 4x to 7.5fps effective rate.
5. **Time windows in peri-event analysis**: Specified in seconds (`pre_event`, `post_event`), converted internally to frames.
