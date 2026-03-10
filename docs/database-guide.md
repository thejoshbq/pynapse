# Pynapse Database Guide

## Table of Contents

- [1. Overview](#1-overview)
- [2. Quick Start](#2-quick-start)
- [3. User Guide](#3-user-guide)
  - [3.1 Ingesting Data](#31-ingesting-data)
  - [3.2 Querying Data](#32-querying-data)
  - [3.3 Common Workflows](#33-common-workflows)
- [4. Architecture Reference](#4-architecture-reference)
  - [4.1 Schema & ERD](#41-schema--erd)
  - [4.2 Design Rationale](#42-design-rationale)
  - [4.3 Pipeline Internals](#43-pipeline-internals)
- [5. API Reference](#5-api-reference)
  - [5.1 pynapse.db.engine](#51-pynapsedbengine)
  - [5.2 pynapse.db.schema](#52-pynapsedbschema)
  - [5.3 pynapse.db.ingest](#53-pynapsedbingest)
  - [5.4 pynapse.db.query](#54-pynapsedbquery)
  - [5.5 pynapse.db.hydrate](#55-pynapsedbhydrate)
- [6. Troubleshooting](#6-troubleshooting)
- [7. Glossary](#7-glossary)

---

## 1. Overview

### What is pynapse?

Pynapse is a neural data analysis library for two-photon calcium imaging experiments paired with behavioral event logs. It provides a unified pipeline from raw data files through peri-event tensor extraction and visualization.

### What problem does the database solve?

Without the database layer, every analysis session must re-read raw `.mat` or `.csv` event files and `.npy` signal arrays, re-align frame timestamps, and re-compute frame indices from scratch. This is slow, error-prone, and makes it difficult to query across subjects or sessions.

The database layer (`pynapse.db`) solves this by:

1. **Ingesting once** -- raw files are parsed, aligned, and stored in a single DuckDB file.
2. **Querying instantly** -- aligned traces, events, and metadata are available through simple function calls.
3. **Preserving raw files** -- original source files are copied into a managed directory alongside their SHA-256 checksums.
4. **Standardizing hierarchy** -- every FOV is placed within a consistent Project > Population > Subject > FOV structure.

### Data hierarchy

```
Project
  └── Population          (e.g., "EtOH", "Saline")
        └── Subject       (e.g., "Mouse1")
              └── FOV     (field of view -- one imaging session)
                    ├── Neural traces     (neurons x frames matrix)
                    ├── Frame timestamps  (1-D array, ms)
                    ├── Events            (code, label, t1, t2, frame_index)
                    └── Neurons           (per-cell metadata)
```

### Data flow

```
Raw files (.mat, .csv, .npy)
        │
        ▼
   Sample object          ← alignment, frame index computation
        │
        ▼
   ingest.fov() / ingest.from_sample()
        │
        ▼
   DuckDB file (~/.pynapse/pynapse.duckdb)
        │
        ▼
   query.get_traces() / query.get_events() / DBSample
        │
        ▼
   SampleEventTensor      ← peri-event windows for analysis
```

---

## 2. Quick Start

### Prerequisites

- Python 3.10+
- pynapse installed (`pip install -e .` from the repo root)
- DuckDB is installed automatically as a dependency

### Minimal example: ingest and query

```python
from pynapse import db

# Connect (creates ~/.pynapse/pynapse.duckdb on first use)
conn = db.connect()

# Ingest a single FOV from raw files
fov_id = db.ingest.fov(
    neural="path/to/signals.npy",
    events="path/to/events.csv",
    sample_name="Mouse1",
    fov_name="Mouse1_D1_FOV1",
    population_name="EtOH",
    project_name="MyProject",
    paradigm_name="reacher",
    fps=30.0,
    frame_averaging=4,
    frame_timestamps="path/to/frame_timestamps.csv",
    conn=conn,
)

# Query the ingested data
traces = db.query.get_traces(fov_id, conn=conn)       # (neurons, frames) ndarray
events = db.query.get_events(fov_id, conn=conn)        # DataFrame
summary = db.query.get_event_summary(fov_id, conn=conn) # event counts by label

print(f"Traces shape: {traces.shape}")
print(summary)

# Use DBSample for peri-event analysis
sample = db.DBSample(fov_id, conn=conn)
tensor = sample.get_tensor(event_id=101, pre_event=2.0, post_event=5.0)
windows = tensor.get_event_windows()  # (trials, neurons, frames) ndarray

conn.close()
```

---

## 3. User Guide

### 3.1 Ingesting Data

There are four ingestion entry points, from most manual to most automated:

| Function | Input | Use case |
|----------|-------|----------|
| `ingest.fov()` | Raw file paths + metadata | Ingest a single FOV from scratch |
| `ingest.from_sample()` | A `Sample` object + subject ID | Ingest a pre-constructed Sample |
| `ingest.from_population()` | A `Population` object + project ID | Ingest all samples in a population |
| `ingest.from_project()` | A `Project` object | Ingest an entire project |

#### `ingest.fov()` -- the primary entry point

This is the recommended way to ingest data. It accepts raw file paths, creates a `Sample` internally, upserts the full hierarchy (project, population, subject), and stores everything.

```python
fov_id = db.ingest.fov(
    neural="path/to/signals.npy",           # or list of .npy files
    events="path/to/events.csv",            # or list of .mat files
    sample_name="Mouse1",                   # becomes the subject name
    fov_name="Mouse1_D1_FOV1",             # unique name for this FOV
    population_name="EtOH",
    project_name="MyProject",
    paradigm_name="reacher",                # must be a seeded paradigm
    fps=30.0,
    frame_averaging=4,                      # default: 1
    frame_timestamps="path/to/frame_ts.csv", # REACHER CSV; omit for legacy
    frame_correction=False,                 # legacy .mat correction
    correction_file=None,                   # path to correction .mat
    start_time=0.0,                         # session start offset (ms)
    conn=conn,                              # optional; uses cached connection if omitted
    copy_raw=True,                          # copy source files into managed dir
    raw_dir=None,                           # custom raw file directory
    project_description=None,
    population_description=None,
)
```

**Parameters:**

- **`neural`** -- Path (or list of paths) to `.npy` signal files. Each file is a 2-D array of shape `(neurons, frames)`.
- **`events`** -- Path (or list of paths) to event files. For REACHER, this is a `.csv` with columns: `device, event, start_timestamp, end_timestamp, start_frame_index, end_frame_index`. For legacy paradigms, this is a `.mat` file containing an `eventlog` variable.
- **`paradigm_name`** -- Must match a seeded paradigm: `"reacher"`, `"legacy_her"`, or `"legacy_eth"`.
- **`frame_timestamps`** -- A CSV file with a `timestamp_ms` column (and optionally `frame_index`). Required for REACHER data. Omit for legacy `.mat` data where frame triggers are extracted from event code 9.
- **`copy_raw`** -- When `True` (default), source files are copied into `~/.pynapse/raw/<fov_id>/` with SHA-256 checksums recorded.

#### `ingest.from_sample()`

Use this when you already have a `Sample` object and have manually created the subject in the hierarchy:

```python
from pynapse.core.sample import Sample
from pynapse.config.events import REACHER

sample = Sample(
    event_data="events.csv",
    signal_data="signals.npy",
    name="Mouse1_D1_FOV1",
    event_dict=REACHER,
    fps=30.0,
    frame_averaging=4,
    frame_timestamps="frame_ts.csv",
)

fov_id = db.ingest.from_sample(
    sample,
    subject_id=1,           # must already exist in the DB
    paradigm_name="reacher",
    fov_name="Mouse1_D1_FOV1",
    conn=conn,
)
```

#### `ingest.from_population()` and `ingest.from_project()`

These bulk-ingest functions iterate over all samples in a `Population` or all populations in a `Project`:

```python
# Ingest an entire population
fov_ids = db.ingest.from_population(
    population,
    project_id=1,
    paradigm_name="reacher",
    conn=conn,
)

# Ingest an entire project
fov_ids = db.ingest.from_project(
    project,
    paradigm_name="reacher",
    conn=conn,
)
```

Each sample's `.name` is used as both the subject name and FOV name. Hierarchy nodes are created automatically via get-or-create logic.

#### Supported file formats

| Format | Extension | Used by | Contents |
|--------|-----------|---------|----------|
| REACHER event CSV | `.csv` | `reacher` paradigm | `device, event, start_timestamp, end_timestamp, start_frame_index, end_frame_index` |
| Legacy event MAT | `.mat` | `legacy_her`, `legacy_eth` | `eventlog` variable: Nx2 array of `[code, timestamp_ms]` |
| Frame timestamps CSV | `.csv` | REACHER | `frame_index, timestamp_ms` |
| Signal array | `.npy` | All paradigms | 2-D float array `(neurons, frames)` |

#### What happens during ingestion

1. A `Sample` object is created (or provided), which parses raw files and aligns frame timestamps.
2. The project/population/subject hierarchy is upserted (get-or-create).
3. A `fovs` row is inserted with imaging parameters and metadata.
4. Neural traces are serialized as a float32 BLOB into `neural_traces`.
5. Frame timestamps are serialized as a float64 BLOB into `frame_timestamps`.
6. Each event is inserted into `events` with its `code`, `label`, `t1_ms`, `t2_ms`, `frame_index`, and `end_frame_index`.
7. Placeholder neuron rows are created in `neurons` (one per neuron index).
8. If `copy_raw=True`, source files are copied to `~/.pynapse/raw/<fov_id>/` and registered in `raw_files` with SHA-256 checksums.

---

### 3.2 Querying Data

#### Listing functions

All listing functions return a pandas DataFrame:

```python
# List all projects
db.query.list_projects(conn=conn)

# List populations (optionally filtered)
db.query.list_populations(project_name="MyProject", conn=conn)
db.query.list_populations(project_id=1, conn=conn)

# List subjects
db.query.list_subjects(population_name="EtOH", conn=conn)
db.query.list_subjects(population_id=1, conn=conn)

# List FOVs
db.query.list_fovs(subject_id=1, conn=conn)
db.query.list_fovs(conn=conn)  # all FOVs
```

#### Getting FOV metadata

```python
fov = db.query.get_fov(fov_id=1, conn=conn)     # by ID
fov = db.query.get_fov(name="Mouse1_D1_FOV1", conn=conn)  # by name

# Returns a dict with all fovs columns:
# id, subject_id, paradigm_id, name, fps, frame_averaging,
# effective_fps, interframe_interval_ms, frame_correction,
# start_time_ms, num_neurons, num_frames, source_format,
# notes, recorded_at, ingested_at
```

#### Retrieving neural traces

```python
traces = db.query.get_traces(fov_id=1, conn=conn)
# Returns: numpy ndarray of shape (num_neurons, num_frames), dtype float32
```

#### Retrieving events

```python
# All events for a FOV
events_df = db.query.get_events(fov_id=1, conn=conn)

# Filtered by label or code
lever_events = db.query.get_events(fov_id=1, label="rh_lever_active_press", conn=conn)
code_events = db.query.get_events(fov_id=1, code=101, conn=conn)

# Event summary (counts by label and code)
summary = db.query.get_event_summary(fov_id=1, conn=conn)

# Just the frame indices for specific events
indices = db.query.get_event_frame_indices(fov_id=1, label="pump_infusion", conn=conn)
```

The `get_events()` DataFrame columns match `Sample.get_dataframe()`: `code`, `t1`, `t2`, `label`, `frame_index`.

#### Retrieving frame timestamps

```python
ts = db.query.get_frame_timestamps(fov_id=1, conn=conn)
# Returns: 1-D numpy array of timestamps in ms, dtype float64
```

#### Peri-event analysis with `DBSample`

`DBSample` is a drop-in replacement for `Sample` that loads data from the database instead of raw files. It implements the same interface consumed by `SampleEventTensor`:

```python
# Create a DBSample from a FOV ID
sample = db.DBSample(fov_id=1, conn=conn)

# Inspect metadata (same properties as Sample)
print(sample.name)              # FOV name
print(sample.fps)               # raw FPS
print(sample.effective_fps)     # fps / frame_averaging
print(sample.interframe_interval)  # 1000.0 / fps (ms)
print(sample.num_neurons)
print(sample.num_frames)

# Access data (lazy-loaded, cached after first call)
signals = sample.get_signals()          # (neurons, frames) ndarray
events_df = sample.get_dataframe()      # events DataFrame
frame_ts = sample._get_frame_timestamps()  # 1-D timestamps array
event_dict = sample.get_event_dict()    # {code: label} from paradigm

# Create a peri-event tensor (via TensorConfigMixin)
tensor = sample.get_tensor(
    event_id=101,          # event code (or list of codes)
    pre_event=2.0,         # seconds before event
    post_event=5.0,        # seconds after event
    buffer_ms=500,         # minimum interval between events (ms)
    min_trials=3,          # skip if fewer valid trials
)
windows = tensor.get_event_windows()  # (trials, neurons, frames) ndarray
```

You can also set default tensor parameters at construction:

```python
sample = db.DBSample(
    fov_id=1,
    conn=conn,
    default_event_id=101,
    default_pre_event=2.0,
    default_post_event=5.0,
    default_buffer_ms=500,
    default_min_trials=3,
)

# Now get_tensor() uses defaults; no arguments needed
tensor = sample.get_tensor()
```

---

### 3.3 Common Workflows

#### Workflow 1: Ingest a REACHER experiment and compare peri-event responses

```python
from pynapse import db

conn = db.connect()

# Ingest two populations
for mouse_dir in Path("data/EtOH").iterdir():
    db.ingest.fov(
        neural=mouse_dir / "signals.npy",
        events=mouse_dir / "events.csv",
        sample_name=mouse_dir.name,
        fov_name=f"{mouse_dir.name}_FOV1",
        population_name="EtOH",
        project_name="EtOH_vs_Saline",
        paradigm_name="reacher",
        fps=30.0,
        frame_averaging=4,
        frame_timestamps=mouse_dir / "frame_timestamps.csv",
        conn=conn,
    )

for mouse_dir in Path("data/Saline").iterdir():
    db.ingest.fov(
        neural=mouse_dir / "signals.npy",
        events=mouse_dir / "events.csv",
        sample_name=mouse_dir.name,
        fov_name=f"{mouse_dir.name}_FOV1",
        population_name="Saline",
        project_name="EtOH_vs_Saline",
        paradigm_name="reacher",
        fps=30.0,
        frame_averaging=4,
        frame_timestamps=mouse_dir / "frame_timestamps.csv",
        conn=conn,
    )

# Query and compare
fovs = db.query.list_fovs(conn=conn)
for _, row in fovs.iterrows():
    sample = db.DBSample(row["id"], conn=conn)
    tensor = sample.get_tensor(event_id=101, pre_event=2.0, post_event=5.0)
    windows = tensor.get_event_windows()
    print(f"{sample.name}: {windows.shape[0]} trials, {sample.num_neurons} neurons")

conn.close()
```

#### Workflow 2: Add a new mouse to an existing project

```python
conn = db.connect()

# The project, population, and paradigm already exist.
# ingest.fov() will find them by name (get-or-create).
fov_id = db.ingest.fov(
    neural="data/NewMouse/signals.npy",
    events="data/NewMouse/events.csv",
    sample_name="NewMouse",
    fov_name="NewMouse_FOV1",
    population_name="EtOH",           # existing population
    project_name="EtOH_vs_Saline",    # existing project
    paradigm_name="reacher",
    fps=30.0,
    frame_averaging=4,
    frame_timestamps="data/NewMouse/frame_timestamps.csv",
    conn=conn,
)

print(f"Ingested as fov_id={fov_id}")
conn.close()
```

#### Workflow 3: Re-query with a different event window

No re-ingestion needed. Just create a new `DBSample` and call `get_tensor()` with different parameters:

```python
conn = db.connect()
sample = db.DBSample(fov_id=1, conn=conn)

# Try a wider window
tensor_wide = sample.get_tensor(event_id=101, pre_event=5.0, post_event=10.0)

# Try a different event
tensor_infusion = sample.get_tensor(event_id=201, pre_event=2.0, post_event=5.0)

# Try multiple event codes combined
tensor_multi = sample.get_tensor(event_id=[101, 111], pre_event=2.0, post_event=5.0)

# Tensors are cached -- calling with the same params returns the cached result
tensor_cached = sample.get_tensor(event_id=101, pre_event=5.0, post_event=10.0)
assert tensor_cached is tensor_wide  # same object

conn.close()
```

---

## 4. Architecture Reference

### 4.1 Schema & ERD

#### Entity-relationship diagram

```mermaid
erDiagram
    projects ||--o{ populations : "has"
    populations ||--o{ subjects : "has"
    subjects ||--o{ fovs : "has"
    paradigms ||--o{ fovs : "defines"
    fovs ||--|| neural_traces : "stores"
    fovs ||--|| frame_timestamps : "stores"
    fovs ||--o{ events : "records"
    fovs ||--o{ neurons : "contains"
    fovs ||--o{ raw_files : "archives"

    projects {
        INTEGER id PK
        TEXT name UK
        TEXT description
        TEXT authors
        TIMESTAMP created_at
        TIMESTAMP updated_at
    }

    populations {
        INTEGER id PK
        INTEGER project_id FK
        TEXT name
        TEXT description
        TIMESTAMP created_at
    }

    subjects {
        INTEGER id PK
        INTEGER population_id FK
        TEXT name
        TEXT sex
        TEXT genotype
        TEXT species
        DATE date_of_birth
        TEXT notes
        TIMESTAMP created_at
    }

    paradigms {
        INTEGER id PK
        TEXT name UK
        TEXT event_dict
        TEXT color_dict
        TEXT description
    }

    fovs {
        INTEGER id PK
        INTEGER subject_id FK
        INTEGER paradigm_id FK
        TEXT name
        DOUBLE fps
        INTEGER frame_averaging
        DOUBLE effective_fps "GENERATED"
        DOUBLE interframe_interval_ms "GENERATED"
        BOOLEAN frame_correction
        DOUBLE start_time_ms
        INTEGER num_neurons
        INTEGER num_frames
        TEXT source_format
        TEXT notes
        TIMESTAMP recorded_at
        TIMESTAMP ingested_at
    }

    neural_traces {
        INTEGER fov_id PK_FK
        INTEGER num_neurons
        INTEGER num_frames
        BLOB trace_data
        TEXT dtype
    }

    frame_timestamps {
        INTEGER fov_id PK_FK
        INTEGER num_timestamps
        BLOB timestamps_ms
        TEXT source
    }

    events {
        INTEGER id PK
        INTEGER fov_id FK
        INTEGER code
        TEXT label
        DOUBLE t1_ms
        DOUBLE t2_ms
        INTEGER frame_index
        INTEGER end_frame_index
    }

    neurons {
        INTEGER id PK
        INTEGER fov_id FK
        INTEGER neuron_index
        TEXT label
        TEXT cell_type
        TEXT region
        DOUBLE x_um
        DOUBLE y_um
    }

    raw_files {
        INTEGER id PK
        INTEGER fov_id FK
        TEXT file_type
        TEXT original_filename
        TEXT stored_path
        TEXT sha256
        BIGINT file_size_bytes
        INTEGER part_number
        TIMESTAMP ingested_at
    }

    _meta {
        TEXT key PK
        TEXT value
    }
```

#### Table reference

**`_meta`** -- Schema versioning. Contains a single row: `key='schema_version'`, `value='1'`.

**`projects`** -- Top-level grouping. `authors` is stored as a JSON string. Names are unique.

**`populations`** -- Experimental groups within a project (e.g., "EtOH", "Saline"). Unique on `(project_id, name)`.

**`subjects`** -- Individual animals. Unique on `(population_id, name)`. Species defaults to `'mouse'`.

**`paradigms`** -- Task definitions. `event_dict` and `color_dict` are JSON strings mapping event codes to labels and labels to hex colors. Three paradigms are seeded on initialization: `legacy_her`, `legacy_eth`, `reacher`.

**`fovs`** -- Core data table. One row per imaging session. Contains two generated columns:
- `effective_fps` = `fps / frame_averaging`
- `interframe_interval_ms` = `1000.0 / fps`

Unique on `(subject_id, name)`.

**`neural_traces`** -- 1:1 with `fovs`. Stores the full `(neurons x frames)` matrix as a float32 BLOB. The `dtype` column records the serialization format.

**`frame_timestamps`** -- 1:1 with `fovs`. Stores the 1-D timestamp array as a float64 BLOB. The `source` column records origin: `"external_csv"` (REACHER) or `"code_9"` (legacy).

**`events`** -- All behavioral events for a FOV. `t1_ms` and `t2_ms` are event start/end timestamps in milliseconds. `frame_index` and `end_frame_index` are the corresponding frame indices (computed via `np.searchsorted` during ingestion).

**`neurons`** -- Per-cell metadata. Ingestion creates placeholder rows (one per neuron index). The `label`, `cell_type`, `region`, `x_um`, and `y_um` columns are available for annotation but are not populated automatically.

**`raw_files`** -- Archive of source files. `file_type` is one of `"event_csv"`, `"event_mat"`, `"signal_npy"`. Multi-part files use `part_number`. Each file has a SHA-256 checksum.

#### Indexes

| Index | Columns | Purpose |
|-------|---------|---------|
| `idx_events_fov_code` | `events(fov_id, code)` | Fast event lookup by code |
| `idx_events_fov_label` | `events(fov_id, label)` | Fast event lookup by label |
| `idx_fovs_subject` | `fovs(subject_id)` | List FOVs for a subject |
| `idx_subjects_population` | `subjects(population_id)` | List subjects for a population |
| `idx_populations_project` | `populations(project_id)` | List populations for a project |
| `idx_raw_files_fov` | `raw_files(fov_id)` | List raw files for a FOV |
| `idx_neurons_fov` | `neurons(fov_id)` | List neurons for a FOV |

---

### 4.2 Design Rationale

#### Why DuckDB?

- **Embedded** -- no server process, no Docker, no network. A single file at `~/.pynapse/pynapse.duckdb`.
- **Columnar analytics** -- fast aggregations over thousands of events without indexes on every column.
- **Python-native** -- `duckdb.connect()` returns a connection that speaks SQL and returns pandas DataFrames directly via `.fetchdf()`.
- **Zero config** -- first call to `get_connection()` creates the directory, file, schema, and paradigm seeds automatically.

#### Normalization decisions

The schema is normalized to 3NF with deliberate denormalization for performance:

- `fovs.num_neurons` and `fovs.num_frames` duplicate what could be derived from `neural_traces`, but they allow metadata queries without touching BLOBs.
- `events.label` duplicates the paradigm's `event_dict` mapping, but it enables direct label-based queries without joins.
- Generated columns (`effective_fps`, `interframe_interval_ms`) avoid recomputing common values.

#### BLOB strategy

Neural traces and frame timestamps are stored as raw byte BLOBs rather than in row-per-value tables:

- A typical FOV has ~100 neurons x ~10,000 frames = 1M float32 values. Storing each as a row would create 1M rows per FOV.
- BLOB storage is ~4 MB per FOV (float32). DuckDB handles this efficiently with its columnar compression.
- Deserialization is a single `np.frombuffer()` call.

#### Raw file storage

Source files are copied (not moved) into `~/.pynapse/raw/<fov_id>/` with SHA-256 checksums. This ensures:

- Original data is never modified.
- Files can be re-processed if the alignment pipeline changes.
- Integrity can be verified by comparing checksums.

Set `copy_raw=False` to skip this step (e.g., when source files are already on a shared drive).

---

### 4.3 Pipeline Internals

#### Ingestion flow

```
ingest.fov()
  │
  ├── Create Sample(event_data, signal_data, fps, ...)
  │     ├── EventLog parses .mat or .csv
  │     ├── SignalRecording loads .npy
  │     └── Frame timestamps: external CSV or code-9 extraction
  │
  ├── Upsert hierarchy
  │     ├── _get_or_create_project()
  │     ├── _get_or_create_population()
  │     └── _get_or_create_subject()
  │
  └── from_sample(sample, subject_id, paradigm_name)
        ├── sample.get_signals()          → neural trace matrix
        ├── sample.get_dataframe()        → events with frame_index
        ├── sample._get_frame_timestamps() → aligned timestamp array
        │
        ├── INSERT INTO fovs
        ├── INSERT INTO neural_traces     (BLOB)
        ├── INSERT INTO frame_timestamps  (BLOB)
        ├── INSERT INTO events            (one row per event)
        ├── INSERT INTO neurons           (placeholder rows)
        └── _store_raw_files()            (copy + SHA-256)
```

#### Frame timestamp alignment: legacy .mat vs REACHER CSV

**Legacy `.mat` path:**
1. Event code 9 (`frame_trigger`) timestamps are extracted from the raw event log.
2. If `frame_correction=True`, a correction `.mat` file is used to reconstruct dropped frames.
3. Otherwise, `_handle_missed_frames()` detects gaps > 1.5x the expected interval and inserts synthetic timestamps.
4. The full timestamp array is downsampled by `frame_averaging` (e.g., every 4th timestamp).

**REACHER CSV path:**
1. Frame timestamps are provided as a separate CSV file with `timestamp_ms` and `frame_index` columns.
2. No code-9 extraction or gap correction is needed.
3. The array is downsampled by `frame_averaging` as above.

In both cases, event `frame_index` values are computed via `np.searchsorted(frame_ts, event_t1, side="right") - 1`, mapping each event to the last frame that started before or at the event timestamp.

#### Source format detection

The `source_format` column on `fovs` is set automatically:

| Detected format | Condition |
|-----------------|-----------|
| `"reacher_csv"` | Event source file ends with `.csv` |
| `"legacy_mat"` | Event source file ends with `.mat` |
| `"unknown"` | Neither detected |

The frame timestamp `source` column is:

| Source | Condition |
|--------|-----------|
| `"external_csv"` | `frame_timestamps` argument was provided |
| `"code_9"` | Frame triggers extracted from event log |

---

## 5. API Reference

### 5.1 `pynapse.db.engine`

Connection factory and managed paths.

---

#### `get_connection(db_path=None)`

Get or create a cached (module-level singleton) DuckDB connection. Initializes the schema on first use.

**Parameters:**
- `db_path` (str | Path | None) -- Path to the database file. Defaults to `~/.pynapse/pynapse.duckdb`.

**Returns:** `duckdb.DuckDBPyConnection`

**Example:**
```python
conn = db.get_connection()
```

> **Note:** This returns the same connection object on repeated calls. Use `connect()` if you need independent connections.

---

#### `connect(db_path=None)`

Open a new independent DuckDB connection with initialized schema. The caller is responsible for closing it.

**Parameters:**
- `db_path` (str | Path | None) -- Path to the database file. Defaults to `~/.pynapse/pynapse.duckdb`. Pass `":memory:"` for an in-memory database.

**Returns:** `duckdb.DuckDBPyConnection`

**Example:**
```python
conn = db.connect(":memory:")  # ephemeral, great for testing
# ... use conn ...
conn.close()
```

---

#### `close()`

Close the cached module-level connection (the one returned by `get_connection()`).

**Example:**
```python
db.close()
```

---

#### `get_raw_dir(base_dir=None)`

Return the managed raw-file directory, creating it if needed.

**Parameters:**
- `base_dir` (str | Path | None) -- Custom base directory. Defaults to `~/.pynapse/raw`.

**Returns:** `Path`

---

### 5.2 `pynapse.db.schema`

DDL, indexes, paradigm seeding, and schema versioning.

---

#### `initialize(conn)`

Create all tables, indexes, and seed paradigms. Idempotent -- safe to call multiple times.

**Parameters:**
- `conn` -- A DuckDB connection.

**Example:**
```python
import duckdb
conn = duckdb.connect(":memory:")
from pynapse.db import schema
schema.initialize(conn)
```

> **Note:** You rarely need to call this directly. Both `get_connection()` and `connect()` call it automatically.

#### Constants

- `SCHEMA_VERSION = "1"` -- Current schema version, stored in `_meta`.

#### Seeded paradigms

Three paradigms are inserted on first initialization:

| Name | Event dict source | Description |
|------|-------------------|-------------|
| `legacy_her` | `LEGACY_HER` from `pynapse.config.events` | Heroin head-fixed self-administration (2024) |
| `legacy_eth` | `LEGACY_ETH` from `pynapse.config.events` | Ethanol head-fixed self-administration (2025) |
| `reacher` | `REACHER` from `pynapse.config.events` | REACHER self-administration (2026+) |

---

### 5.3 `pynapse.db.ingest`

Ingestion pipeline: raw files or pynapse objects to DuckDB.

---

#### `fov(neural, events, sample_name, fov_name, population_name, project_name, paradigm_name, fps, ...)`

Ingest a single FOV from raw file paths. Creates a `Sample` internally, upserts the full hierarchy, and stores everything.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `neural` | str \| Path \| list | *required* | Path(s) to `.npy` signal files |
| `events` | str \| Path \| list | *required* | Path(s) to event files (`.csv` or `.mat`) |
| `sample_name` | str | *required* | Subject name |
| `fov_name` | str | *required* | Unique FOV name |
| `population_name` | str | *required* | Population name (get-or-create) |
| `project_name` | str | *required* | Project name (get-or-create) |
| `paradigm_name` | str | *required* | Paradigm name (must exist) |
| `fps` | float | *required* | Raw imaging frame rate |
| `frame_averaging` | int | `1` | Number of frames averaged |
| `frame_timestamps` | str \| Path \| ndarray \| None | `None` | External frame timestamp source |
| `frame_correction` | bool | `False` | Use legacy correction file |
| `correction_file` | str \| Path \| None | `None` | Path to correction `.mat` file |
| `start_time` | float | `0.0` | Session start offset (ms) |
| `conn` | DuckDBPyConnection \| None | `None` | Database connection |
| `copy_raw` | bool | `True` | Copy source files to managed dir |
| `raw_dir` | str \| Path \| None | `None` | Custom raw file directory |
| `project_description` | str \| None | `None` | Description for new project |
| `population_description` | str \| None | `None` | Description for new population |

**Returns:** `int` -- the newly created `fov_id`.

**Raises:** `ValueError` if `paradigm_name` is not a known paradigm.

---

#### `from_sample(sample, subject_id, paradigm_name, fov_name=None, conn=None, copy_raw=True, raw_dir=None)`

Ingest a pre-constructed `Sample` object.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sample` | Sample | *required* | A fully constructed Sample |
| `subject_id` | int | *required* | Subject ID (must exist in DB) |
| `paradigm_name` | str | *required* | Paradigm name (must exist) |
| `fov_name` | str \| None | `None` | Override FOV name (defaults to `sample.name`) |
| `conn` | DuckDBPyConnection \| None | `None` | Database connection |
| `copy_raw` | bool | `True` | Copy source files |
| `raw_dir` | str \| Path \| None | `None` | Custom raw file directory |

**Returns:** `int` -- the `fov_id`.

---

#### `from_population(population, project_id, paradigm_name, conn=None, copy_raw=True, raw_dir=None)`

Ingest all samples in a `Population`.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `population` | Population | *required* | A Population object |
| `project_id` | int | *required* | Project ID (must exist in DB) |
| `paradigm_name` | str | *required* | Paradigm name |
| `conn` | DuckDBPyConnection \| None | `None` | Database connection |
| `copy_raw` | bool | `True` | Copy source files |
| `raw_dir` | str \| Path \| None | `None` | Custom raw file directory |

**Returns:** `list[int]` -- list of `fov_id` values.

---

#### `from_project(project, paradigm_name, conn=None, copy_raw=True, raw_dir=None)`

Ingest all populations in a `Project`.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `project` | Project | *required* | A Project object |
| `paradigm_name` | str | *required* | Paradigm name |
| `conn` | DuckDBPyConnection \| None | `None` | Database connection |
| `copy_raw` | bool | `True` | Copy source files |
| `raw_dir` | str \| Path \| None | `None` | Custom raw file directory |

**Returns:** `list[int]` -- flat list of all `fov_id` values created.

---

### 5.4 `pynapse.db.query`

Query functions for the pynapse database. All functions accept an optional `conn` parameter; if omitted, the cached module-level connection is used.

---

#### `list_projects(conn=None)`

**Returns:** DataFrame with columns: `id`, `name`, `description`, `authors`, `created_at`.

---

#### `list_populations(project_name=None, project_id=None, conn=None)`

**Parameters:**
- `project_name` (str | None) -- Filter by project name.
- `project_id` (int | None) -- Filter by project ID.
- If both are `None`, returns all populations.

**Returns:** DataFrame with columns: `id`, `name`, `description`, `project_id`, `created_at`.

---

#### `list_subjects(population_name=None, population_id=None, conn=None)`

**Parameters:**
- `population_name` (str | None) -- Filter by population name.
- `population_id` (int | None) -- Filter by population ID.
- If both are `None`, returns all subjects.

**Returns:** DataFrame with columns: `id`, `name`, `sex`, `genotype`, `species`, `population_id`.

---

#### `list_fovs(subject_id=None, conn=None)`

**Parameters:**
- `subject_id` (int | None) -- Filter by subject ID. If `None`, returns all FOVs.

**Returns:** DataFrame with columns: `id`, `name`, `subject_id`, `paradigm_id`, `fps`, `frame_averaging`, `effective_fps`, `num_neurons`, `num_frames`, `source_format`, `ingested_at`.

---

#### `get_fov(fov_id=None, name=None, conn=None)`

Return a single FOV row as a dict.

**Parameters:**
- `fov_id` (int | None) -- Lookup by ID.
- `name` (str | None) -- Lookup by name.
- At least one must be provided.

**Returns:** `dict` with all `fovs` columns, or `None` if not found.

**Raises:** `ValueError` if neither `fov_id` nor `name` is provided.

---

#### `get_traces(fov_id, conn=None)`

Load the neural trace matrix for a FOV.

**Parameters:**
- `fov_id` (int) -- The FOV ID.

**Returns:** `numpy.ndarray` of shape `(num_neurons, num_frames)`, dtype `float32`.

**Raises:** `KeyError` if no neural traces exist for the given `fov_id`.

---

#### `get_frame_timestamps(fov_id, conn=None)`

Load the 1-D frame timestamp array for a FOV.

**Parameters:**
- `fov_id` (int) -- The FOV ID.

**Returns:** `numpy.ndarray` of timestamps in ms, dtype `float64`.

**Raises:** `KeyError` if no frame timestamps exist for the given `fov_id`.

---

#### `get_events(fov_id, label=None, code=None, conn=None)`

Return events for a FOV as a DataFrame.

**Parameters:**
- `fov_id` (int) -- The FOV ID.
- `label` (str | None) -- Filter events by label.
- `code` (int | None) -- Filter events by code.

**Returns:** DataFrame with columns: `code`, `t1`, `t2`, `label`, `frame_index` (ordered by `t1`).

---

#### `get_event_summary(fov_id, conn=None)`

Return event counts grouped by label and code.

**Parameters:**
- `fov_id` (int) -- The FOV ID.

**Returns:** DataFrame with columns: `code`, `label`, `count` (ordered by `code`).

---

#### `get_event_frame_indices(fov_id, label=None, code=None, conn=None)`

Return frame indices for matching events as an integer array.

**Parameters:**
- `fov_id` (int) -- The FOV ID.
- `label` (str | None) -- Filter by label.
- `code` (int | None) -- Filter by code.

**Returns:** `numpy.ndarray` of integer frame indices (ordered by `t1_ms`).

---

### 5.5 `pynapse.db.hydrate`

`DBSample` -- a Sample-compatible object backed by the pynapse database.

---

#### `class DBSample(TensorConfigMixin)`

A lightweight drop-in replacement for `Sample` that reads data lazily from DuckDB. Data is cached after the first access.

**Constructor:**

```python
DBSample(
    fov_id: int,
    conn=None,
    default_event_id: int | list[int] | None = None,
    default_pre_event: float | None = None,
    default_post_event: float | None = None,
    default_buffer_ms: int = 0,
    default_min_trials: int = 1,
)
```

**Parameters:**
- `fov_id` -- The FOV ID to load from the database.
- `conn` -- DuckDB connection. Uses cached connection if omitted.
- `default_event_id` -- Default event code(s) for `get_tensor()`.
- `default_pre_event` -- Default pre-event window (seconds).
- `default_post_event` -- Default post-event window (seconds).
- `default_buffer_ms` -- Default minimum inter-event interval (ms).
- `default_min_trials` -- Default minimum trial count.

**Raises:** `KeyError` if no FOV exists with the given `fov_id`.

**Properties:**

| Property | Type | Description |
|----------|------|-------------|
| `name` | str | FOV name |
| `fps` | float | Raw imaging frame rate |
| `frame_averaging` | int | Number of frames averaged |
| `effective_fps` | float | `fps / frame_averaging` |
| `interframe_interval` | float | `1000.0 / fps` (ms) |
| `num_neurons` | int | Number of neurons |
| `num_frames` | int | Number of frames |
| `num_events` | int | Total event count |

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `get_signals()` | `ndarray (neurons, frames)` | Neural trace matrix (cached) |
| `get_dataframe()` | `DataFrame` | Events with `code, t1, t2, label, frame_index` (cached) |
| `_get_frame_timestamps()` | `ndarray` | 1-D timestamp array in ms (cached) |
| `get_event_dict()` | `dict[int, str]` | Event code-to-label mapping from paradigm (cached) |
| `get_num_events(event_id)` | `int` | Count of events with the given code |
| `count_events(target=None)` | `int` | Count events by code (int), label (str), codes (list), or all (None) |
| `get_event_timestamps(event_id)` | `ndarray` | Array of `t1` timestamps for the given event code |
| `get_tensor(...)` | `SampleEventTensor` | Create/retrieve a peri-event tensor (inherited from `TensorConfigMixin`) |
| `clear_tensor_cache()` | None | Free cached tensors (inherited from `TensorConfigMixin`) |

---

## 6. Troubleshooting

### `KeyError: "No neural traces for fov_id=X"`

The FOV exists in the `fovs` table but has no corresponding row in `neural_traces`. This should not happen during normal ingestion. Check if ingestion was interrupted:

```python
conn = db.connect()
fov = db.query.get_fov(fov_id=X, conn=conn)
print(fov)  # Verify the FOV exists

# Check for orphaned FOV
result = conn.execute(
    "SELECT COUNT(*) FROM neural_traces WHERE fov_id = ?", [X]
).fetchone()
print(f"Trace rows: {result[0]}")
```

If the FOV is incomplete, delete it and re-ingest:

```python
conn.execute("DELETE FROM events WHERE fov_id = ?", [X])
conn.execute("DELETE FROM neurons WHERE fov_id = ?", [X])
conn.execute("DELETE FROM neural_traces WHERE fov_id = ?", [X])
conn.execute("DELETE FROM frame_timestamps WHERE fov_id = ?", [X])
conn.execute("DELETE FROM raw_files WHERE fov_id = ?", [X])
conn.execute("DELETE FROM fovs WHERE id = ?", [X])
```

### `ValueError: "Unknown paradigm 'X'"`

The paradigm name passed to `ingest.fov()` or `ingest.from_sample()` does not match any seeded paradigm. Valid names are `"reacher"`, `"legacy_her"`, `"legacy_eth"`.

```python
# Check available paradigms
result = conn.execute("SELECT name FROM paradigms").fetchdf()
print(result)
```

To add a custom paradigm:

```python
import json
conn.execute(
    "INSERT INTO paradigms (name, event_dict, color_dict, description) VALUES (?, ?, ?, ?)",
    ["my_paradigm", json.dumps({1: "event_a", 2: "event_b"}), "{}", "Custom paradigm"],
)
```

### `KeyError: "No FOV with id=X"` from `DBSample`

The `fov_id` passed to `DBSample()` does not exist. Verify with:

```python
fovs = db.query.list_fovs(conn=conn)
print(fovs[["id", "name"]])
```

### Database health check

```python
conn = db.connect()

# Schema version
version = conn.execute("SELECT value FROM _meta WHERE key = 'schema_version'").fetchone()
print(f"Schema version: {version[0]}")

# Row counts
for table in ["projects", "populations", "subjects", "fovs", "events", "neurons",
              "neural_traces", "frame_timestamps", "raw_files", "paradigms"]:
    count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    print(f"{table}: {count} rows")

conn.close()
```

### Reset / rebuild the database

> **Warning:** This deletes all ingested data. Raw files in `~/.pynapse/raw/` are NOT deleted automatically.

```python
from pathlib import Path

db_path = Path.home() / ".pynapse" / "pynapse.duckdb"
if db_path.exists():
    db_path.unlink()
    print("Database deleted. It will be recreated on next connect().")
```

To also remove raw file copies:

```bash
rm -rf ~/.pynapse/raw/
```

---

## 7. Glossary

| Term | Definition |
|------|------------|
| **FOV** | Field of View. A single imaging session from one region of tissue. The fundamental unit of data in the database. |
| **Sample** | A pynapse `Sample` object that encapsulates aligned neural signals and events from raw files. `DBSample` is its database-backed equivalent. |
| **Population** | A group of subjects within a project (e.g., "EtOH", "Saline", "Control"). Maps to an experimental group. |
| **Project** | Top-level container for an experiment. Contains one or more populations. |
| **Subject** | An individual animal. Belongs to one population. |
| **Paradigm** | A task definition: a mapping of integer event codes to human-readable labels, plus associated colors. Seeded from `pynapse.config.events`. |
| **Event** | A behavioral occurrence during a session, defined by an integer `code`, a text `label`, a start timestamp (`t1_ms`), an optional end timestamp (`t2_ms`), and a `frame_index`. |
| **Event code** | An integer identifying an event type. Defined per paradigm (e.g., `101` = `rh_lever_active_press` in REACHER). |
| **Trace** | A 1-D time series of fluorescence values for a single neuron across all frames. The full trace matrix is `(neurons, frames)`. |
| **Frame index** | The zero-based index into the frame timestamp array. Maps a time in ms to a specific imaging frame. |
| **Frame timestamp** | The time (in ms) at which an imaging frame was acquired. |
| **Peri-event window** | A slice of neural data centered around an event occurrence: `[event - pre_event, event + post_event]` in seconds. Extracted by `SampleEventTensor`. |
| **Tensor** | A 3-D numpy array of shape `(trials, neurons, frames)` containing peri-event windows for one event type across all occurrences. |
| **Alignment** | The process of mapping event timestamps (ms) to frame indices using `np.searchsorted` against the frame timestamp array. |
| **Frame averaging** | Hardware-level averaging of consecutive frames (e.g., 4x). Reduces effective FPS: `effective_fps = fps / frame_averaging`. |
| **Effective FPS** | The actual frame rate after accounting for frame averaging. |
| **Interframe interval** | Time between consecutive raw frames: `1000.0 / fps` (ms). |
| **BLOB** | Binary Large Object. Neural traces and frame timestamps are stored as raw byte arrays in DuckDB. |
| **Source format** | The origin format of event data: `"reacher_csv"` or `"legacy_mat"`. Stored on the `fovs` table. |
| **Event dict** | A Python dictionary mapping integer event codes to string labels. Defined in `pynapse/config/events.py`. |
