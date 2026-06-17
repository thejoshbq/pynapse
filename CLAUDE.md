# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Pynapse is the data engine for the REACHER analysis stack: it loads raw two-photon fluorescence + behavioral event logs, aligns timescales, applies preprocessing, extracts peri-event tensors, and (optionally) persists everything to DuckDB. Higher-level tools like `axplorer` consume pynapse objects rather than reimplementing this logic.

## Commands

```bash
pip install -e ".[dev]"             # install with pytest + pytest-cov
pytest                              # run full test suite
pytest tests/test_db.py             # single module
pytest tests/test_core_io.py::TestEventLog::test_count_events_by_code   # single test
pytest -m unit                      # only unit-marked tests (markers: unit, integration, slow)
pytest --cov=pynapse --cov-report=html
```

There is **no linter/formatter configured for pynapse** (the wider Phoxel Workbench uses ruff, but pynapse is not ruff-configured). Don't add one without asking.

The dataset-ingestion script under `scripts/` (`ingest_prl_nac_g6_her.py`) is hard-coded to a local data path — treat it as a per-dataset template, not a generic CLI.

## Core architecture

### Three-tier data hierarchy

`Project → Population → Sample` (a `Sample` is one FOV / imaging session). All three implement `TensorConfigMixin` (`pynapse/core/mixins.py`), which provides `.get_tensor(...)` with parameter defaults + a tensor cache. Subclasses must override `create_tensor()` to return the right `EventTensor` flavor.

`Sample` (`pynapse/core/sample.py`) wires together:
- `EventLog` (`core/io/behavior.py`) — parses MATLAB `.mat` (legacy) or REACHER `.csv` event logs
- `SignalRecording` (`core/io/microscopy.py`) — loads `.npy` fluorescence arrays of shape `(neurons, frames)`
- Frame-timestamp alignment — derived from event-code 9 frame triggers (legacy) or an external `frame_timestamps` CSV (REACHER)

Two FPS concepts coexist and are not interchangeable:
- `Sample.fps` = raw imaging rate (used for `interframe_interval = 1000.0 / fps`)
- `Sample.effective_fps` = `fps / frame_averaging` (what tensor extraction uses to convert seconds → frame indices)

### Peri-event tensor extraction

`pynapse/analysis/peri_event.py` defines `EventTensor`, `SampleEventTensor`, and `PopulationEventTensor`. Construction takes an `event_id` (or list), `pre_event`/`post_event` in seconds, optional `buffer_ms` (drop events closer than N ms apart), and optional `min_trials`. Two preprocessing slots:
- `trace_preprocess` — applied to the full trace before window extraction
- `window_preprocess` — applied per extracted window (this is where `OTIS_PIPE` lives)

### Preprocessing — two parallel families

`pynapse/analysis/preprocessing/` splits into:
- `epoch/` — per-window: `DFOverF`, `ZScore`, `GaussianSmoothing`, plus the canonical `OTIS_PIPE` (DF/F → Z-score `(-3000, -500) ms` → Gaussian smoothing). This is the lab-standard pipeline used in publications.
- `continuous/` — full-trace: `BaselineSubtraction`, `Normalize`. Marked **legacy** — prefer epoch preprocessors for new work.

All preprocessors subclass `Preprocessor` (`base.py`) and compose via `Pipeline` (`pipeline.py`).

### Event configuration — single source of truth

`pynapse/config/events.py` is the **only** place event code → label dictionaries live. `LEGACY_HER`, `LEGACY_ETH`, and `REACHER` are exposed via `TASK_TO_DICT["legacy_her" | "legacy_eth" | "reacher"]`. `COLORS` provides standard plot colors per label. **Never define event dictionaries elsewhere** — analysis code, viz, DB schemas, and the (planned) LLM module all import from here.

REACHER codes are grouped by device: 100s levers, 200s drug delivery, 300s cues, 400s stim, 500s licking, 600s pavlovian, 700s session control, 900s unknown auto-assigned.

### DuckDB persistence layer (`pynapse/db/`)

The DB layer caches aligned data so repeat analyses skip raw-file parsing. Default file: `~/.pynapse/pynapse.duckdb`; raw-file copies + SHA-256 checksums in `~/.pynapse/raw/<fov_id>/`.

Hierarchy in DB: `projects → populations → subjects → fovs`, with `events`, `neurons`, `neural_traces` (float32 BLOB), `frame_timestamps` (float64 BLOB), `raw_files`, `paradigms` (seeded from `config/events.py`), and a `_meta` versioning table. Generated columns on `fovs`: `effective_fps`, `interframe_interval_ms`.

Module roles:
- `engine.py` — `connect()` (fresh, caller closes) vs `get_connection()` (cached module-level singleton); `close()` clears the cached one.
- `schema.py` — DDL + paradigm seeding; `SCHEMA_VERSION` lives here.
- `ingest.py` — entry points from most manual to most automated: `fov(...)` (raw paths), `from_sample(sample, subject_id, ...)`, `from_population(pop, project_id, ...)`, `from_project(proj, ...)`.
- `query.py` — `get_traces`, `get_events`, `get_event_summary`, `get_fov`, etc.
- `hydrate.py` — `DBSample`: drop-in replacement for `Sample` that lazy-loads from DuckDB. Exposes the same interface `SampleEventTensor` consumes (`.effective_fps`, `.interframe_interval`, `.num_neurons`, `.get_dataframe()`, `.get_signals()`, `._get_frame_timestamps()`, `.get_num_events()`).

When adding new `Sample` properties or methods that tensors/analysis depend on, update `DBSample` in lockstep — otherwise DB-backed analysis silently diverges from file-backed analysis.

See `docs/database-guide.md` for the full DB user/architecture reference and `docs/motherduck-migration.md` for the cloud-migration plan.

## Conventions

- **Python ≥ 3.8** per `pyproject.toml` (vs. ≥ 3.10 elsewhere in Phoxel Workbench). The DB layer assumes 3.10+ idioms in places — if you touch `db/`, target 3.10+.
- **Dependencies are pinned** in `pyproject.toml`. Don't bump versions casually.
- **Test suite has known pre-existing failures** in `test_core_io`, `test_peri_event`, `test_preprocessing` unrelated to current DB work. `test_db.py` (35 unit tests, in-memory DuckDB) and `test_db_integration.py` (real REACHER data, skipped if absent) should stay green.
- **Pytest markers**: `unit`, `integration`, `slow` — use them when adding tests so the suite stays filterable.
- **Don't add event dictionaries outside `config/events.py`.** Extend `REACHER`/`LEGACY_*` in place and re-seed paradigms via `db/schema.py` if needed.
