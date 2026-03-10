# MotherDuck Migration Guide

## Table of Contents

- [1. Overview](#1-overview)
- [2. Prerequisites](#2-prerequisites)
- [3. Migration Steps](#3-migration-steps)
  - [3.1 Backup](#31-backup)
  - [3.2 Connection Configuration](#32-connection-configuration)
  - [3.3 Authentication](#33-authentication)
  - [3.4 Version Bump](#34-version-bump)
  - [3.5 Database Upload](#35-database-upload)
  - [3.6 Raw File Storage](#36-raw-file-storage)
  - [3.7 Pipeline & Query Updates](#37-pipeline--query-updates)
  - [3.8 Verification Checklist](#38-verification-checklist)
- [4. Hybrid Execution](#4-hybrid-execution)
- [5. Rollback](#5-rollback)
- [6. Post-Migration Considerations](#6-post-migration-considerations)
- [7. Troubleshooting](#7-troubleshooting)

---

## 1. Overview

### What is MotherDuck?

MotherDuck is a cloud-hosted DuckDB service. It runs a full DuckDB engine on remote infrastructure and exposes it through the same Python `duckdb` package you already use. The connection string changes from a local file path to a `md:` URI -- everything else (SQL dialect, Python API, `.fetchdf()`, `.fetchnumpy()`) stays identical.

### What changes?

| Component | Before (local) | After (MotherDuck) |
|-----------|----------------|---------------------|
| Connection string | `~/.pynapse/pynapse.duckdb` | `md:pynapse?motherduck_token=...` |
| `engine.py` | Hardcoded local path, `mkdir()` on every connect | Env-var-driven path, guards for remote URIs |
| `pyproject.toml` | `duckdb>=1.0.0` | `duckdb>=1.4.0` |
| Raw files (`raw/`) | Local filesystem | Unchanged (local-only); see [3.6](#36-raw-file-storage) |
| Schema DDL | All `CREATE IF NOT EXISTS` | Identical -- fully compatible |
| Query functions | All `conn=None` fallback | Identical -- no changes needed |
| `DBSample` | Lazy caching from local DB | Lazy caching from MotherDuck (same API) |

### What stays the same?

Nearly everything. The pynapse schema (10 tables, 8 sequences, 2 generated columns, 7 indexes) is fully compatible with MotherDuck. All query functions in `query.py` use standard DuckDB SQL and return types (`.fetchdf()`, `.fetchnumpy()`, `.fetchone()`). `DBSample` in `hydrate.py` works unchanged because it only holds a connection reference and calls `query.*` functions.

### Scope of this guide

This document covers:

1. Preparing and backing up your local database
2. Code changes to `engine.py` and `pyproject.toml` (presented as copy-paste diffs)
3. Uploading your database to MotherDuck
4. Verifying the migration
5. Operating in hybrid (local + cloud) mode
6. Rolling back if needed

> **Note:** This guide describes code changes but does not apply them. You will manually edit the files using the diffs provided.

---

## 2. Prerequisites

### MotherDuck account

1. Create an account at [motherduck.com](https://motherduck.com)
2. Generate an API token from the MotherDuck UI (Settings > API Tokens)
3. Save the token securely -- you will need it for the `motherduck_token` environment variable

### DuckDB version

MotherDuck requires DuckDB **1.4.0 or later**. Check your installed version:

```python
import duckdb
print(duckdb.__version__)  # Must be >= 1.4.0
```

Or from the command line:

```bash
python -c "import duckdb; print(duckdb.__version__)"
```

> **Note:** The pynapse venv currently pins `duckdb>=1.0.0` in `pyproject.toml`. Step [3.4](#34-version-bump) updates this constraint. If your installed version is already >= 1.4.0, you only need the constraint change for future installs.

### Backup your local database

Before making any changes, back up your data. Step [3.1](#31-backup) walks through this in detail, but the short version:

```bash
cp ~/.pynapse/pynapse.duckdb ~/.pynapse/pynapse.duckdb.bak
cp -r ~/.pynapse/raw ~/.pynapse/raw.bak
```

---

## 3. Migration Steps

Each step follows a **WHAT / WHY / VERIFY** structure.

### 3.1 Backup

**WHAT:** Snapshot the local DuckDB file and the managed raw-file directory.

```bash
# Create a timestamped backup
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
cp ~/.pynapse/pynapse.duckdb ~/.pynapse/pynapse.duckdb.bak.$TIMESTAMP
cp -r ~/.pynapse/raw ~/.pynapse/raw.bak.$TIMESTAMP
```

**WHY:** The upload step ([3.5](#35-database-upload)) reads from the local file but does not modify it. Still, having a snapshot lets you restore quickly if anything goes wrong downstream.

> **Warning:** If your `raw/` directory is large (tens of GB), the copy may take a while. Consider `rsync` for interruptible copies: `rsync -a ~/.pynapse/raw/ ~/.pynapse/raw.bak.$TIMESTAMP/`

**VERIFY:**

```bash
ls -lh ~/.pynapse/pynapse.duckdb.bak.*
ls -d ~/.pynapse/raw.bak.*
```

Both should exist and match the size of the originals.

---

### 3.2 Connection Configuration

**WHAT:** Modify `pynapse/db/engine.py` to support remote connection strings (like `md:`) and read the database path from an environment variable.

**WHY:** The current `engine.py` has two issues that prevent MotherDuck connections:

1. **`mkdir()` crash** -- `get_connection()` (line 22) calls `Path(db_path).parent.mkdir()` on every connection. For a `md:pynapse` string, `Path("md:pynapse").parent` resolves to `Path(".")` and `mkdir()` may fail or create an unwanted directory. `connect()` (line 38) has the same issue but already guards against `:memory:`.
2. **No env var support** -- the database path is hardcoded to `~/.pynapse/pynapse.duckdb`. There is no way to switch to a MotherDuck URI without editing source code.

**CURRENT CODE** (`pynapse/db/engine.py`):

```python
# engine.py
# Connection factory and managed paths for the pynapse database.

import duckdb
from pathlib import Path

_DEFAULT_DB_DIR = Path.home() / ".pynapse"
_DEFAULT_DB_PATH = _DEFAULT_DB_DIR / "pynapse.duckdb"
_DEFAULT_RAW_DIR = _DEFAULT_DB_DIR / "raw"

_connection = None


def get_connection(db_path=None):
    """Get or create a cached DuckDB connection, initializing schema on first use."""
    global _connection
    if _connection is not None:
        return _connection
    if db_path is None:
        db_path = _DEFAULT_DB_PATH
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    _connection = duckdb.connect(str(db_path))
    from . import schema
    schema.initialize(_connection)
    return _connection


def connect(db_path=None):
    """Open a new independent DuckDB connection with initialized schema.

    Unlike ``get_connection``, this always creates a fresh connection and
    does **not** cache it.  The caller is responsible for closing it.
    """
    if db_path is None:
        db_path = _DEFAULT_DB_PATH
    if str(db_path) != ":memory:":
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(db_path))
    from . import schema
    schema.initialize(conn)
    return conn


def close():
    """Close the cached module-level connection."""
    global _connection
    if _connection is not None:
        _connection.close()
        _connection = None


def get_raw_dir(base_dir=None):
    """Return the managed raw-file directory, creating it if needed."""
    d = Path(base_dir) if base_dir else _DEFAULT_RAW_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d
```

**RECOMMENDED REPLACEMENT** (`pynapse/db/engine.py`):

```python
# engine.py
# Connection factory and managed paths for the pynapse database.

import os

import duckdb
from pathlib import Path

_DEFAULT_DB_DIR = Path.home() / ".pynapse"
_DEFAULT_DB_PATH = _DEFAULT_DB_DIR / "pynapse.duckdb"
_DEFAULT_RAW_DIR = _DEFAULT_DB_DIR / "raw"

_connection = None


def _is_remote_or_memory(db_path):
    """Return True if db_path is a MotherDuck URI or in-memory."""
    s = str(db_path)
    return s == ":memory:" or s.startswith("md:")


def get_connection(db_path=None):
    """Get or create a cached DuckDB connection, initializing schema on first use."""
    global _connection
    if _connection is not None:
        return _connection
    if db_path is None:
        db_path = os.environ.get("PYNAPSE_DB_PATH", str(_DEFAULT_DB_PATH))
    if not _is_remote_or_memory(db_path):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    _connection = duckdb.connect(str(db_path))
    from . import schema
    schema.initialize(_connection)
    return _connection


def connect(db_path=None):
    """Open a new independent DuckDB connection with initialized schema.

    Unlike ``get_connection``, this always creates a fresh connection and
    does **not** cache it.  The caller is responsible for closing it.
    """
    if db_path is None:
        db_path = os.environ.get("PYNAPSE_DB_PATH", str(_DEFAULT_DB_PATH))
    if not _is_remote_or_memory(db_path):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(db_path))
    from . import schema
    schema.initialize(conn)
    return conn


def close():
    """Close the cached module-level connection."""
    global _connection
    if _connection is not None:
        _connection.close()
        _connection = None


def get_raw_dir(base_dir=None):
    """Return the managed raw-file directory, creating it if needed."""
    d = Path(base_dir) if base_dir else _DEFAULT_RAW_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d
```

**Summary of changes:**

| Change | Lines affected | Purpose |
|--------|---------------|---------|
| `import os` | Line 3 (new) | Read environment variables |
| `_is_remote_or_memory()` | Lines 15--17 (new) | Centralized check for `md:` and `:memory:` |
| `get_connection()` env var | Line 22 | Falls back to `PYNAPSE_DB_PATH` env var before hardcoded default |
| `get_connection()` mkdir guard | Line 24 | Skips `mkdir()` for remote/memory connections |
| `connect()` env var | Line 33 | Same env var fallback as `get_connection()` |
| `connect()` mkdir guard | Line 35 | Replaces the old `:memory:`-only check with the unified helper |

> **Note:** The `_is_remote_or_memory()` helper also fixes a pre-existing bug: `get_connection()` never guarded against `:memory:` paths (unlike `connect()` which did). Both functions now use the same guard.

**VERIFY:**

```python
# Quick smoke test after applying the diff
from pynapse.db import engine

# Should not crash (no mkdir on "md:" string)
import os
os.environ["PYNAPSE_DB_PATH"] = ":memory:"
conn = engine.connect()
conn.execute("SELECT 1").fetchone()
conn.close()
del os.environ["PYNAPSE_DB_PATH"]
print("engine.py changes verified.")
```

---

### 3.3 Authentication

**WHAT:** Set the `motherduck_token` environment variable so DuckDB can authenticate with MotherDuck.

**WHY:** When `duckdb.connect("md:pynapse")` is called, the DuckDB extension looks for the `motherduck_token` environment variable automatically. Without it, MotherDuck falls back to interactive browser-based authentication -- which does not work in headless environments (SSH sessions, CI, scripts).

**For bash** (`~/.bashrc` or `~/.bash_profile`):

```bash
export motherduck_token="your_token_here"
```

**For zsh** (`~/.zshrc`):

```zsh
export motherduck_token="your_token_here"
```

Then reload your shell:

```bash
source ~/.bashrc  # or ~/.zshrc
```

> **Warning:** Never hardcode your MotherDuck token in source code, commit it to git, or include it in `pyproject.toml`. Always use environment variables or a secrets manager.

**Alternative: per-session token**

If you prefer not to persist the token in your shell profile, you can set it inline:

```bash
motherduck_token="your_token_here" python my_script.py
```

Or include the token in the connection string (less secure, but useful for one-off testing):

```python
conn = duckdb.connect("md:pynapse?motherduck_token=your_token_here")
```

**VERIFY:**

```bash
# Confirm the variable is set
echo $motherduck_token
# Should print your token (not empty)
```

```python
# Confirm DuckDB can connect to MotherDuck
import duckdb
conn = duckdb.connect("md:")
print(conn.execute("SELECT current_database()").fetchone())
conn.close()
```

---

### 3.4 Version Bump

**WHAT:** Update the DuckDB version constraint in `pyproject.toml`.

**WHY:** MotherDuck requires DuckDB >= 1.4.0. The current constraint (`duckdb>=1.0.0`) allows versions that cannot connect to MotherDuck. Bumping the floor ensures all installs are MotherDuck-compatible.

**CURRENT** (`pyproject.toml`, line 27):

```toml
"duckdb>=1.0.0",
```

**REPLACEMENT:**

```toml
"duckdb>=1.4.0",
```

After changing the file, reinstall to validate:

```bash
pip install -e .
python -c "import duckdb; assert tuple(int(x) for x in duckdb.__version__.split('.')[:2]) >= (1, 4), 'DuckDB too old'; print(f'OK: duckdb {duckdb.__version__}')"
```

**VERIFY:**

```bash
pip show duckdb | grep Version
# Should show 1.4.0 or later
```

---

### 3.5 Database Upload

**WHAT:** Upload the local `pynapse.duckdb` file to MotherDuck as a cloud database.

**WHY:** This is the core migration step. After uploading, MotherDuck hosts a copy of your database that any authenticated user can query remotely.

> **Warning:** This creates a new cloud database from your local file. It does **not** delete or modify the local file. However, from this point forward, the cloud database and local file will diverge unless you take steps to keep them in sync.

**Step 1: Connect to MotherDuck**

```python
import duckdb

# Connect to MotherDuck (uses motherduck_token env var)
conn = duckdb.connect("md:")
```

**Step 2: Upload the local database**

```python
# Upload from local file to MotherDuck
conn.execute("""
    CREATE OR REPLACE DATABASE pynapse
    FROM '/home/YOUR_USERNAME/.pynapse/pynapse.duckdb'
""")
print("Upload complete.")
```

> **Note:** Replace `/home/YOUR_USERNAME/.pynapse/pynapse.duckdb` with the actual path to your local database file. You can find it with: `python -c "from pathlib import Path; print(Path.home() / '.pynapse' / 'pynapse.duckdb')"`

**Step 3: Verify the upload**

```python
# Switch to the uploaded database
conn.execute("USE pynapse")

# Check schema version
version = conn.execute(
    "SELECT value FROM _meta WHERE key = 'schema_version'"
).fetchone()
print(f"Schema version: {version[0]}")

# Check table row counts
for table in ["projects", "populations", "subjects", "fovs",
              "neural_traces", "frame_timestamps", "events", "neurons"]:
    count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    print(f"  {table}: {count} rows")

conn.close()
```

**VERIFY:** The schema version and row counts should exactly match your local database. Cross-check with:

```python
import duckdb

local = duckdb.connect(str(Path.home() / ".pynapse" / "pynapse.duckdb"))
for table in ["projects", "populations", "subjects", "fovs",
              "neural_traces", "frame_timestamps", "events", "neurons"]:
    count = local.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    print(f"  local {table}: {count} rows")
local.close()
```

---

### 3.6 Raw File Storage

**WHAT:** Decide how to handle the local `~/.pynapse/raw/` directory, which contains copies of source files (`.mat`, `.csv`, `.npy`).

**WHY:** The `raw_files` table stores metadata (filename, SHA-256, size) and a `stored_path` column containing **absolute local filesystem paths** (e.g., `/home/user/.pynapse/raw/3/signals.npy`). These paths are written by `ingest._copy_raw_file()` (line 105 in `ingest.py`). MotherDuck cannot serve these files -- they exist only on the machine that ran the ingestion.

**Option A: Shared filesystem (recommended for lab use)**

If all lab members mount the same network drive (e.g., NFS, SMB), keep `raw/` on that shared mount:

```bash
# Move raw files to shared storage
mv ~/.pynapse/raw /mnt/lab-share/pynapse-raw

# Update stored_path entries to match (run against MotherDuck)
import duckdb
conn = duckdb.connect("md:pynapse")
conn.execute("""
    UPDATE raw_files
    SET stored_path = REPLACE(
        stored_path,
        '/home/YOUR_USERNAME/.pynapse/raw',
        '/mnt/lab-share/pynapse-raw'
    )
""")
conn.close()
```

**Option B: Skip raw file copies**

If raw file provenance is tracked elsewhere (e.g., a lab data server), you can skip copying during ingestion:

```python
fov_id = db.ingest.fov(
    # ... other arguments ...
    copy_raw=False,  # Do not copy source files to ~/.pynapse/raw/
)
```

The `copy_raw=False` parameter is already supported by `ingest.fov()` (line 278), `ingest.from_sample()` (line 163), `ingest.from_population()` (line 324), and `ingest.from_project()` (line 351).

**Option C: Leave as-is**

The raw files remain on the original machine. The `stored_path` values in the database will be stale for other users but the core pipeline (traces, events, timestamps) is unaffected since that data is stored as BLOBs directly in the database tables.

> **Note:** The raw file directory is a convenience feature for audit/provenance. All data needed for analysis (neural traces, frame timestamps, events) is stored directly in the database as BLOBs and structured rows. The pipeline works fully without raw file access.

---

### 3.7 Pipeline & Query Updates

**WHAT:** Demonstrate that ingestion and querying work with a MotherDuck connection.

**WHY:** Virtually no code changes are needed beyond the connection string. This section shows the updated workflow.

**Ingestion (after applying engine.py changes)**

```bash
# Set the env var to point at MotherDuck
export PYNAPSE_DB_PATH="md:pynapse"
```

```python
from pynapse import db

# Engine reads PYNAPSE_DB_PATH automatically
conn = db.connect()

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
    copy_raw=False,  # Recommended for cloud -- see section 3.6
)

print(f"Ingested FOV: {fov_id}")
conn.close()
```

**Querying**

```python
from pynapse import db

conn = db.connect()  # Uses PYNAPSE_DB_PATH="md:pynapse"

# All query functions work identically
projects = db.query.list_projects(conn=conn)
traces = db.query.get_traces(fov_id=1, conn=conn)
events = db.query.get_events(fov_id=1, conn=conn)
summary = db.query.get_event_summary(fov_id=1, conn=conn)

print(f"Traces shape: {traces.shape}")
print(summary)

conn.close()
```

**DBSample and SampleEventTensor**

```python
from pynapse import db

conn = db.connect()

# DBSample works identically -- lazy caching means each query runs once
sample = db.DBSample(fov_id=1, conn=conn)
print(sample)
# DBSample(fov_id=1, name=Mouse1_D1_FOV1, neurons=42, frames=9000)

# Tensor extraction uses the same API
tensor = sample.get_tensor(event_id=101, pre_event=2.0, post_event=5.0)
windows = tensor.get_event_windows()
print(f"Tensor shape: {windows.shape}")

conn.close()
```

> **Note:** `DBSample`'s lazy caching (in `hydrate.py`) naturally minimizes cloud round-trips. After the first call to `get_signals()`, `get_dataframe()`, or `_get_frame_timestamps()`, subsequent calls return the cached result without querying MotherDuck again.

---

### 3.8 Verification Checklist

**WHAT:** A consolidated script that verifies the full migration.

**WHY:** Run this after completing steps 3.1--3.7 to confirm everything works end-to-end.

```python
"""
Post-migration verification script.
Run with: PYNAPSE_DB_PATH="md:pynapse" python verify_migration.py
"""
import os
import sys

# Ensure we're pointing at MotherDuck
db_path = os.environ.get("PYNAPSE_DB_PATH", "")
if not db_path.startswith("md:"):
    print(f"ERROR: PYNAPSE_DB_PATH={db_path!r} -- expected 'md:...'")
    print("Set it with: export PYNAPSE_DB_PATH='md:pynapse'")
    sys.exit(1)

import duckdb
print(f"DuckDB version: {duckdb.__version__}")
assert tuple(int(x) for x in duckdb.__version__.split(".")[:2]) >= (1, 4), \
    f"DuckDB {duckdb.__version__} < 1.4.0 -- upgrade required"

from pynapse import db

conn = db.connect()

# 1. Schema version
version = conn.execute(
    "SELECT value FROM _meta WHERE key = 'schema_version'"
).fetchone()
assert version is not None, "Missing schema version"
print(f"[OK] Schema version: {version[0]}")

# 2. Row counts
tables = ["projects", "populations", "subjects", "fovs",
          "neural_traces", "frame_timestamps", "events", "neurons"]
for table in tables:
    count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    print(f"  {table}: {count} rows")

# 3. Trace shape for first FOV
fovs = db.query.list_fovs(conn=conn)
if not fovs.empty:
    fov_id = int(fovs.iloc[0]["id"])
    traces = db.query.get_traces(fov_id, conn=conn)
    fov_info = db.query.get_fov(fov_id=fov_id, conn=conn)
    expected_shape = (int(fov_info["num_neurons"]), int(fov_info["num_frames"]))
    assert traces.shape == expected_shape, \
        f"Trace shape mismatch: {traces.shape} != {expected_shape}"
    print(f"[OK] Trace shape for FOV {fov_id}: {traces.shape}")

    # 4. Events
    events = db.query.get_events(fov_id, conn=conn)
    print(f"[OK] Events for FOV {fov_id}: {len(events)} rows")

    # 5. DBSample
    sample = db.DBSample(fov_id, conn=conn)
    assert sample.num_neurons == traces.shape[0]
    assert sample.num_frames == traces.shape[1]
    print(f"[OK] DBSample: {sample}")

    # 6. SampleEventTensor (if events exist)
    if not events.empty:
        first_code = int(events.iloc[0]["code"])
        n_events = sample.get_num_events(first_code)
        if n_events > 0:
            tensor = sample.get_tensor(
                event_id=first_code, pre_event=1.0, post_event=2.0
            )
            print(f"[OK] SampleEventTensor created for code={first_code}")
else:
    print("[WARN] No FOVs found -- database may be empty")

conn.close()
print("\nAll checks passed.")
```

---

## 4. Hybrid Execution

### What is hybrid execution?

DuckDB + MotherDuck supports **hybrid execution**: a single connection can query both local and cloud databases in the same session. MotherDuck automatically decides where to execute each query based on which databases are referenced.

### How pynapse benefits

`DBSample` in `hydrate.py` is already a natural hybrid object. On first access:

1. Metadata (FOV row, paradigm) is fetched from MotherDuck -- small, fast
2. Traces and events are fetched from MotherDuck -- one-time, cached locally in Python
3. All subsequent access (tensor extraction, analysis) uses the in-memory Python cache

This means a typical analysis session sends only a few queries to MotherDuck, then operates entirely locally.

### Attaching a local scratch database

You can attach a local database alongside MotherDuck for temporary or experimental data:

```python
import duckdb

conn = duckdb.connect("md:pynapse")

# Attach a local scratch database
conn.execute("ATTACH '/tmp/scratch.duckdb' AS scratch")

# Query across both
conn.execute("""
    SELECT f.name, f.num_neurons
    FROM pynapse.fovs f
    WHERE f.id NOT IN (SELECT fov_id FROM scratch.processed_fovs)
""").fetchdf()
```

### Forcing local execution

Use the `MD_RUN` pragma to force a query to run locally or remotely:

```python
# Force local execution
conn.execute("PRAGMA MD_RUN='LOCAL'")
conn.execute("SELECT * FROM my_local_table").fetchdf()

# Force remote execution
conn.execute("PRAGMA MD_RUN='MOTHERDUCK'")
conn.execute("SELECT * FROM pynapse.fovs").fetchdf()

# Reset to automatic (default)
conn.execute("PRAGMA MD_RUN='AUTO'")
```

---

## 5. Rollback

If you need to revert to the local-only setup:

### Step 1: Unset the environment variable

```bash
unset PYNAPSE_DB_PATH
```

This causes `engine.py` to fall back to the default local path (`~/.pynapse/pynapse.duckdb`).

### Step 2: Restore the backup (if needed)

If you modified the local database file during migration:

```bash
# Find your backup
ls ~/.pynapse/pynapse.duckdb.bak.*

# Restore (replace TIMESTAMP with your actual backup timestamp)
cp ~/.pynapse/pynapse.duckdb.bak.TIMESTAMP ~/.pynapse/pynapse.duckdb
```

### Step 3: Revert engine.py (optional)

If you applied the `engine.py` changes from [3.2](#32-connection-configuration), they are backward-compatible. The `_is_remote_or_memory()` helper and `PYNAPSE_DB_PATH` env var do not affect local-only usage when the env var is unset. You can leave the changes in place.

### Step 4: Drop the MotherDuck database (optional)

If you want to remove the cloud copy entirely:

```python
import duckdb
conn = duckdb.connect("md:")
conn.execute("DROP DATABASE IF EXISTS pynapse")
conn.close()
```

### Step 5: Revert version constraint (optional)

If you need to support DuckDB < 1.4.0 again, change `pyproject.toml` line 27 back:

```toml
"duckdb>=1.0.0",
```

---

## 6. Post-Migration Considerations

### Multi-user access

MotherDuck supports up to **16 concurrent read replicas**. Multiple lab members can query the same database simultaneously without coordination. Write operations (ingestion) are serialized by MotherDuck automatically.

To grant access:

1. Add team members to your MotherDuck organization
2. Share the database with them via the MotherDuck UI
3. Each member sets their own `motherduck_token` env var

All users should use the same DuckDB version to avoid compatibility issues (see below).

### DuckDB version synchronization

MotherDuck is version-locked: a cloud database created with DuckDB 1.4.0 must be accessed with DuckDB 1.4.x. If one team member upgrades to 1.5.x while the cloud database is on 1.4.x, queries may fail.

**Recommendation:** Pin the exact DuckDB version in a development requirements file or use `duckdb>=1.4.0,<1.5.0` as the constraint during the transition period.

### Cost and storage monitoring

| Resource | Monitoring |
|----------|-----------|
| Storage | MotherDuck dashboard > Database > Storage |
| Compute | MotherDuck dashboard > Usage |
| Query count | MotherDuck dashboard > Query Log |

Typical pynapse storage: ~4 MB per FOV (traces + timestamps + events). A lab with 500 FOVs uses approximately 2 GB.

### BLOB size and timeout headroom

MotherDuck has a **55-second query timeout**. The largest BLOBs in pynapse are neural trace matrices. Typical sizes:

| FOV size | BLOB size | Estimated retrieval time |
|----------|-----------|------------------------|
| 50 neurons x 9,000 frames | ~1.7 MB | < 1 second |
| 200 neurons x 18,000 frames | ~13.7 MB | < 2 seconds |
| 500 neurons x 36,000 frames | ~68.7 MB | < 5 seconds |

MotherDuck's per-cell BLOB limit is 4 GB. Even extreme cases (1000 neurons x 100,000 frames = ~381 MB) are well within both the cell limit and the timeout window.

### Schema initialization on MotherDuck

The `schema.initialize()` function runs `CREATE TABLE IF NOT EXISTS` and `CREATE SEQUENCE IF NOT EXISTS` statements, which are fully idempotent. These execute on every connection via `get_connection()` and `connect()`. On MotherDuck, this adds a small overhead to the first connection (~200ms for the DDL checks) but is harmless.

---

## 7. Troubleshooting

| Problem | Cause | Solution |
|---------|-------|----------|
| `OSError: [Errno 2] No such file or directory` when connecting | `Path("md:pynapse").parent.mkdir()` fails because `engine.py` tries to create a directory for the `md:` string | Apply the `engine.py` changes from [3.2](#32-connection-configuration) -- the `_is_remote_or_memory()` guard skips `mkdir()` for remote URIs |
| `duckdb.ConnectionException: Failed to connect to MotherDuck` | Missing or invalid `motherduck_token` | Set the env var: `export motherduck_token="your_token"`. Verify with `echo $motherduck_token` |
| `duckdb.InvalidInputException: ... version mismatch` | DuckDB version does not match what MotherDuck expects | Upgrade: `pip install --upgrade duckdb>=1.4.0`. Check: `python -c "import duckdb; print(duckdb.__version__)"` |
| `duckdb.IOException: ... query timeout` | Query exceeded 55-second limit | Unlikely for typical pynapse data. If hit, check if a BLOB is unusually large: `SELECT fov_id, LENGTH(trace_data) FROM neural_traces ORDER BY LENGTH(trace_data) DESC LIMIT 5` |
| `FileNotFoundError` when accessing raw files | `stored_path` in `raw_files` table points to local paths that don't exist on the current machine | See [3.6](#36-raw-file-storage) -- use shared filesystem, update paths, or rely on BLOBs (raw files are not needed for analysis) |
| Schema appears empty after upload | Connected to default MotherDuck database instead of `pynapse` | Use `md:pynapse` (not `md:`) in your connection string, or run `USE pynapse` after connecting |
| `PYNAPSE_DB_PATH` env var is ignored | Using `db_path` argument directly in `get_connection()` or `connect()` | Explicit `db_path` arguments take priority over the env var by design. Remove the argument to use the env var. |
| Paradigm seeds missing after upload | `_seed_paradigms()` checks for existing rows, so it won't re-insert if the paradigm names already exist | Verify with: `SELECT name FROM paradigms`. If missing, the upload may have failed -- re-run [3.5](#35-database-upload) |
| Multiple users see stale data | MotherDuck read replicas have slight propagation delay | Writes are consistent. Reads may lag by a few seconds on other replicas. Call `conn.execute("CHECKPOINT")` to force sync if needed |

---

*This guide was written for pynapse v0.1.0 with DuckDB >= 1.4.0 and MotherDuck (2025). For questions or issues, open an issue on the pynapse repository.*
