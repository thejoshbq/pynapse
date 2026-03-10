# ingest.py
# Ingestion pipeline: Sample/Population/Project → DuckDB.

import hashlib
import json
import shutil
import warnings
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd

from . import engine


# ---------------------------------------------------------------------------
# Hierarchy helpers (get-or-create)
# ---------------------------------------------------------------------------

def _get_or_create_project(name, conn, description=None, authors=None):
    row = conn.execute("SELECT id FROM projects WHERE name = ?", [name]).fetchone()
    if row:
        return row[0]
    authors_json = json.dumps(authors) if authors else None
    conn.execute(
        "INSERT INTO projects (name, description, authors) VALUES (?, ?, ?)",
        [name, description, authors_json],
    )
    return conn.execute("SELECT id FROM projects WHERE name = ?", [name]).fetchone()[0]


def _get_or_create_population(name, project_id, conn, description=None):
    row = conn.execute(
        "SELECT id FROM populations WHERE project_id = ? AND name = ?",
        [project_id, name],
    ).fetchone()
    if row:
        return row[0]
    conn.execute(
        "INSERT INTO populations (project_id, name, description) VALUES (?, ?, ?)",
        [project_id, name, description],
    )
    return conn.execute(
        "SELECT id FROM populations WHERE project_id = ? AND name = ?",
        [project_id, name],
    ).fetchone()[0]


def _get_or_create_subject(name, population_id, conn, **kwargs):
    row = conn.execute(
        "SELECT id FROM subjects WHERE population_id = ? AND name = ?",
        [population_id, name],
    ).fetchone()
    if row:
        return row[0]
    sex = kwargs.get("sex")
    genotype = kwargs.get("genotype")
    species = kwargs.get("species", "mouse")
    notes = kwargs.get("notes")
    conn.execute(
        "INSERT INTO subjects (population_id, name, sex, genotype, species, notes) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        [population_id, name, sex, genotype, species, notes],
    )
    return conn.execute(
        "SELECT id FROM subjects WHERE population_id = ? AND name = ?",
        [population_id, name],
    ).fetchone()[0]


def _resolve_paradigm(name, conn):
    row = conn.execute("SELECT id FROM paradigms WHERE name = ?", [name]).fetchone()
    if not row:
        raise ValueError(f"Unknown paradigm '{name}'. Seed it first or use a known name.")
    return row[0]


# ---------------------------------------------------------------------------
# Raw file helpers
# ---------------------------------------------------------------------------

def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _copy_raw_file(src_path, fov_id, file_type, raw_dir, conn, part_number=None):
    src = Path(src_path)
    if not src.is_file():
        warnings.warn(f"Raw file not found, skipping: {src}")
        return
    dest_dir = raw_dir / str(fov_id)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / src.name
    shutil.copy2(str(src), str(dest))
    sha = _sha256(dest)
    size = dest.stat().st_size
    conn.execute(
        "INSERT INTO raw_files (fov_id, file_type, original_filename, stored_path, "
        "sha256, file_size_bytes, part_number) VALUES (?, ?, ?, ?, ?, ?, ?)",
        [fov_id, file_type, src.name, str(dest), sha, size, part_number],
    )


def _store_raw_files(sample, fov_id, raw_dir, conn):
    """Copy event/signal source files into the managed raw directory."""
    # Event sources
    evt_src = sample._event_log.source
    if isinstance(evt_src, (str, Path)):
        ext = str(evt_src).rsplit(".", 1)[-1].lower()
        ftype = "event_csv" if ext == "csv" else "event_mat"
        _copy_raw_file(evt_src, fov_id, ftype, raw_dir, conn)
    elif isinstance(evt_src, list):
        for i, s in enumerate(evt_src):
            ext = str(s).rsplit(".", 1)[-1].lower()
            ftype = "event_csv" if ext == "csv" else "event_mat"
            _copy_raw_file(s, fov_id, ftype, raw_dir, conn, part_number=i)

    # Signal sources
    sig_src = sample._signals.source
    if isinstance(sig_src, (str, Path)):
        _copy_raw_file(sig_src, fov_id, "signal_npy", raw_dir, conn)
    elif isinstance(sig_src, list):
        for i, s in enumerate(sig_src):
            _copy_raw_file(s, fov_id, "signal_npy", raw_dir, conn, part_number=i)


def _detect_source_format(sample):
    source = sample._event_log.source
    if isinstance(source, (str, Path)):
        if str(source).endswith(".csv"):
            return "reacher_csv"
        if str(source).endswith(".mat"):
            return "legacy_mat"
    elif isinstance(source, list) and source:
        if any(str(s).endswith(".mat") for s in source):
            return "legacy_mat"
        if any(str(s).endswith(".csv") for s in source):
            return "reacher_csv"
    return "unknown"


def _detect_frame_ts_source(sample):
    if sample._external_frame_ts is not None:
        return "external_csv"
    return "code_9"


# ---------------------------------------------------------------------------
# Core ingestion
# ---------------------------------------------------------------------------

def from_sample(
    sample,
    subject_id,
    paradigm_name,
    fov_name=None,
    conn=None,
    copy_raw=True,
    raw_dir=None,
):
    """Ingest a pre-constructed ``Sample`` object into the database.

    Returns the newly created ``fov_id``.
    """
    conn = conn or engine.get_connection()
    paradigm_id = _resolve_paradigm(paradigm_name, conn)

    # Extract aligned data from Sample (triggers full alignment pipeline)
    signals = sample.get_signals()
    df = sample.get_dataframe()
    frame_ts = sample._get_frame_timestamps()
    source_format = _detect_source_format(sample)
    ts_source = _detect_frame_ts_source(sample)
    name = fov_name or sample.name

    # 1. Insert FOV
    conn.execute(
        "INSERT INTO fovs (subject_id, paradigm_id, name, fps, frame_averaging, "
        "frame_correction, start_time_ms, num_neurons, num_frames, source_format) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            subject_id,
            paradigm_id,
            name,
            float(sample.fps),
            int(sample.frame_averaging),
            bool(getattr(sample, "_frame_correction", False)),
            float(getattr(sample, "_start_time", 0.0)),
            int(sample.num_neurons),
            int(sample.num_frames),
            source_format,
        ],
    )
    fov_id = conn.execute(
        "SELECT id FROM fovs WHERE subject_id = ? AND name = ?",
        [subject_id, name],
    ).fetchone()[0]

    # 2. Neural traces (BLOB)
    trace_blob = signals.astype(np.float32).tobytes()
    conn.execute(
        "INSERT INTO neural_traces (fov_id, num_neurons, num_frames, trace_data, dtype) "
        "VALUES (?, ?, ?, ?, ?)",
        [fov_id, int(signals.shape[0]), int(signals.shape[1]), trace_blob, "float32"],
    )

    # 3. Frame timestamps (BLOB)
    ts_blob = frame_ts.astype(np.float64).tobytes()
    conn.execute(
        "INSERT INTO frame_timestamps (fov_id, num_timestamps, timestamps_ms, source) "
        "VALUES (?, ?, ?, ?)",
        [fov_id, len(frame_ts), ts_blob, ts_source],
    )

    # 4. Events
    # Compute end_frame_index for events with valid t2
    end_frame_indices = np.full(len(df), np.nan)
    if "t2" in df.columns:
        t2_valid = df["t2"].notna()
        if t2_valid.any():
            t2_vals = df.loc[t2_valid, "t2"].values
            end_idx = np.searchsorted(frame_ts, t2_vals, side="right") - 1
            end_idx = np.clip(end_idx, 0, len(frame_ts) - 1)
            end_frame_indices[t2_valid.values] = end_idx

    event_rows = []
    for i, (_, row) in enumerate(df.iterrows()):
        t2 = None if pd.isna(row["t2"]) else float(row["t2"])
        efi = None if np.isnan(end_frame_indices[i]) else int(end_frame_indices[i])
        event_rows.append((
            fov_id, int(row["code"]), str(row["label"]),
            float(row["t1"]), t2,
            int(row["frame_index"]), efi,
        ))
    if event_rows:
        conn.executemany(
            "INSERT INTO events (fov_id, code, label, t1_ms, t2_ms, frame_index, "
            "end_frame_index) VALUES (?, ?, ?, ?, ?, ?, ?)",
            event_rows,
        )

    # 5. Neurons (placeholder rows)
    neuron_rows = [(fov_id, i) for i in range(sample.num_neurons)]
    if neuron_rows:
        conn.executemany(
            "INSERT INTO neurons (fov_id, neuron_index) VALUES (?, ?)",
            neuron_rows,
        )

    # 6. Raw file copies
    if copy_raw:
        rd = engine.get_raw_dir(raw_dir)
        _store_raw_files(sample, fov_id, rd, conn)

    return fov_id


def fov(
    neural: Union[str, Path, List],
    events: Union[str, Path, List],
    sample_name: str,
    fov_name: str,
    population_name: str,
    project_name: str,
    paradigm_name: str,
    fps: float,
    frame_averaging: int = 1,
    frame_timestamps: Union[str, Path, np.ndarray, None] = None,
    frame_correction: bool = False,
    correction_file: Union[str, Path, None] = None,
    start_time: float = 0.0,
    conn=None,
    copy_raw: bool = True,
    raw_dir=None,
    project_description: Optional[str] = None,
    population_description: Optional[str] = None,
):
    """Ingest a single FOV from raw file paths.

    Creates Sample internally, upserts hierarchy, and stores everything.
    Returns the ``fov_id``.
    """
    from pynapse.config.events import TASK_TO_DICT
    from pynapse.core.sample import Sample

    conn = conn or engine.get_connection()

    event_dict = TASK_TO_DICT.get(paradigm_name)

    sample = Sample(
        event_data=events,
        signal_data=neural,
        name=fov_name,
        event_dict=event_dict,
        fps=fps,
        frame_averaging=frame_averaging,
        frame_correction=frame_correction,
        correction_file=correction_file,
        start_time=start_time,
        frame_timestamps=frame_timestamps,
    )

    project_id = _get_or_create_project(project_name, conn, description=project_description)
    pop_id = _get_or_create_population(population_name, project_id, conn,
                                       description=population_description)
    subject_id = _get_or_create_subject(sample_name, pop_id, conn)

    return from_sample(
        sample, subject_id, paradigm_name, fov_name=fov_name,
        conn=conn, copy_raw=copy_raw, raw_dir=raw_dir,
    )


def from_population(
    population,
    project_id,
    paradigm_name,
    conn=None,
    copy_raw=True,
    raw_dir=None,
):
    """Ingest all samples in a ``Population``.

    Each sample's ``.name`` is used as both the subject name and FOV name.
    Returns a list of ``fov_id`` values.
    """
    conn = conn or engine.get_connection()
    pop_id = _get_or_create_population(
        population.name, project_id, conn, description=population.description,
    )
    fov_ids = []
    for sample in population.get_samples():
        subject_id = _get_or_create_subject(sample.name, pop_id, conn)
        fov_id = from_sample(
            sample, subject_id, paradigm_name,
            conn=conn, copy_raw=copy_raw, raw_dir=raw_dir,
        )
        fov_ids.append(fov_id)
    return fov_ids


def from_project(
    project,
    paradigm_name,
    conn=None,
    copy_raw=True,
    raw_dir=None,
):
    """Ingest all populations in a ``Project``.

    Returns a flat list of all ``fov_id`` values created.
    """
    conn = conn or engine.get_connection()
    project_id = _get_or_create_project(
        project.name, conn,
        description=project.description,
        authors=project.get_authors(),
    )
    all_fov_ids = []
    for population in project.get_populations():
        fov_ids = from_population(
            population, project_id, paradigm_name,
            conn=conn, copy_raw=copy_raw, raw_dir=raw_dir,
        )
        all_fov_ids.extend(fov_ids)
    return all_fov_ids
