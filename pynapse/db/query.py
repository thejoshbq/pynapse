# query.py
# Query functions for the pynapse database.

import numpy as np
import pandas as pd

from . import engine


# ---------------------------------------------------------------------------
# Listing helpers
# ---------------------------------------------------------------------------

def list_projects(conn=None):
    conn = conn or engine.get_connection()
    return conn.execute(
        "SELECT id, name, description, authors, created_at FROM projects ORDER BY id"
    ).fetchdf()


def list_populations(project_name=None, project_id=None, conn=None):
    conn = conn or engine.get_connection()
    if project_name:
        return conn.execute(
            "SELECT p.id, p.name, p.description, p.project_id, p.created_at "
            "FROM populations p JOIN projects pr ON p.project_id = pr.id "
            "WHERE pr.name = ? ORDER BY p.id",
            [project_name],
        ).fetchdf()
    if project_id is not None:
        return conn.execute(
            "SELECT id, name, description, project_id, created_at "
            "FROM populations WHERE project_id = ? ORDER BY id",
            [project_id],
        ).fetchdf()
    return conn.execute(
        "SELECT id, name, description, project_id, created_at FROM populations ORDER BY id"
    ).fetchdf()


def list_subjects(population_name=None, population_id=None, conn=None):
    conn = conn or engine.get_connection()
    if population_name:
        return conn.execute(
            "SELECT s.id, s.name, s.sex, s.genotype, s.species, s.population_id "
            "FROM subjects s JOIN populations p ON s.population_id = p.id "
            "WHERE p.name = ? ORDER BY s.id",
            [population_name],
        ).fetchdf()
    if population_id is not None:
        return conn.execute(
            "SELECT id, name, sex, genotype, species, population_id "
            "FROM subjects WHERE population_id = ? ORDER BY id",
            [population_id],
        ).fetchdf()
    return conn.execute(
        "SELECT id, name, sex, genotype, species, population_id FROM subjects ORDER BY id"
    ).fetchdf()


def list_fovs(subject_id=None, conn=None):
    conn = conn or engine.get_connection()
    if subject_id is not None:
        return conn.execute(
            "SELECT id, name, subject_id, paradigm_id, fps, frame_averaging, "
            "effective_fps, num_neurons, num_frames, source_format, ingested_at "
            "FROM fovs WHERE subject_id = ? ORDER BY id",
            [subject_id],
        ).fetchdf()
    return conn.execute(
        "SELECT id, name, subject_id, paradigm_id, fps, frame_averaging, "
        "effective_fps, num_neurons, num_frames, source_format, ingested_at "
        "FROM fovs ORDER BY id"
    ).fetchdf()


# ---------------------------------------------------------------------------
# Single-record getters
# ---------------------------------------------------------------------------

def get_fov(fov_id=None, name=None, conn=None):
    """Return a single FOV row as a dict. Lookup by id or name."""
    conn = conn or engine.get_connection()
    if fov_id is not None:
        df = conn.execute("SELECT * FROM fovs WHERE id = ?", [fov_id]).fetchdf()
    elif name is not None:
        df = conn.execute("SELECT * FROM fovs WHERE name = ?", [name]).fetchdf()
    else:
        raise ValueError("Provide fov_id or name")
    if df.empty:
        return None
    return df.iloc[0].to_dict()


# ---------------------------------------------------------------------------
# Data retrieval
# ---------------------------------------------------------------------------

def get_traces(fov_id, conn=None):
    """Load the neural trace matrix for a FOV.

    Returns an ``(num_neurons, num_frames)`` float32 ndarray.
    """
    conn = conn or engine.get_connection()
    row = conn.execute(
        "SELECT trace_data, num_neurons, num_frames, dtype FROM neural_traces WHERE fov_id = ?",
        [fov_id],
    ).fetchone()
    if row is None:
        raise KeyError(f"No neural traces for fov_id={fov_id}")
    blob, n_neurons, n_frames, dtype = row
    return np.frombuffer(blob, dtype=dtype).reshape(n_neurons, n_frames).copy()


def get_frame_timestamps(fov_id, conn=None):
    """Load the 1-D frame timestamp array (ms) for a FOV."""
    conn = conn or engine.get_connection()
    row = conn.execute(
        "SELECT timestamps_ms, num_timestamps FROM frame_timestamps WHERE fov_id = ?",
        [fov_id],
    ).fetchone()
    if row is None:
        raise KeyError(f"No frame timestamps for fov_id={fov_id}")
    blob, n_ts = row
    return np.frombuffer(blob, dtype=np.float64).copy()


def get_events(fov_id, label=None, code=None, conn=None):
    """Return events for a FOV as a DataFrame.

    Column names match ``Sample.get_dataframe()``: code, t1, t2, label, frame_index.
    """
    conn = conn or engine.get_connection()
    base = (
        "SELECT code, t1_ms AS t1, t2_ms AS t2, label, frame_index "
        "FROM events WHERE fov_id = ?"
    )
    params = [fov_id]
    if label is not None:
        base += " AND label = ?"
        params.append(label)
    if code is not None:
        base += " AND code = ?"
        params.append(code)
    base += " ORDER BY t1"
    return conn.execute(base, params).fetchdf()


def get_event_summary(fov_id, conn=None):
    """Return a DataFrame with event counts grouped by label and code."""
    conn = conn or engine.get_connection()
    return conn.execute(
        "SELECT code, label, COUNT(*) AS count FROM events "
        "WHERE fov_id = ? GROUP BY code, label ORDER BY code",
        [fov_id],
    ).fetchdf()


def get_event_frame_indices(fov_id, label=None, code=None, conn=None):
    """Return frame indices for matching events as an integer array."""
    conn = conn or engine.get_connection()
    base = "SELECT frame_index FROM events WHERE fov_id = ?"
    params = [fov_id]
    if label is not None:
        base += " AND label = ?"
        params.append(label)
    if code is not None:
        base += " AND code = ?"
        params.append(code)
    base += " ORDER BY t1_ms"
    result = conn.execute(base, params).fetchnumpy()
    return result["frame_index"]
