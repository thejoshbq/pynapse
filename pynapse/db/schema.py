# schema.py
# DDL, indexes, paradigm seeding, and schema versioning.

import json

SCHEMA_VERSION = "1"

_DDL_STATEMENTS = [
    # Meta
    """CREATE TABLE IF NOT EXISTS _meta (
        key TEXT PRIMARY KEY,
        value TEXT
    )""",
    # Sequences
    "CREATE SEQUENCE IF NOT EXISTS seq_projects START 1",
    "CREATE SEQUENCE IF NOT EXISTS seq_populations START 1",
    "CREATE SEQUENCE IF NOT EXISTS seq_subjects START 1",
    "CREATE SEQUENCE IF NOT EXISTS seq_paradigms START 1",
    "CREATE SEQUENCE IF NOT EXISTS seq_fovs START 1",
    "CREATE SEQUENCE IF NOT EXISTS seq_raw_files START 1",
    "CREATE SEQUENCE IF NOT EXISTS seq_events START 1",
    "CREATE SEQUENCE IF NOT EXISTS seq_neurons START 1",
    # Projects
    """CREATE TABLE IF NOT EXISTS projects (
        id INTEGER PRIMARY KEY DEFAULT nextval('seq_projects'),
        name TEXT UNIQUE NOT NULL,
        description TEXT,
        authors TEXT,
        created_at TIMESTAMP DEFAULT current_timestamp,
        updated_at TIMESTAMP DEFAULT current_timestamp
    )""",
    # Populations
    """CREATE TABLE IF NOT EXISTS populations (
        id INTEGER PRIMARY KEY DEFAULT nextval('seq_populations'),
        project_id INTEGER NOT NULL REFERENCES projects(id),
        name TEXT NOT NULL,
        description TEXT,
        created_at TIMESTAMP DEFAULT current_timestamp,
        UNIQUE(project_id, name)
    )""",
    # Subjects
    """CREATE TABLE IF NOT EXISTS subjects (
        id INTEGER PRIMARY KEY DEFAULT nextval('seq_subjects'),
        population_id INTEGER NOT NULL REFERENCES populations(id),
        name TEXT NOT NULL,
        sex TEXT,
        genotype TEXT,
        species TEXT DEFAULT 'mouse',
        date_of_birth DATE,
        notes TEXT,
        created_at TIMESTAMP DEFAULT current_timestamp,
        UNIQUE(population_id, name)
    )""",
    # Paradigms
    """CREATE TABLE IF NOT EXISTS paradigms (
        id INTEGER PRIMARY KEY DEFAULT nextval('seq_paradigms'),
        name TEXT UNIQUE NOT NULL,
        event_dict TEXT,
        color_dict TEXT,
        description TEXT
    )""",
    # Fields of view
    """CREATE TABLE IF NOT EXISTS fovs (
        id INTEGER PRIMARY KEY DEFAULT nextval('seq_fovs'),
        subject_id INTEGER NOT NULL REFERENCES subjects(id),
        paradigm_id INTEGER NOT NULL REFERENCES paradigms(id),
        name TEXT NOT NULL,
        fps DOUBLE NOT NULL,
        frame_averaging INTEGER NOT NULL DEFAULT 1,
        effective_fps DOUBLE GENERATED ALWAYS AS (fps / frame_averaging),
        interframe_interval_ms DOUBLE GENERATED ALWAYS AS (1000.0 / fps),
        frame_correction BOOLEAN DEFAULT FALSE,
        start_time_ms DOUBLE DEFAULT 0.0,
        num_neurons INTEGER NOT NULL,
        num_frames INTEGER NOT NULL,
        source_format TEXT,
        notes TEXT,
        recorded_at TIMESTAMP,
        ingested_at TIMESTAMP DEFAULT current_timestamp,
        UNIQUE(subject_id, name)
    )""",
    # Raw files
    """CREATE TABLE IF NOT EXISTS raw_files (
        id INTEGER PRIMARY KEY DEFAULT nextval('seq_raw_files'),
        fov_id INTEGER NOT NULL REFERENCES fovs(id),
        file_type TEXT NOT NULL,
        original_filename TEXT,
        stored_path TEXT,
        sha256 TEXT,
        file_size_bytes BIGINT,
        part_number INTEGER,
        ingested_at TIMESTAMP DEFAULT current_timestamp
    )""",
    # Neural traces (1:1 with fovs)
    """CREATE TABLE IF NOT EXISTS neural_traces (
        fov_id INTEGER PRIMARY KEY REFERENCES fovs(id),
        num_neurons INTEGER NOT NULL,
        num_frames INTEGER NOT NULL,
        trace_data BLOB NOT NULL,
        dtype TEXT NOT NULL DEFAULT 'float32'
    )""",
    # Frame timestamps (1:1 with fovs)
    """CREATE TABLE IF NOT EXISTS frame_timestamps (
        fov_id INTEGER PRIMARY KEY REFERENCES fovs(id),
        num_timestamps INTEGER NOT NULL,
        timestamps_ms BLOB NOT NULL,
        source TEXT NOT NULL
    )""",
    # Events
    """CREATE TABLE IF NOT EXISTS events (
        id INTEGER PRIMARY KEY DEFAULT nextval('seq_events'),
        fov_id INTEGER NOT NULL REFERENCES fovs(id),
        code INTEGER NOT NULL,
        label TEXT NOT NULL,
        t1_ms DOUBLE NOT NULL,
        t2_ms DOUBLE,
        frame_index INTEGER NOT NULL,
        end_frame_index INTEGER
    )""",
    # Neurons
    """CREATE TABLE IF NOT EXISTS neurons (
        id INTEGER PRIMARY KEY DEFAULT nextval('seq_neurons'),
        fov_id INTEGER NOT NULL REFERENCES fovs(id),
        neuron_index INTEGER NOT NULL,
        label TEXT,
        cell_type TEXT,
        region TEXT,
        x_um DOUBLE,
        y_um DOUBLE,
        UNIQUE(fov_id, neuron_index)
    )""",
]

_INDEX_STATEMENTS = [
    "CREATE INDEX IF NOT EXISTS idx_events_fov_code ON events(fov_id, code)",
    "CREATE INDEX IF NOT EXISTS idx_events_fov_label ON events(fov_id, label)",
    "CREATE INDEX IF NOT EXISTS idx_fovs_subject ON fovs(subject_id)",
    "CREATE INDEX IF NOT EXISTS idx_subjects_population ON subjects(population_id)",
    "CREATE INDEX IF NOT EXISTS idx_populations_project ON populations(project_id)",
    "CREATE INDEX IF NOT EXISTS idx_raw_files_fov ON raw_files(fov_id)",
    "CREATE INDEX IF NOT EXISTS idx_neurons_fov ON neurons(fov_id)",
]


def _seed_paradigms(conn):
    """Seed standard paradigms from pynapse.config.events if not present."""
    from pynapse.config.events import LEGACY_HER, LEGACY_ETH, REACHER, COLORS

    seeds = [
        ("legacy_her", LEGACY_HER, "Heroin head-fixed self-administration (2024)"),
        ("legacy_eth", LEGACY_ETH, "Ethanol head-fixed self-administration (2025)"),
        ("reacher", REACHER, "REACHER self-administration (2026+)"),
    ]
    color_json = json.dumps(COLORS)
    for name, event_dict, desc in seeds:
        exists = conn.execute(
            "SELECT 1 FROM paradigms WHERE name = ?", [name]
        ).fetchone()
        if not exists:
            ed_json = json.dumps({str(k): v for k, v in event_dict.items()})
            conn.execute(
                "INSERT INTO paradigms (name, event_dict, color_dict, description) "
                "VALUES (?, ?, ?, ?)",
                [name, ed_json, color_json, desc],
            )


def initialize(conn):
    """Create all tables, indexes, and seed paradigms. Idempotent."""
    for stmt in _DDL_STATEMENTS:
        conn.execute(stmt)
    for stmt in _INDEX_STATEMENTS:
        conn.execute(stmt)
    _seed_paradigms(conn)
    existing = conn.execute(
        "SELECT value FROM _meta WHERE key = 'schema_version'"
    ).fetchone()
    if not existing:
        conn.execute(
            "INSERT INTO _meta (key, value) VALUES ('schema_version', ?)",
            [SCHEMA_VERSION],
        )
