# test_db.py
# Unit tests for the pynapse database layer.

import json
import pytest
import numpy as np
import pandas as pd
import duckdb

from pynapse.db import schema, engine
from pynapse.db import ingest, query
from pynapse.db.hydrate import DBSample
from pynapse.config.events import LEGACY_HER, LEGACY_ETH, REACHER, COLORS, TASK_TO_DICT


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def conn():
    """Fresh in-memory DuckDB with initialized schema."""
    c = duckdb.connect(":memory:")
    schema.initialize(c)
    return c


@pytest.fixture
def sample_in_db(conn, mock_data_files):
    """Ingest a mock legacy Sample and return (fov_id, conn)."""
    event_file, signal_file, event_dict = mock_data_files

    from pynapse.core.sample import Sample
    sample = Sample(
        event_data=event_file,
        signal_data=signal_file,
        name="TestFOV1",
        event_dict=event_dict,
        fps=30.0,
        frame_averaging=1,
    )

    project_id = ingest._get_or_create_project("TestProject", conn)
    pop_id = ingest._get_or_create_population("TestPop", project_id, conn)
    subject_id = ingest._get_or_create_subject("TestMouse", pop_id, conn)
    fov_id = ingest.from_sample(
        sample, subject_id, "legacy_her",
        fov_name="TestFOV1", conn=conn, copy_raw=False,
    )
    return fov_id, conn, sample


@pytest.fixture
def reacher_in_db(conn, mock_reacher_data_files):
    """Ingest a mock REACHER Sample and return (fov_id, conn)."""
    behavior_csv, frame_csv, signal_file = mock_reacher_data_files

    from pynapse.core.sample import Sample
    from pynapse.config.events import REACHER

    sample = Sample(
        event_data=behavior_csv,
        signal_data=signal_file,
        name="ReacherFOV1",
        event_dict=REACHER,
        fps=30.0,
        frame_averaging=1,
        frame_timestamps=frame_csv,
    )

    project_id = ingest._get_or_create_project("ReacherProject", conn)
    pop_id = ingest._get_or_create_population("ReacherPop", project_id, conn)
    subject_id = ingest._get_or_create_subject("ReacherMouse", pop_id, conn)
    fov_id = ingest.from_sample(
        sample, subject_id, "reacher",
        fov_name="ReacherFOV1", conn=conn, copy_raw=False,
    )
    return fov_id, conn, sample


# ===================================================================
# 1. Schema Tests
# ===================================================================

class TestSchema:
    def test_creation_idempotent(self, conn):
        """Calling initialize twice does not raise."""
        schema.initialize(conn)
        schema.initialize(conn)

    def test_all_tables_exist(self, conn):
        tables = conn.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'main' ORDER BY table_name"
        ).fetchdf()
        expected = {
            "_meta", "events", "fovs", "frame_timestamps", "neural_traces",
            "neurons", "paradigms", "populations", "projects", "raw_files",
            "subjects",
        }
        assert expected == set(tables["table_name"].values)

    def test_schema_version(self, conn):
        row = conn.execute(
            "SELECT value FROM _meta WHERE key = 'schema_version'"
        ).fetchone()
        assert row[0] == schema.SCHEMA_VERSION

    def test_paradigm_seeding_legacy_her(self, conn):
        row = conn.execute(
            "SELECT event_dict FROM paradigms WHERE name = 'legacy_her'"
        ).fetchone()
        assert row is not None
        ed = json.loads(row[0])
        for code, label in LEGACY_HER.items():
            assert ed[str(code)] == label

    def test_paradigm_seeding_legacy_eth(self, conn):
        row = conn.execute(
            "SELECT event_dict FROM paradigms WHERE name = 'legacy_eth'"
        ).fetchone()
        assert row is not None
        ed = json.loads(row[0])
        for code, label in LEGACY_ETH.items():
            assert ed[str(code)] == label

    def test_paradigm_seeding_reacher(self, conn):
        row = conn.execute(
            "SELECT event_dict FROM paradigms WHERE name = 'reacher'"
        ).fetchone()
        assert row is not None
        ed = json.loads(row[0])
        for code, label in REACHER.items():
            assert ed[str(code)] == label

    def test_paradigm_color_dict(self, conn):
        row = conn.execute(
            "SELECT color_dict FROM paradigms WHERE name = 'reacher'"
        ).fetchone()
        cd = json.loads(row[0])
        for label, color in COLORS.items():
            assert cd[label] == color


# ===================================================================
# 2. Ingestion Tests
# ===================================================================

class TestIngestion:
    def test_from_sample_returns_int(self, sample_in_db):
        fov_id, conn, _ = sample_in_db
        assert isinstance(fov_id, int)

    def test_fov_row_exists(self, sample_in_db):
        fov_id, conn, sample = sample_in_db
        fov = query.get_fov(fov_id=fov_id, conn=conn)
        assert fov is not None
        assert fov["name"] == "TestFOV1"
        assert fov["fps"] == 30.0
        assert fov["num_neurons"] == sample.num_neurons
        assert fov["num_frames"] == sample.num_frames

    def test_generated_columns(self, sample_in_db):
        fov_id, conn, _ = sample_in_db
        fov = query.get_fov(fov_id=fov_id, conn=conn)
        assert fov["effective_fps"] == pytest.approx(30.0 / 1)
        assert fov["interframe_interval_ms"] == pytest.approx(1000.0 / 30.0)

    def test_neural_traces_stored(self, sample_in_db):
        fov_id, conn, sample = sample_in_db
        traces = query.get_traces(fov_id, conn=conn)
        expected = sample.get_signals().astype(np.float32)
        assert traces.shape == expected.shape
        np.testing.assert_array_almost_equal(traces, expected)

    def test_frame_timestamps_stored(self, sample_in_db):
        fov_id, conn, sample = sample_in_db
        ts = query.get_frame_timestamps(fov_id, conn=conn)
        expected = sample._get_frame_timestamps()
        assert len(ts) == len(expected)
        np.testing.assert_array_almost_equal(ts, expected)

    def test_events_stored(self, sample_in_db):
        fov_id, conn, sample = sample_in_db
        df_db = query.get_events(fov_id, conn=conn)
        df_orig = sample.get_dataframe()
        assert len(df_db) == len(df_orig)
        assert set(df_db.columns) == {"code", "t1", "t2", "label", "frame_index"}

    def test_event_counts_match(self, sample_in_db):
        fov_id, conn, sample = sample_in_db
        summary = query.get_event_summary(fov_id, conn=conn)
        for _, row in summary.iterrows():
            code = int(row["code"])
            expected_count = sample.count_events(code)
            assert int(row["count"]) == expected_count

    def test_neurons_created(self, sample_in_db):
        fov_id, conn, sample = sample_in_db
        count = conn.execute(
            "SELECT COUNT(*) FROM neurons WHERE fov_id = ?", [fov_id]
        ).fetchone()[0]
        assert count == sample.num_neurons

    def test_hierarchy_upsert(self, conn):
        """get-or-create returns the same id on second call."""
        pid1 = ingest._get_or_create_project("P1", conn)
        pid2 = ingest._get_or_create_project("P1", conn)
        assert pid1 == pid2

        pop1 = ingest._get_or_create_population("Pop1", pid1, conn)
        pop2 = ingest._get_or_create_population("Pop1", pid1, conn)
        assert pop1 == pop2

        sid1 = ingest._get_or_create_subject("S1", pop1, conn)
        sid2 = ingest._get_or_create_subject("S1", pop1, conn)
        assert sid1 == sid2

    def test_reacher_ingest(self, reacher_in_db):
        fov_id, conn, sample = reacher_in_db
        fov = query.get_fov(fov_id=fov_id, conn=conn)
        assert fov["source_format"] == "reacher_csv"
        traces = query.get_traces(fov_id, conn=conn)
        assert traces.shape == (sample.num_neurons, sample.num_frames)


# ===================================================================
# 3. Query Tests
# ===================================================================

class TestQuery:
    def test_list_projects(self, sample_in_db):
        _, conn, _ = sample_in_db
        df = query.list_projects(conn=conn)
        assert len(df) >= 1
        assert "TestProject" in df["name"].values

    def test_list_populations(self, sample_in_db):
        _, conn, _ = sample_in_db
        df = query.list_populations(project_name="TestProject", conn=conn)
        assert len(df) >= 1
        assert "TestPop" in df["name"].values

    def test_list_subjects(self, sample_in_db):
        _, conn, _ = sample_in_db
        df = query.list_subjects(population_name="TestPop", conn=conn)
        assert len(df) >= 1
        assert "TestMouse" in df["name"].values

    def test_list_fovs(self, sample_in_db):
        fov_id, conn, _ = sample_in_db
        df = query.list_fovs(conn=conn)
        assert fov_id in df["id"].values

    def test_get_events_by_label(self, sample_in_db):
        fov_id, conn, sample = sample_in_db
        df = query.get_events(fov_id, label="active_lever", conn=conn)
        expected = sample.count_events("active_lever")
        assert len(df) == expected

    def test_get_event_frame_indices(self, sample_in_db):
        fov_id, conn, _ = sample_in_db
        indices = query.get_event_frame_indices(fov_id, conn=conn)
        assert isinstance(indices, np.ndarray)

    def test_get_event_summary(self, sample_in_db):
        fov_id, conn, _ = sample_in_db
        summary = query.get_event_summary(fov_id, conn=conn)
        assert "code" in summary.columns
        assert "label" in summary.columns
        assert "count" in summary.columns


# ===================================================================
# 4. Hydration (DBSample) Tests
# ===================================================================

class TestDBSample:
    def test_properties(self, sample_in_db):
        fov_id, conn, sample = sample_in_db
        db = DBSample(fov_id, conn=conn)
        assert db.name == "TestFOV1"
        assert db.fps == sample.fps
        assert db.frame_averaging == sample.frame_averaging
        assert db.effective_fps == pytest.approx(sample.effective_fps)
        assert db.interframe_interval == pytest.approx(sample.interframe_interval)
        assert db.num_neurons == sample.num_neurons
        assert db.num_frames == sample.num_frames

    def test_get_signals(self, sample_in_db):
        fov_id, conn, sample = sample_in_db
        db = DBSample(fov_id, conn=conn)
        signals = db.get_signals()
        expected = sample.get_signals().astype(np.float32)
        assert signals.shape == expected.shape
        np.testing.assert_array_almost_equal(signals, expected)

    def test_get_dataframe(self, sample_in_db):
        fov_id, conn, sample = sample_in_db
        db = DBSample(fov_id, conn=conn)
        df = db.get_dataframe()
        assert set(df.columns) == {"code", "t1", "t2", "label", "frame_index"}
        assert len(df) == len(sample.get_dataframe())

    def test_get_frame_timestamps(self, sample_in_db):
        fov_id, conn, sample = sample_in_db
        db = DBSample(fov_id, conn=conn)
        ts = db._get_frame_timestamps()
        expected = sample._get_frame_timestamps()
        np.testing.assert_array_almost_equal(ts, expected)

    def test_get_num_events(self, sample_in_db):
        fov_id, conn, sample = sample_in_db
        db = DBSample(fov_id, conn=conn)
        for code in [22, 21, 7, 4]:
            assert db.get_num_events(code) == sample.get_num_events(code)

    def test_count_events(self, sample_in_db):
        fov_id, conn, sample = sample_in_db
        db = DBSample(fov_id, conn=conn)
        assert db.count_events() == sample.count_events()

    def test_get_event_dict(self, sample_in_db):
        fov_id, conn, _ = sample_in_db
        db = DBSample(fov_id, conn=conn)
        ed = db.get_event_dict()
        for code, label in LEGACY_HER.items():
            assert ed[code] == label

    def test_tensor_extraction(self, sample_in_db):
        """DBSample plugs into SampleEventTensor identically to Sample."""
        fov_id, conn, sample = sample_in_db
        from pynapse.analysis.peri_event import SampleEventTensor

        # Only test if there are enough events
        if sample.count_events(22) < 1:
            pytest.skip("No active_lever events in mock data")

        db = DBSample(fov_id, conn=conn)
        tensor = SampleEventTensor(
            sample=db, event_id=22, pre_event=0.1, post_event=0.2,
        )
        windows = tensor.get_event_windows()
        assert windows.ndim == 3
        assert windows.shape[1] == sample.num_neurons

    def test_caching(self, sample_in_db):
        fov_id, conn, _ = sample_in_db
        db = DBSample(fov_id, conn=conn)
        s1 = db.get_signals()
        s2 = db.get_signals()
        assert s1 is s2  # same cached object


# ===================================================================
# 5. Convenience API (ingest.fov) Tests
# ===================================================================

class TestFovConvenience:
    def test_fov_function(self, conn, mock_data_files):
        event_file, signal_file, event_dict = mock_data_files
        fov_id = ingest.fov(
            neural=signal_file,
            events=event_file,
            sample_name="Mouse1",
            fov_name="Mouse1_FOV1",
            population_name="EarlyAcq",
            project_name="HeroinSA",
            paradigm_name="legacy_her",
            fps=30.0,
            conn=conn,
            copy_raw=False,
        )
        assert isinstance(fov_id, int)

        # Verify hierarchy was created
        projects = query.list_projects(conn=conn)
        assert "HeroinSA" in projects["name"].values

        pops = query.list_populations(project_name="HeroinSA", conn=conn)
        assert "EarlyAcq" in pops["name"].values

        subjects = query.list_subjects(population_name="EarlyAcq", conn=conn)
        assert "Mouse1" in subjects["name"].values

        fov = query.get_fov(fov_id=fov_id, conn=conn)
        assert fov["name"] == "Mouse1_FOV1"

    def test_fov_reacher(self, conn, mock_reacher_data_files):
        behavior_csv, frame_csv, signal_file = mock_reacher_data_files
        fov_id = ingest.fov(
            neural=signal_file,
            events=behavior_csv,
            sample_name="ReacherMouse1",
            fov_name="RM1_FOV1",
            population_name="TestPop",
            project_name="ReacherProj",
            paradigm_name="reacher",
            fps=30.0,
            frame_timestamps=frame_csv,
            conn=conn,
            copy_raw=False,
        )
        assert isinstance(fov_id, int)
        fov = query.get_fov(fov_id=fov_id, conn=conn)
        assert fov["source_format"] == "reacher_csv"
