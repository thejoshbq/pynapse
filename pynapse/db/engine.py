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
