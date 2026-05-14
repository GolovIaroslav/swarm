"""SQLite-backed checkpoint store for one project run.

One DB file per project at projects/<name>/_state.db. WAL mode for safety
during long runs. All other modules should go through Store — no raw SQL.

Schema:
  tasks(id, agent, status, started_at, finished_at, output_file, handoff,
        retry_count, error)
  shared_state(key, value)
  search_cache(query, results, ts)
  metrics(key, value)
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
  id TEXT PRIMARY KEY,
  agent TEXT NOT NULL,
  status TEXT NOT NULL,
  started_at INTEGER,
  finished_at INTEGER,
  output_file TEXT,
  handoff TEXT,
  retry_count INTEGER DEFAULT 0,
  error TEXT
);
CREATE TABLE IF NOT EXISTS shared_state (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS search_cache (
  query TEXT PRIMARY KEY,
  results TEXT NOT NULL,
  ts INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS metrics (
  key TEXT PRIMARY KEY,
  value REAL NOT NULL
);
"""


@dataclass
class TaskRow:
    id: str
    agent: str
    status: str                       # pending | running | done | failed
    started_at: Optional[int] = None
    finished_at: Optional[int] = None
    output_file: Optional[str] = None
    handoff: Optional[str] = None
    retry_count: int = 0
    error: Optional[str] = None


class Store:
    """Thin wrapper around sqlite3. Open once per run, close on exit."""

    def __init__(self, db_path: Path):
        self.path = db_path
        self.conn: Optional[sqlite3.Connection] = None

    def open(self) -> None:
        """Connect, enable WAL, ensure schema exists."""
        raise NotImplementedError("session 2")

    def close(self) -> None:
        raise NotImplementedError("session 2")

    # ----- tasks -------------------------------------------------------
    def upsert_task(self, row: TaskRow) -> None:
        raise NotImplementedError("session 2")

    def get_task(self, task_id: str) -> Optional[TaskRow]:
        raise NotImplementedError("session 2")

    def all_tasks(self) -> list[TaskRow]:
        raise NotImplementedError("session 2")

    def unfinished_tasks(self) -> list[TaskRow]:
        raise NotImplementedError("session 2")

    # ----- shared state ------------------------------------------------
    def get_state(self, key: str) -> Optional[str]:
        raise NotImplementedError("session 2")

    def set_state(self, key: str, value: str) -> None:
        raise NotImplementedError("session 2")

    # ----- search cache ------------------------------------------------
    def cache_search(self, query: str, results: str) -> None:
        raise NotImplementedError("session 2")

    def fetch_search(self, query: str, ttl_days: int) -> Optional[str]:
        raise NotImplementedError("session 2")

    # ----- metrics -----------------------------------------------------
    def bump_metric(self, key: str, delta: float = 1.0) -> None:
        raise NotImplementedError("session 2")

    def get_metric(self, key: str) -> float:
        raise NotImplementedError("session 2")


@contextmanager
def open_store(db_path: Path):
    s = Store(db_path)
    s.open()
    try:
        yield s
    finally:
        s.close()
