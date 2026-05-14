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
import threading
import time
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
        self._lock = threading.RLock()

    def open(self) -> None:
        """Connect, enable WAL, ensure schema exists."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            self.conn = sqlite3.connect(str(self.path), check_same_thread=False)
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.execute("PRAGMA synchronous=NORMAL")
            for stmt in SCHEMA.strip().split(";"):
                stmt = stmt.strip()
                if stmt:
                    self.conn.execute(stmt)
            self.conn.commit()

    def close(self) -> None:
        with self._lock:
            if self.conn:
                self.conn.close()
                self.conn = None

    # ----- tasks -------------------------------------------------------
    def upsert_task(self, row: TaskRow) -> None:
        with self._lock:
            if self.conn is None:
                return
            self.conn.execute(
                """INSERT INTO tasks
                   (id, agent, status, started_at, finished_at, output_file,
                    handoff, retry_count, error)
                   VALUES (?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(id) DO UPDATE SET
                     agent=excluded.agent, status=excluded.status,
                     started_at=excluded.started_at, finished_at=excluded.finished_at,
                     output_file=excluded.output_file, handoff=excluded.handoff,
                     retry_count=excluded.retry_count, error=excluded.error""",
                (row.id, row.agent, row.status, row.started_at, row.finished_at,
                 row.output_file, row.handoff, row.retry_count, row.error),
            )
            self.conn.commit()

    def get_task(self, task_id: str) -> Optional[TaskRow]:
        with self._lock:
            if self.conn is None:
                return None
            cur = self.conn.execute(
                "SELECT id,agent,status,started_at,finished_at,output_file,"
                "handoff,retry_count,error FROM tasks WHERE id=?",
                (task_id,),
            )
            row = cur.fetchone()
            return TaskRow(*row) if row else None

    def all_tasks(self) -> list[TaskRow]:
        with self._lock:
            if self.conn is None:
                return []
            cur = self.conn.execute(
                "SELECT id,agent,status,started_at,finished_at,output_file,"
                "handoff,retry_count,error FROM tasks ORDER BY rowid"
            )
            return [TaskRow(*r) for r in cur.fetchall()]

    def unfinished_tasks(self) -> list[TaskRow]:
        with self._lock:
            if self.conn is None:
                return []
            cur = self.conn.execute(
                "SELECT id,agent,status,started_at,finished_at,output_file,"
                "handoff,retry_count,error FROM tasks "
                "WHERE status NOT IN ('done') ORDER BY rowid"
            )
            return [TaskRow(*r) for r in cur.fetchall()]

    # ----- shared state ------------------------------------------------
    def get_state(self, key: str) -> Optional[str]:
        with self._lock:
            if self.conn is None:
                return None
            cur = self.conn.execute(
                "SELECT value FROM shared_state WHERE key=?", (key,)
            )
            row = cur.fetchone()
            return row[0] if row else None

    def set_state(self, key: str, value: str) -> None:
        with self._lock:
            if self.conn is None:
                return
            self.conn.execute(
                "INSERT INTO shared_state(key,value) VALUES(?,?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, value),
            )
            self.conn.commit()

    # ----- search cache ------------------------------------------------
    def cache_search(self, query: str, results: str) -> None:
        with self._lock:
            if self.conn is None:
                return
            self.conn.execute(
                "INSERT INTO search_cache(query,results,ts) VALUES(?,?,?) "
                "ON CONFLICT(query) DO UPDATE SET results=excluded.results, ts=excluded.ts",
                (query, results, int(time.time())),
            )
            self.conn.commit()

    def fetch_search(self, query: str, ttl_days: int) -> Optional[str]:
        with self._lock:
            if self.conn is None:
                return None
            cutoff = int(time.time()) - ttl_days * 86400
            cur = self.conn.execute(
                "SELECT results FROM search_cache WHERE query=? AND ts > ?",
                (query, cutoff),
            )
            row = cur.fetchone()
            return row[0] if row else None

    # ----- metrics -----------------------------------------------------
    def bump_metric(self, key: str, delta: float = 1.0) -> None:
        with self._lock:
            if self.conn is None:
                return
            self.conn.execute(
                "INSERT INTO metrics(key,value) VALUES(?,?) "
                "ON CONFLICT(key) DO UPDATE SET value=value+excluded.value",
                (key, float(delta)),
            )
            self.conn.commit()

    def get_metric(self, key: str) -> float:
        with self._lock:
            if self.conn is None:
                return 0.0
            cur = self.conn.execute(
                "SELECT value FROM metrics WHERE key=?", (key,)
            )
            row = cur.fetchone()
            return row[0] if row else 0.0


@contextmanager
def open_store(db_path: Path):
    s = Store(db_path)
    s.open()
    try:
        yield s
    finally:
        s.close()
