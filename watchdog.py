"""Three-tier hang detector.

Tier 0 — mechanical. Background thread, polls every 60s. If the mtime of
         _logs/run.log hasn't moved in `task_timeout_minutes`, the active
         agent is presumed hung.

Tier 1 — auto-retry. Kill the task, increment retry_count, re-run with the
         same context (not from scratch — models usually fail mid-task).
         Up to cfg.execution.max_retry attempts.

Tier 2 — escalate. Retries exhausted. Pause everything and prompt:
         Continue retrying / Skip task / Abort & save.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Callable, Optional

from config import Config
from state import Store


class Watchdog:
    def __init__(
        self,
        log_path: Path,
        store: Store,
        cfg: Config,
        on_hang: Callable[[str], None],            # called with task_id (Tier 1 signal)
        on_exhausted: Callable[[str], str],        # task_id -> "retry"|"skip"|"abort"
    ):
        self.log_path = log_path
        self.store = store
        self.cfg = cfg
        self.on_hang = on_hang
        self.on_exhausted = on_exhausted
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    def start(self) -> None:
        """Spawn the polling thread (daemon)."""
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="watchdog"
        )
        self._thread.start()

    def stop(self) -> None:
        """Signal the polling thread to exit and join it."""
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None

    def _loop(self) -> None:
        """Polling loop — runs in background thread.

        Checks every 60 s. If the log file is stale for task_timeout_minutes,
        treats each running task as hung and applies Tier 1 / Tier 2 logic.
        """
        timeout_secs = self.cfg.execution.task_timeout_minutes * 60

        while not self._stop.wait(60):
            try:
                self._check(timeout_secs)
            except Exception:
                pass

    def _check(self, timeout_secs: float) -> None:
        running = [t for t in self.store.all_tasks() if t.status == "running"]
        if not running:
            return

        if not self.log_path.exists():
            return

        idle_secs = time.time() - self.log_path.stat().st_mtime
        if idle_secs <= timeout_secs:
            return

        for task in running:
            row = self.store.get_task(task.id)
            if row is None:
                continue

            if row.retry_count < self.cfg.execution.max_retry:
                row.retry_count += 1
                row.status = "pending"
                self.store.upsert_task(row)
                self.on_hang(task.id)
            else:
                action = self.on_exhausted(task.id)
                if action == "abort":
                    self._stop.set()
                    return
                elif action == "skip":
                    row.status = "failed"
                    row.error = "skipped after exhausted retries"
                    self.store.upsert_task(row)
