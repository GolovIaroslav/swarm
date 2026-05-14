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
        on_hang: Callable[[str], None],            # called with task_id
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
        raise NotImplementedError("session 3")

    def stop(self) -> None:
        """Signal the polling thread to exit and join it."""
        raise NotImplementedError("session 3")

    def _loop(self) -> None:
        """Polling loop — runs in background thread."""
        raise NotImplementedError("session 3")
