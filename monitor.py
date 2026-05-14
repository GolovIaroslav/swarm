"""Rich live monitor — the dashboard the user watches while crew runs.

Layout (see ARCHITECTURE.md):
  header  — project name | model | uptime | ctx % | rpm
  tasks   — per-task status & duration
  stats   — tokens in/out, files written, searches, retries
  log     — tail of _logs/run.log (~20 lines)
  footer  — q/p/l/s shortcuts

Runs in the main thread alongside crew.kickoff() via rich.live.Live with
refresh_per_second=2. Reads everything from the Store + log file — pulls
nothing from the LLM directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from state import Store


class Monitor:
    def __init__(self, store: Store, log_path: Path, project_name: str, model_id: str):
        self.store = store
        self.log_path = log_path
        self.project_name = project_name
        self.model_id = model_id

    def render(self):
        """Build the rich.layout.Layout tree for the current frame."""
        raise NotImplementedError("session 3")

    def run(self) -> None:
        """Block on rich.live.Live until the crew finishes or user hits a key."""
        raise NotImplementedError("session 3")


def plain_stdout_monitor(store: Store, log_path: Path) -> None:
    """Fallback when `[ui].live_monitor = false`. Prints task transitions only."""
    raise NotImplementedError("session 3")
