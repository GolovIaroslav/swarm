#!/usr/bin/env python3
"""swarm — entry point.

Boot order:
  1. config.load()
  2. backend.start() (spawns llama-server if configured)
  3. TUI setup screen — questionary, with resume detection
  4. Pre-flight — ping LLM, JSON-capability test, web-search test
  5. Build crew from presets/custom + Store + tools
  6. Start Monitor + Watchdog
  7. crew.kickoff(); checkpoint after every task
  8. On exit (clean or signal): stop watchdog, close store, backend.stop()

Ctrl+C anywhere is safe — SIGINT handler flushes the store and exits.
"""

from __future__ import annotations

import signal
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    raise NotImplementedError("session 3")


def setup_screen():
    """Questionary flow — project name, preset, description, agents, process.
    Detects existing checkpoints and offers Resume/Start over/Delete."""
    raise NotImplementedError("session 3")


def preflight(backend, cfg) -> bool:
    """Ping LLM, JSON test (warn if hierarchical risky), web-search test.
    Returns True on user confirmation to proceed."""
    raise NotImplementedError("session 3")


def install_signal_handlers(on_exit) -> None:
    raise NotImplementedError("session 3")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
