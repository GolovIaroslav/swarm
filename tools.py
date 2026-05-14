"""Tools exposed to agents.

Every tool is a thin wrapper around either disk I/O, a shell command, or the
SQLite store. They take strings, return strings — that's what CrewAI's BaseTool
contract expects. Side effects must be logged to _logs/run.log.

Provided tools:
  read_file(path)              -> str
  write_file(path, content)    -> str  (status message)
  run_command(cmd, timeout)    -> str  (stdout+stderr+rc, capped)
  get_state(key)               -> str
  set_state(key, value)        -> str
  web_search(query)            -> str  (markdown-ish list)
  fetch_url(url)               -> str  (text-extracted page)
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from config import Config
from state import Store


def make_tools(project_root: Path, store: Store, cfg: Config) -> list:
    """Build the full list of CrewAI tools, bound to this project's root + store.

    Tools must NEVER write outside `project_root`. Validate every path.
    """
    raise NotImplementedError("session 2")


# Individual tool factories — kept separate so agents.py can pick subsets.

def tool_read_file(project_root: Path):
    raise NotImplementedError("session 2")


def tool_write_file(project_root: Path):
    raise NotImplementedError("session 2")


def tool_run_command(project_root: Path):
    raise NotImplementedError("session 2")


def tool_get_state(store: Store):
    raise NotImplementedError("session 2")


def tool_set_state(store: Store):
    raise NotImplementedError("session 2")


def tool_web_search(store: Store, cfg: Config):
    """DuckDuckGo by default. Falls back to Tavily on CAPTCHA if a key is set.
    Hard-caps at cfg.tools.search_max_per_task per task, caches in SQLite for
    cfg.tools.search_cache_days days.
    """
    raise NotImplementedError("session 2")


def tool_fetch_url():
    raise NotImplementedError("session 2")
