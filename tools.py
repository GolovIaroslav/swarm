"""Tools exposed to agents.

Every tool is a thin wrapper around either disk I/O, a shell command, or the
SQLite store. They take strings, return strings — that's what CrewAI's BaseTool
contract expects.

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

import subprocess
import textwrap
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Type

if TYPE_CHECKING:
    from state import Store as _StoreType

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from config import Config
from state import Store


# ---------------------------------------------------------------------------
# Schema models for typed BaseTool subclasses
# ---------------------------------------------------------------------------

class _PathInput(BaseModel):
    path: str = Field(description="File path relative to project root")


class _WriteInput(BaseModel):
    path: str = Field(description="File path relative to project root")
    content: str = Field(description="Content to write")


class _CmdInput(BaseModel):
    cmd: str = Field(description="Shell command to run")
    timeout: int = Field(default=60, description="Timeout in seconds")


class _KeyInput(BaseModel):
    key: str = Field(description="State key")


class _KeyValueInput(BaseModel):
    key: str = Field(description="State key")
    value: str = Field(description="Value to store")


class _SearchInput(BaseModel):
    query: str = Field(description="Search query")


class _UrlInput(BaseModel):
    url: str = Field(description="URL to fetch")


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def _safe_path(project_root: Path, rel: str) -> Path:
    """Resolve rel inside project_root; raise if it escapes."""
    target = (project_root / rel).resolve()
    if not str(target).startswith(str(project_root.resolve())):
        raise ValueError(f"Path escapes project root: {rel}")
    return target


def tool_read_file(project_root: Path) -> BaseTool:
    root = project_root

    class ReadFileTool(BaseTool):
        name: str = "read_file"
        description: str = "Read a file by path (relative to project root). Returns its text content."
        args_schema: Type[BaseModel] = _PathInput

        def _run(self, path: str) -> str:
            try:
                target = _safe_path(root, path)
                return target.read_text(encoding="utf-8")
            except Exception as e:
                return f"ERROR: {e}"

    return ReadFileTool()


def tool_write_file(project_root: Path, store: Optional["Store"] = None) -> BaseTool:
    root = project_root
    _store = store

    class WriteFileTool(BaseTool):
        name: str = "write_file"
        description: str = "Write content to a file (relative to project root). Creates parent dirs."
        args_schema: Type[BaseModel] = _WriteInput

        def _run(self, path: str, content: str) -> str:
            try:
                target = _safe_path(root, path)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content, encoding="utf-8")
                if _store is not None:
                    try:
                        _store.bump_metric("files_made")
                    except Exception:
                        pass
                return f"Written {len(content)} bytes to {path}"
            except Exception as e:
                return f"ERROR: {e}"

    return WriteFileTool()


def tool_run_command(project_root: Path) -> BaseTool:
    root = project_root

    class RunCommandTool(BaseTool):
        name: str = "run_command"
        description: str = "Run a shell command in the project directory. Returns stdout+stderr (capped at 4000 chars) and exit code."
        args_schema: Type[BaseModel] = _CmdInput

        def _run(self, cmd: str, timeout: int = 60) -> str:
            try:
                result = subprocess.run(
                    cmd,
                    shell=True,
                    cwd=str(root),
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )
                out = (result.stdout + result.stderr)[:4000]
                return f"rc={result.returncode}\n{out}"
            except subprocess.TimeoutExpired:
                return f"ERROR: command timed out after {timeout}s"
            except Exception as e:
                return f"ERROR: {e}"

    return RunCommandTool()


def tool_get_state(store: Store) -> BaseTool:
    _store = store

    class GetStateTool(BaseTool):
        name: str = "get_state"
        description: str = "Retrieve a shared value by key from the project's SQLite store."
        args_schema: Type[BaseModel] = _KeyInput

        def _run(self, key: str) -> str:
            val = _store.get_state(key)
            return val if val is not None else f"(no value for key '{key}')"

    return GetStateTool()


def tool_set_state(store: Store) -> BaseTool:
    _store = store

    class SetStateTool(BaseTool):
        name: str = "set_state"
        description: str = "Store a key/value pair in the project's SQLite store for other agents to read."
        args_schema: Type[BaseModel] = _KeyValueInput

        def _run(self, key: str, value: str) -> str:
            _store.set_state(key, value)
            return f"Stored: {key}"

    return SetStateTool()


def tool_web_search(store: Store, cfg: Config) -> BaseTool:
    """DuckDuckGo by default. Falls back to Tavily on CAPTCHA if a key is set.
    Hard-caps at cfg.tools.search_max_per_task per task, caches in SQLite."""
    _store = store
    _cfg = cfg
    _task_count: list[int] = [0]  # mutable counter shared across calls

    class WebSearchTool(BaseTool):
        name: str = "web_search"
        description: str = (
            "Search the web for information. Returns a markdown list of results. "
            f"Capped at {_cfg.tools.search_max_per_task} searches per task."
        )
        args_schema: Type[BaseModel] = _SearchInput

        def _run(self, query: str) -> str:
            if _task_count[0] >= _cfg.tools.search_max_per_task:
                return f"Search cap reached ({_cfg.tools.search_max_per_task} per task). Use cached results."

            cached = _store.fetch_search(query, _cfg.tools.search_cache_days)
            if cached:
                try:
                    _store.bump_metric("searches")
                except Exception:
                    pass
                return cached

            result = _do_search(query, _cfg)
            if result:
                _store.cache_search(query, result)
                _task_count[0] += 1
                try:
                    _store.bump_metric("searches")
                except Exception:
                    pass
            return result or "No results found."

    return WebSearchTool()


def _do_search(query: str, cfg: Config) -> str:
    """Try DDG, fallback to Tavily if key is set and DDG fails."""
    try:
        return _ddg_search(query)
    except Exception as ddg_err:
        if cfg.tools.tavily_api_key:
            try:
                return _tavily_search(query, cfg.tools.tavily_api_key)
            except Exception:
                pass
        return f"Search failed: {ddg_err}"


def _ddg_search(query: str) -> str:
    from duckduckgo_search import DDGS
    with DDGS() as ddgs:
        hits = list(ddgs.text(query, max_results=5))
    lines = []
    for h in hits:
        lines.append(f"- **{h.get('title','')}** — {h.get('href','')}\n  {h.get('body','')[:200]}")
    return "\n".join(lines)


def _tavily_search(query: str, api_key: str) -> str:
    import requests as req
    resp = req.post(
        "https://api.tavily.com/search",
        json={"api_key": api_key, "query": query, "max_results": 5},
        timeout=15,
    )
    resp.raise_for_status()
    results = resp.json().get("results", [])
    lines = []
    for r in results:
        lines.append(f"- **{r.get('title','')}** — {r.get('url','')}\n  {r.get('content','')[:200]}")
    return "\n".join(lines)


def tool_fetch_url() -> BaseTool:
    class FetchUrlTool(BaseTool):
        name: str = "fetch_url"
        description: str = "Fetch a URL and return its text content (stripped of HTML tags, capped at 8000 chars)."
        args_schema: Type[BaseModel] = _UrlInput

        def _run(self, url: str) -> str:
            try:
                import requests as req
                import re
                resp = req.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
                resp.raise_for_status()
                text = re.sub(r"<[^>]+>", " ", resp.text)
                text = re.sub(r"\s+", " ", text).strip()
                return text[:8000]
            except Exception as e:
                return f"ERROR fetching {url}: {e}"

    return FetchUrlTool()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_tools(project_root: Path, store: Store, cfg: Config) -> dict[str, BaseTool]:
    """Build all tools keyed by name. Agents pick subsets via ROLE_TOOLS."""
    return {
        "read_file":   tool_read_file(project_root),
        "write_file":  tool_write_file(project_root, store),
        "run_command": tool_run_command(project_root),
        "get_state":   tool_get_state(store),
        "set_state":   tool_set_state(store),
        "web_search":  tool_web_search(store, cfg),
        "fetch_url":   tool_fetch_url(),
    }
