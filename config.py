"""Load and validate config.toml.

Single source of truth for runtime settings. Anything that isn't in `config.toml`
must have a documented default here. Other modules import `load()` and never
touch the TOML directly.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib


DEFAULT_PATH = Path(__file__).resolve().parent / "config.toml"


@dataclass
class LlamaCppCfg:
    binary: str = ""
    model: str = ""
    ctx: int = 32768
    ngl: int = 0
    port: int = 8090


@dataclass
class BackendCfg:
    type: str = "lm_studio"          # lm_studio | llama_cpp | custom
    url: str = "http://localhost:1234/v1"
    llama_cpp: LlamaCppCfg = field(default_factory=LlamaCppCfg)


@dataclass
class ExecutionCfg:
    process: str = "hierarchical"    # hierarchical | sequential
    max_iter: int = 15
    max_retry: int = 3
    max_rpm: int = 30
    task_timeout_minutes: int = 30
    context_window: int = 60000
    max_response_tokens: int = 4096


@dataclass
class ToolsCfg:
    web_search: bool = True
    search_provider: str = "ddg"     # ddg | tavily
    search_max_per_task: int = 5
    search_cache_days: int = 7
    tavily_api_key: str = ""


@dataclass
class UiCfg:
    live_monitor: bool = True


@dataclass
class PathsCfg:
    projects_dir: str = "~/progs/crewai/projects"


@dataclass
class Config:
    backend: BackendCfg = field(default_factory=BackendCfg)
    execution: ExecutionCfg = field(default_factory=ExecutionCfg)
    tools: ToolsCfg = field(default_factory=ToolsCfg)
    ui: UiCfg = field(default_factory=UiCfg)
    paths: PathsCfg = field(default_factory=PathsCfg)


def load(path: Path | str = DEFAULT_PATH) -> Config:
    """Read TOML file and return a populated Config.

    Falls back to defaults for any missing keys. Raises FileNotFoundError if
    the file is missing — callers should prompt the user to copy the example.
    """
    raise NotImplementedError("implement in session 2")


def projects_root(cfg: Config) -> Path:
    """Resolve `[paths].projects_dir` to an absolute, expanded Path."""
    raise NotImplementedError("implement in session 2")
