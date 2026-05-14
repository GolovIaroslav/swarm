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
class ApiCfg:
    """Generic remote LLM via LiteLLM (any provider with the right prefix).

    Examples for `model`:
        openrouter/anthropic/claude-3.5-sonnet
        openrouter/qwen/qwen-2.5-coder-32b-instruct
        nvidia_nim/meta/llama-3.1-405b-instruct
        groq/llama-3.1-70b-versatile
        anthropic/claude-3-5-sonnet-20241022
        openai/gpt-4o
        gemini/gemini-2.0-flash-exp
    """
    model: str = ""
    api_key_env: str = ""              # name of env var that holds the key
    base_url: str = ""                  # optional, for self-hosted endpoints


@dataclass
class BackendCfg:
    type: str = "lm_studio"          # lm_studio | llama_cpp | custom | api
    url: str = "http://localhost:1234/v1"
    llama_cpp: LlamaCppCfg = field(default_factory=LlamaCppCfg)
    api: ApiCfg = field(default_factory=ApiCfg)
    # per-role model override; key = role name, value = litellm model string
    per_role: dict = field(default_factory=dict)


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


def _apply(dataclass_instance, data: dict) -> None:
    """Shallow-merge dict keys into a dataclass, ignoring unknown keys."""
    for key, val in data.items():
        if hasattr(dataclass_instance, key):
            setattr(dataclass_instance, key, val)


def load(path: Path | str = DEFAULT_PATH) -> Config:
    """Read TOML file and return a populated Config.

    Falls back to defaults for any missing keys. Raises FileNotFoundError if
    the file is missing — callers should prompt the user to copy the example.
    """
    path = Path(path)
    with path.open("rb") as fh:
        raw = tomllib.load(fh)

    cfg = Config()

    if "backend" in raw:
        braw = raw["backend"]
        nested_keys = {"llama_cpp", "api", "per_role"}
        _apply(cfg.backend, {k: v for k, v in braw.items() if k not in nested_keys})
        if "llama_cpp" in braw:
            _apply(cfg.backend.llama_cpp, braw["llama_cpp"])
        if "api" in braw:
            _apply(cfg.backend.api, braw["api"])
        if "per_role" in braw:
            cfg.backend.per_role = dict(braw["per_role"])

    if "execution" in raw:
        _apply(cfg.execution, raw["execution"])

    if "tools" in raw:
        _apply(cfg.tools, raw["tools"])

    if "ui" in raw:
        _apply(cfg.ui, raw["ui"])

    if "paths" in raw:
        _apply(cfg.paths, raw["paths"])

    return cfg


def projects_root(cfg: Config) -> Path:
    """Resolve `[paths].projects_dir` to an absolute, expanded Path."""
    return Path(cfg.paths.projects_dir).expanduser().resolve()
