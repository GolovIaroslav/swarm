"""Persistent store of named backend configurations.

Lives at $XDG_CONFIG_HOME/swarm/backends.json (default ~/.config/swarm/...).
Pure data: the TUI wizard reads & writes it, nobody else touches it. This
replaces the [backend] section of config.toml for end users who shouldn't
have to edit TOML files by hand.

Schema (versioned):

  {
    "version": 1,
    "last_used": "gemma_local" | null,
    "backends": {
      "<name>": {
        "type": "lm_studio" | "llama_cpp" | "custom" | "api",
        "url": "...",                       # lm_studio / custom
        "llama_cpp": {binary, model, ctx, ngl, port, extra_args, env},
        "api":       {model, api_key_env, base_url},
        "per_role":  {role: model_string},  # optional
        "last_used_at": <unix_ts>
      }
    }
  }

Anything missing falls back to dataclass defaults in config.py.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Optional

from config import ApiCfg, BackendCfg, LlamaCppCfg


def store_path() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / "swarm" / "backends.json"


def _empty() -> dict:
    return {"version": 1, "last_used": None, "backends": {}}


def load_all(path: Optional[Path] = None) -> dict:
    p = path or store_path()
    if not p.exists():
        return _empty()
    try:
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict) or "backends" not in data:
            return _empty()
        return data
    except (json.JSONDecodeError, OSError):
        return _empty()


def save_all(data: dict, path: Optional[Path] = None) -> None:
    p = path or store_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def list_names(path: Optional[Path] = None) -> list[str]:
    """Names sorted by last_used_at desc (most recently used first)."""
    data = load_all(path)
    items = list(data.get("backends", {}).items())
    items.sort(key=lambda kv: kv[1].get("last_used_at", 0), reverse=True)
    return [name for name, _ in items]


def get(name: str, path: Optional[Path] = None) -> Optional[dict]:
    return load_all(path).get("backends", {}).get(name)


def save(name: str, entry: dict, path: Optional[Path] = None) -> None:
    data = load_all(path)
    entry = dict(entry)
    entry["last_used_at"] = int(time.time())
    data["backends"][name] = entry
    data["last_used"] = name
    save_all(data, path)


def remove(name: str, path: Optional[Path] = None) -> bool:
    data = load_all(path)
    if name not in data.get("backends", {}):
        return False
    del data["backends"][name]
    if data.get("last_used") == name:
        data["last_used"] = None
    save_all(data, path)
    return True


def touch_last_used(name: str, path: Optional[Path] = None) -> None:
    data = load_all(path)
    if name in data.get("backends", {}):
        data["backends"][name]["last_used_at"] = int(time.time())
        data["last_used"] = name
        save_all(data, path)


def short_summary(entry: dict) -> str:
    """One-line description for the picker (`name — summary`)."""
    t = entry.get("type", "?")
    if t == "lm_studio":
        return f"lm_studio · {entry.get('url', 'localhost:1234')}"
    if t == "custom":
        return f"custom · {entry.get('url', '?')}"
    if t == "llama_cpp":
        lc = entry.get("llama_cpp", {})
        model = Path(lc.get("model", "?")).name or "?"
        return f"llama.cpp · {model} · ctx {lc.get('ctx', '?')}"
    if t == "api":
        api = entry.get("api", {})
        return f"api · {api.get('model', '?')}"
    return t


def apply_to_cfg(entry: dict, backend_cfg: BackendCfg) -> None:
    """Populate a BackendCfg in-place from a saved entry dict."""
    backend_cfg.type = entry.get("type", "lm_studio")
    if "url" in entry:
        backend_cfg.url = entry["url"]
    if entry.get("per_role"):
        backend_cfg.per_role = dict(entry["per_role"])
    if "llama_cpp" in entry:
        lc = entry["llama_cpp"]
        backend_cfg.llama_cpp = LlamaCppCfg(
            binary=lc.get("binary", ""),
            model=lc.get("model", ""),
            ctx=int(lc.get("ctx", 32768)),
            ngl=int(lc.get("ngl", 0)),
            port=int(lc.get("port", 8090)),
            extra_args=list(lc.get("extra_args", [])),
            env=dict(lc.get("env", {})),
        )
    if "api" in entry:
        api = entry["api"]
        backend_cfg.api = ApiCfg(
            model=api.get("model", ""),
            api_key_env=api.get("api_key_env", ""),
            base_url=api.get("base_url", ""),
        )
