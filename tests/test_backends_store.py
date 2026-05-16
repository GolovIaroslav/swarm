"""Unit tests for backends_store — pure persistence layer."""

from __future__ import annotations

import json

import pytest

import backends_store as bs
from config import BackendCfg


@pytest.fixture
def store_file(tmp_path):
    return tmp_path / "backends.json"


def test_load_all_missing_returns_empty(store_file):
    data = bs.load_all(store_file)
    assert data["backends"] == {}
    assert data["last_used"] is None


def test_load_all_corrupt_returns_empty(store_file):
    store_file.parent.mkdir(parents=True, exist_ok=True)
    store_file.write_text("{not valid json")
    data = bs.load_all(store_file)
    assert data["backends"] == {}


def test_save_and_get_roundtrip(store_file):
    bs.save("my_lm", {"type": "lm_studio", "url": "http://localhost:1234/v1"}, store_file)
    entry = bs.get("my_lm", store_file)
    assert entry["type"] == "lm_studio"
    assert "last_used_at" in entry


def test_list_names_sorted_by_recency(store_file, monkeypatch):
    times = iter([100, 200, 50])
    monkeypatch.setattr(bs.time, "time", lambda: next(times))
    bs.save("a", {"type": "lm_studio"}, store_file)
    bs.save("b", {"type": "custom"}, store_file)
    bs.save("c", {"type": "api"}, store_file)
    # last_used_at for a=100, b=200, c=50 → order b,a,c
    names = bs.list_names(store_file)
    assert names == ["b", "a", "c"]


def test_remove_existing(store_file):
    bs.save("victim", {"type": "lm_studio"}, store_file)
    assert bs.remove("victim", store_file) is True
    assert bs.get("victim", store_file) is None


def test_remove_missing_returns_false(store_file):
    assert bs.remove("nothing", store_file) is False


def test_remove_clears_last_used(store_file):
    bs.save("only", {"type": "lm_studio"}, store_file)
    bs.remove("only", store_file)
    data = bs.load_all(store_file)
    assert data["last_used"] is None


def test_short_summary_each_type():
    assert "lm_studio" in bs.short_summary({"type": "lm_studio", "url": "u"})
    assert "custom" in bs.short_summary({"type": "custom", "url": "u"})
    assert "llama.cpp" in bs.short_summary(
        {"type": "llama_cpp", "llama_cpp": {"model": "/path/to/foo.gguf", "ctx": 16384}}
    )
    assert "foo.gguf" in bs.short_summary(
        {"type": "llama_cpp", "llama_cpp": {"model": "/path/to/foo.gguf", "ctx": 16384}}
    )
    assert "api" in bs.short_summary(
        {"type": "api", "api": {"model": "openrouter/anthropic/claude"}}
    )


def test_apply_to_cfg_llama_cpp_full():
    cfg = BackendCfg()
    entry = {
        "type": "llama_cpp",
        "llama_cpp": {
            "binary": "/opt/llama-server",
            "model": "/data/model.gguf",
            "ctx": 131072,
            "ngl": 999,
            "port": 8080,
            "extra_args": ["-fa", "on"],
            "env": {"LD_LIBRARY_PATH": "/opt/lib"},
        },
    }
    bs.apply_to_cfg(entry, cfg)
    assert cfg.type == "llama_cpp"
    assert cfg.llama_cpp.binary == "/opt/llama-server"
    assert cfg.llama_cpp.ctx == 131072
    assert cfg.llama_cpp.port == 8080
    assert cfg.llama_cpp.extra_args == ["-fa", "on"]
    assert cfg.llama_cpp.env["LD_LIBRARY_PATH"] == "/opt/lib"


def test_apply_to_cfg_api():
    cfg = BackendCfg()
    entry = {
        "type": "api",
        "api": {
            "model": "openrouter/anthropic/claude-3.5-sonnet",
            "api_key_env": "OPENROUTER_API_KEY",
            "base_url": "",
        },
    }
    bs.apply_to_cfg(entry, cfg)
    assert cfg.type == "api"
    assert cfg.api.model == "openrouter/anthropic/claude-3.5-sonnet"
    assert cfg.api.api_key_env == "OPENROUTER_API_KEY"


def test_apply_to_cfg_per_role():
    cfg = BackendCfg()
    entry = {
        "type": "lm_studio",
        "per_role": {"coder": "openai/gpt-4o"},
    }
    bs.apply_to_cfg(entry, cfg)
    assert cfg.per_role == {"coder": "openai/gpt-4o"}


def test_save_persists_disk_format(store_file):
    bs.save("x", {"type": "lm_studio", "url": "u"}, store_file)
    raw = json.loads(store_file.read_text())
    assert raw["version"] == 1
    assert raw["last_used"] == "x"
    assert raw["backends"]["x"]["type"] == "lm_studio"


def test_touch_last_used_bumps_timestamp(store_file, monkeypatch):
    times = iter([100, 500])
    monkeypatch.setattr(bs.time, "time", lambda: next(times))
    bs.save("x", {"type": "lm_studio"}, store_file)
    assert bs.get("x", store_file)["last_used_at"] == 100
    bs.touch_last_used("x", store_file)
    assert bs.get("x", store_file)["last_used_at"] == 500


def test_store_path_respects_xdg(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    p = bs.store_path()
    assert str(p).startswith(str(tmp_path / "xdg" / "swarm"))
    assert p.name == "backends.json"
