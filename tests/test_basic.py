"""Basic smoke tests for all core modules.

Run with: pytest tests/ -v
"""

import time
import pytest


# ===========================================================================
# config
# ===========================================================================

def test_config_load_all_sections(tmp_path):
    toml = tmp_path / "config.toml"
    toml.write_text(
        """
[backend]
type = "llama_cpp"
url  = "http://localhost:8090/v1"

[backend.llama_cpp]
binary = "/usr/bin/llama-server"
model  = "/models/test.gguf"
ctx    = 16384
ngl    = 20
port   = 8090

[execution]
process              = "sequential"
max_iter             = 10
max_retry            = 2
max_rpm              = 15
task_timeout_minutes = 20
context_window       = 32000
max_response_tokens  = 2048

[tools]
web_search          = false
search_provider     = "tavily"
search_max_per_task = 3
search_cache_days   = 3
tavily_api_key      = "test-key"

[ui]
live_monitor = false

[paths]
projects_dir = "/tmp/test_projects"
"""
    )
    from config import load

    cfg = load(toml)

    assert cfg.backend.type == "llama_cpp"
    assert cfg.backend.url == "http://localhost:8090/v1"
    assert cfg.backend.llama_cpp.binary == "/usr/bin/llama-server"
    assert cfg.backend.llama_cpp.model == "/models/test.gguf"
    assert cfg.backend.llama_cpp.ctx == 16384
    assert cfg.backend.llama_cpp.ngl == 20
    assert cfg.backend.llama_cpp.port == 8090

    assert cfg.execution.process == "sequential"
    assert cfg.execution.max_iter == 10
    assert cfg.execution.max_retry == 2
    assert cfg.execution.max_rpm == 15
    assert cfg.execution.task_timeout_minutes == 20
    assert cfg.execution.context_window == 32000
    assert cfg.execution.max_response_tokens == 2048

    assert cfg.tools.web_search is False
    assert cfg.tools.search_provider == "tavily"
    assert cfg.tools.search_max_per_task == 3
    assert cfg.tools.search_cache_days == 3
    assert cfg.tools.tavily_api_key == "test-key"

    assert cfg.ui.live_monitor is False
    assert cfg.paths.projects_dir == "/tmp/test_projects"


def test_config_defaults(tmp_path):
    toml = tmp_path / "minimal.toml"
    toml.write_text("[backend]\ntype = \"lm_studio\"\n")
    from config import load

    cfg = load(toml)
    assert cfg.execution.max_iter == 15
    assert cfg.execution.context_window == 60000
    assert cfg.tools.web_search is True
    assert cfg.ui.live_monitor is True


# ===========================================================================
# state
# ===========================================================================

def test_store_upsert_get_task(tmp_path):
    from state import Store, TaskRow

    s = Store(tmp_path / "test.db")
    s.open()

    row = TaskRow(id="t1", agent="coder", status="pending", started_at=100)
    s.upsert_task(row)

    got = s.get_task("t1")
    assert got is not None
    assert got.agent == "coder"
    assert got.status == "pending"
    assert got.started_at == 100

    row.status = "done"
    row.finished_at = 200
    s.upsert_task(row)
    got2 = s.get_task("t1")
    assert got2.status == "done"
    assert got2.finished_at == 200

    assert s.get_task("nonexistent") is None
    s.close()


def test_store_all_and_unfinished(tmp_path):
    from state import Store, TaskRow

    s = Store(tmp_path / "test.db")
    s.open()

    s.upsert_task(TaskRow(id="a", agent="architect", status="done"))
    s.upsert_task(TaskRow(id="b", agent="coder", status="running"))
    s.upsert_task(TaskRow(id="c", agent="tester", status="pending"))

    assert len(s.all_tasks()) == 3
    unfinished = s.unfinished_tasks()
    ids = {r.id for r in unfinished}
    assert "b" in ids and "c" in ids
    assert "a" not in ids
    s.close()


def test_store_set_get_state(tmp_path):
    from state import Store

    s = Store(tmp_path / "test.db")
    s.open()

    s.set_state("arch", "some value")
    assert s.get_state("arch") == "some value"
    assert s.get_state("missing") is None

    s.set_state("arch", "updated")
    assert s.get_state("arch") == "updated"
    s.close()


def test_store_cache_search_fetch(tmp_path):
    from state import Store

    s = Store(tmp_path / "test.db")
    s.open()

    s.cache_search("python 3.12", "result text")

    assert s.fetch_search("python 3.12", ttl_days=7) == "result text"
    assert s.fetch_search("python 3.12", ttl_days=0) is None
    assert s.fetch_search("unknown", ttl_days=7) is None
    s.close()


def test_store_bump_metric(tmp_path):
    from state import Store

    s = Store(tmp_path / "test.db")
    s.open()

    assert s.get_metric("files_made") == 0.0
    s.bump_metric("files_made", 3.0)
    assert s.get_metric("files_made") == 3.0
    s.bump_metric("files_made", 2.0)
    assert s.get_metric("files_made") == 5.0
    s.close()


def test_store_wal_mode(tmp_path):
    from state import Store

    s = Store(tmp_path / "test.db")
    s.open()

    cur = s.conn.execute("PRAGMA journal_mode")
    assert cur.fetchone()[0] == "wal"
    s.close()


# ===========================================================================
# extractor
# ===========================================================================

_MD_TWO_BLOCKS = """
## src/hello.py
```python
def hello():
    return "hello world"
```

## src/utils.py
```python
def add(a, b):
    return a + b
```
"""

_MD_HEADING_NO_EXT = """
## not_a_file
```python
x = 1
```
"""

_MD_NO_HEADING = """
```python
def solve():
    return 42
```

```javascript
console.log("hi");
```
"""

_MD_PATH_ESCAPE = """
## ../escape.py
```python
x = 1
```
"""

_MD_TRIPLE_BLOCKS = """
## src/a.py
```python
A = 1
```

## src/b.py
```python
B = 2
```

## src/c.py
```python
C = 3
```
"""


def test_extract_two_blocks(tmp_path):
    from extractor import extract

    results = extract(_MD_TWO_BLOCKS, tmp_path)
    names = {r.path.name for r in results}
    assert "hello.py" in names
    assert "utils.py" in names
    assert len(results) == 2


def test_extract_three_blocks(tmp_path):
    from extractor import extract

    results = extract(_MD_TRIPLE_BLOCKS, tmp_path)
    assert len(results) == 3
    names = {r.path.name for r in results}
    assert names == {"a.py", "b.py", "c.py"}


def test_extract_heading_no_extension_becomes_snippet(tmp_path):
    from extractor import extract

    results = extract(_MD_HEADING_NO_EXT, tmp_path)
    assert len(results) == 1
    assert results[0].path.name == "snippet_1.py"


def test_extract_no_heading_snippet_fallback(tmp_path):
    from extractor import extract

    results = extract(_MD_NO_HEADING, tmp_path)
    assert len(results) == 2
    names = [r.path.name for r in results]
    assert "snippet_1.py" in names
    assert "snippet_2.js" in names


def test_extract_path_escape_blocked(tmp_path):
    from extractor import extract

    results = extract(_MD_PATH_ESCAPE, tmp_path)
    assert results == []


def test_extract_writes_correct_content(tmp_path):
    from extractor import extract

    results = extract(_MD_TWO_BLOCKS, tmp_path)
    for r in results:
        assert r.path.exists()
        assert r.bytes_written > 0
        text = r.path.read_text()
        assert "def " in text or "return" in text


# ===========================================================================
# tools
# ===========================================================================

def test_tool_path_escape_blocked(tmp_path):
    from tools import tool_read_file

    tool = tool_read_file(tmp_path)
    result = tool._run(path="../escape.txt")
    assert "ERROR" in result


def test_tool_write_path_escape_blocked(tmp_path):
    from tools import tool_write_file

    tool = tool_write_file(tmp_path)
    result = tool._run(path="../evil.py", content="x=1")
    assert "ERROR" in result


def test_tool_write_read_roundtrip(tmp_path):
    from tools import tool_write_file, tool_read_file

    write_tool = tool_write_file(tmp_path)
    read_tool = tool_read_file(tmp_path)

    write_result = write_tool._run(path="src/hello.py", content="print('hi')")
    assert "Written" in write_result

    content = read_tool._run(path="src/hello.py")
    assert "print('hi')" in content


def test_tool_read_missing_file(tmp_path):
    from tools import tool_read_file

    tool = tool_read_file(tmp_path)
    result = tool._run(path="nonexistent.py")
    assert "ERROR" in result


def test_tool_run_command_rc0(tmp_path):
    from tools import tool_run_command

    tool = tool_run_command(tmp_path)
    result = tool._run(cmd="echo hello_world", timeout=10)
    assert "rc=0" in result
    assert "hello_world" in result


def test_tool_run_command_rc_nonzero(tmp_path):
    from tools import tool_run_command

    tool = tool_run_command(tmp_path)
    result = tool._run(cmd="exit 42", timeout=10)
    assert "rc=42" in result


# ===========================================================================
# presets
# ===========================================================================

def test_all_pipelines_nonempty():
    from presets import PIPELINES

    for name, fn in PIPELINES.items():
        specs = fn("test goal")
        assert len(specs) > 0, f"Pipeline {name!r} returned empty list"


def test_all_pipelines_have_agent_names():
    from presets import PIPELINES
    from agents import ROLES

    for name, fn in PIPELINES.items():
        for spec in fn("test goal"):
            assert spec.agent_name in ROLES, (
                f"Pipeline {name!r} spec {spec.id!r} has unknown role {spec.agent_name!r}"
            )


def test_python_lib_pipeline_order():
    from presets import python_lib

    specs = python_lib("build a library")
    roles = [s.agent_name for s in specs]
    assert roles == ["researcher", "architect", "coder", "tester", "docs"]


def test_research_prototype_pipeline():
    from presets import research_prototype

    specs = research_prototype("write a fib function")
    assert len(specs) == 3
    assert specs[0].agent_name == "researcher"
    assert specs[-1].agent_name == "coder"


def test_custom_builds_requested_roles():
    from presets import custom

    specs = custom("test goal", ["architect", "coder"])
    assert len(specs) == 2
    assert specs[0].agent_name == "architect"
    assert specs[1].agent_name == "coder"


def test_custom_unknown_role_raises():
    from presets import custom

    with pytest.raises(ValueError, match="Unknown role"):
        custom("test goal", ["nonexistent_role"])


def test_custom_empty_raises_or_returns_empty():
    from presets import custom

    specs = custom("test goal", [])
    assert specs == []


# ===========================================================================
# project name validation
# ===========================================================================

def test_validate_project_name_empty():
    from swarm import _validate_project_name
    assert _validate_project_name("") is not None


def test_validate_project_name_slash():
    from swarm import _validate_project_name
    assert _validate_project_name("foo/bar") is not None


def test_validate_project_name_backslash():
    from swarm import _validate_project_name
    assert _validate_project_name("foo\\bar") is not None


def test_validate_project_name_dotdot():
    from swarm import _validate_project_name
    assert _validate_project_name("../evil") is not None


def test_validate_project_name_valid():
    from swarm import _validate_project_name
    assert _validate_project_name("my_project") is None
    assert _validate_project_name("my-project-2") is None


def test_validate_project_name_spaces_allowed():
    from swarm import _validate_project_name
    # spaces are a warning, not an error
    assert _validate_project_name("my project") is None


# ===========================================================================
# json capability check (preflight)
# ===========================================================================

def test_json_capable_clean_object():
    from swarm import _check_json_capable
    assert _check_json_capable('{"status":"ok"}') is True


def test_json_capable_with_backticks():
    from swarm import _check_json_capable
    assert _check_json_capable('`{"status":"ok"}`') is True


def test_json_capable_with_fenced_json():
    from swarm import _check_json_capable
    assert _check_json_capable('```json\n{"status":"ok"}\n```'.strip("\n").strip("`")) is True
    # also the realistic shape we strip in _check_json_capable
    assert _check_json_capable('json\n{"status":"ok"}') is True


def test_json_capable_rejects_prose():
    from swarm import _check_json_capable
    assert _check_json_capable("Sure, here is the JSON: {status: ok}") is False
    assert _check_json_capable("ok") is False
    assert _check_json_capable("") is False


def test_json_capable_array_ok():
    from swarm import _check_json_capable
    assert _check_json_capable("[1, 2, 3]") is True


# ===========================================================================
# subcommands: list / rm
# ===========================================================================

def test_cmd_list_empty_dir(tmp_path, capsys):
    from config import Config
    from swarm import _cmd_list

    cfg = Config()
    cfg.paths.projects_dir = str(tmp_path / "projects")
    rc = _cmd_list(cfg)
    assert rc == 0


def test_cmd_list_missing_dir(tmp_path, capsys):
    from config import Config
    from swarm import _cmd_list

    cfg = Config()
    cfg.paths.projects_dir = str(tmp_path / "does_not_exist")
    rc = _cmd_list(cfg)
    assert rc == 0
    out = capsys.readouterr().out
    assert "No projects" in out


def test_cmd_rm_missing_project(tmp_path, capsys):
    from config import Config
    from swarm import _cmd_rm

    cfg = Config()
    cfg.paths.projects_dir = str(tmp_path / "projects")
    rc = _cmd_rm(cfg, "nonexistent", assume_yes=True)
    assert rc == 1
    out = capsys.readouterr().out
    assert "No such project" in out


def test_cmd_rm_existing_project(tmp_path):
    from config import Config
    from swarm import _cmd_rm

    proj_dir = tmp_path / "projects" / "victim"
    proj_dir.mkdir(parents=True)
    (proj_dir / "marker.txt").write_text("x")

    cfg = Config()
    cfg.paths.projects_dir = str(tmp_path / "projects")

    rc = _cmd_rm(cfg, "victim", assume_yes=True)
    assert rc == 0
    assert not proj_dir.exists()


# ===========================================================================
# backend: api type validation
# ===========================================================================

def test_backend_api_missing_env_var(monkeypatch):
    from backend import Backend
    from config import Config

    cfg = Config()
    cfg.backend.type = "api"
    cfg.backend.api.model = "openrouter/some/model"
    cfg.backend.api.api_key_env = "FAKE_PROVIDER_KEY_THAT_DOES_NOT_EXIST"

    monkeypatch.delenv("FAKE_PROVIDER_KEY_THAT_DOES_NOT_EXIST", raising=False)

    b = Backend(cfg=cfg)
    with pytest.raises(RuntimeError, match="FAKE_PROVIDER_KEY_THAT_DOES_NOT_EXIST"):
        b.start()


def test_backend_api_empty_model():
    from backend import Backend
    from config import Config

    cfg = Config()
    cfg.backend.type = "api"
    cfg.backend.api.model = ""

    b = Backend(cfg=cfg)
    with pytest.raises(RuntimeError, match="model"):
        b.start()


def test_backend_api_with_env_var_set(monkeypatch):
    from backend import Backend
    from config import Config

    cfg = Config()
    cfg.backend.type = "api"
    cfg.backend.api.model = "openrouter/some/model"
    cfg.backend.api.api_key_env = "FAKE_PROVIDER_KEY_PRESENT"

    monkeypatch.setenv("FAKE_PROVIDER_KEY_PRESENT", "sk-test")

    b = Backend(cfg=cfg)
    b.start()
    assert b.model_id == "openrouter/some/model"


# ===========================================================================
# search cache TTL
# ===========================================================================

def test_search_cache_expired(tmp_path):
    from state import Store

    s = Store(tmp_path / "test.db")
    s.open()

    s.cache_search("expired_query", "old result")
    # backdate the timestamp to 8 days ago
    eight_days_ago = int(time.time()) - 8 * 86400
    s.conn.execute(
        "UPDATE search_cache SET ts=? WHERE query=?",
        (eight_days_ago, "expired_query"),
    )
    s.conn.commit()

    assert s.fetch_search("expired_query", ttl_days=7) is None
    # but with ttl=9 days it should still be there
    assert s.fetch_search("expired_query", ttl_days=9) == "old result"
    s.close()


def test_search_cache_fresh_hit(tmp_path):
    from state import Store

    s = Store(tmp_path / "test.db")
    s.open()

    s.cache_search("fresh_query", "fresh result")
    assert s.fetch_search("fresh_query", ttl_days=7) == "fresh result"
    s.close()


# ===========================================================================
# web search per-task cap
# ===========================================================================

def test_web_search_cap_per_task(tmp_path, monkeypatch):
    from config import Config
    from state import Store
    import tools as tools_mod

    s = Store(tmp_path / "test.db")
    s.open()

    cfg = Config()
    cfg.tools.search_max_per_task = 5

    call_count = [0]
    def fake_ddg(query):
        call_count[0] += 1
        return f"result for {query}"
    monkeypatch.setattr(tools_mod, "_ddg_search", fake_ddg)

    tool = tools_mod.tool_web_search(s, cfg)

    # first 5 unique queries — all should pass through to fake_ddg
    for i in range(5):
        out = tool._run(query=f"q_{i}")
        assert "result for" in out

    # 6th, 7th — should be capped, fake_ddg not called again
    for i in range(5, 7):
        out = tool._run(query=f"q_{i}")
        assert "cap reached" in out.lower()

    assert call_count[0] == 5
    s.close()


# ===========================================================================
# backend: per-role model override
# ===========================================================================

def test_backend_llm_override_local_wraps_openai_prefix():
    from backend import Backend
    from config import Config

    cfg = Config()
    cfg.backend.type = "lm_studio"
    cfg.backend.url = "http://localhost:1234/v1"

    b = Backend(cfg=cfg)
    b.model_id = "default-model"

    llm_default = b.llm()
    llm_override = b.llm(model_override="other/coder-model")

    assert llm_default.model == "openai/default-model"
    assert llm_override.model == "openai/other/coder-model"


def test_backend_llm_override_strips_existing_openai_prefix():
    from backend import Backend
    from config import Config

    cfg = Config()
    cfg.backend.type = "lm_studio"

    b = Backend(cfg=cfg)
    b.model_id = "anything"

    llm = b.llm(model_override="openai/already-prefixed")
    assert llm.model == "openai/already-prefixed"
    # not "openai/openai/already-prefixed"


def test_backend_llm_override_api_passes_full_model_string(monkeypatch):
    """For api backend, the override must be passed as-is (provider prefix
    intact) into LLM(). CrewAI internally strips the prefix on LLM.model,
    so we capture the constructor kwargs instead of inspecting .model."""
    import backend as backend_mod
    from backend import Backend
    from config import Config

    captured: dict = {}

    class _FakeLLM:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.model = kwargs.get("model")

    monkeypatch.setattr(backend_mod, "LLM", _FakeLLM)

    cfg = Config()
    cfg.backend.type = "api"
    cfg.backend.api.model = "openrouter/default/model"
    cfg.backend.api.api_key_env = "TEST_API_KEY"
    monkeypatch.setenv("TEST_API_KEY", "sk-x")

    b = Backend(cfg=cfg)
    b.model_id = cfg.backend.api.model

    b.llm(model_override="openrouter/qwen/qwen-2.5-coder-32b-instruct")
    assert captured["model"] == "openrouter/qwen/qwen-2.5-coder-32b-instruct"
    assert captured["api_key"] == "sk-x"


def test_per_role_override_wiring(monkeypatch):
    """Simulates the per-role wiring loop from swarm._cmd_run."""
    from backend import Backend
    from config import Config

    cfg = Config()
    cfg.backend.type = "lm_studio"
    cfg.backend.per_role = {"coder": "specialised/coder-model"}

    b = Backend(cfg=cfg)
    b.model_id = "default-model"

    # mirror the loop in _cmd_run
    needed_roles = {"architect", "coder"}
    role_llms = {}
    for role in needed_roles:
        override = cfg.backend.per_role.get(role)
        if override:
            role_llms[role] = b.llm(model_override=override)
        else:
            role_llms[role] = b.llm()

    assert role_llms["architect"].model == "openai/default-model"
    assert role_llms["coder"].model == "openai/specialised/coder-model"


def test_web_search_cap_does_not_count_cache_hits(tmp_path, monkeypatch):
    from config import Config
    from state import Store
    import tools as tools_mod

    s = Store(tmp_path / "test.db")
    s.open()

    cfg = Config()
    cfg.tools.search_max_per_task = 3

    call_count = [0]
    def fake_ddg(query):
        call_count[0] += 1
        return f"result for {query}"
    monkeypatch.setattr(tools_mod, "_ddg_search", fake_ddg)

    tool = tools_mod.tool_web_search(s, cfg)

    # first call — real search
    tool._run(query="repeat")
    # repeat — cache hit, counter must not bump
    for _ in range(10):
        out = tool._run(query="repeat")
        assert "result for repeat" in out

    # after 10 cache hits, fresh queries should still work up to the cap
    tool._run(query="other_1")
    tool._run(query="other_2")

    # one more fresh query should be capped (3 fresh = limit)
    out = tool._run(query="other_3")
    assert "cap reached" in out.lower()

    s.close()
