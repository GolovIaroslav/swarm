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
