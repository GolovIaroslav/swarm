"""Smoke tests for resource leaks during long runs.

A 24h real run isn't feasible in CI, so we exercise the open/close cycles
of Store and extractor a lot, and check that file descriptors and Python
allocations stay flat. Catches the common patterns that bite multi-day runs.
"""

from __future__ import annotations

import gc
import os
import sys
import tracemalloc

import pytest

from extractor import extract
from state import Store, TaskRow


_LINUX_FD_DIR = "/proc/self/fd"


def _open_fd_count() -> int:
    """Count open file descriptors of the current process. Linux-only."""
    if not os.path.isdir(_LINUX_FD_DIR):
        pytest.skip("requires Linux /proc/self/fd")
    return len(os.listdir(_LINUX_FD_DIR))


_SAMPLE_MD = """
## src/a.py
```python
def a():
    return 1
```

## src/b.py
```python
def b():
    return 2
```

```python
# no heading — fallback snippet
x = 3
```
""".strip()


def test_store_open_close_does_not_leak_fds(tmp_path):
    """200 open/close cycles on the same DB must not grow fd count."""
    db = tmp_path / "leak.db"

    # warm up — first open may allocate static handles
    s = Store(db)
    s.open()
    s.close()

    before = _open_fd_count()

    for _ in range(200):
        s = Store(db)
        s.open()
        s.upsert_task(TaskRow(id="t1", agent="coder", status="done"))
        s.get_task("t1")
        s.close()

    gc.collect()
    after = _open_fd_count()

    # allow tiny noise (sqlite WAL/shm temp files) but not 200x growth
    assert after - before <= 5, f"fd count grew from {before} to {after}"


def test_extract_many_blocks_stable_memory(tmp_path):
    """Calling extract() 100 times on a non-trivial markdown must not blow up RSS."""
    tracemalloc.start()
    gc.collect()
    snap_before = tracemalloc.take_snapshot()

    for _ in range(100):
        extract(_SAMPLE_MD, tmp_path)

    gc.collect()
    snap_after = tracemalloc.take_snapshot()

    stats = snap_after.compare_to(snap_before, "filename")
    growth = sum(s.size_diff for s in stats if s.size_diff > 0)
    tracemalloc.stop()

    # 100 calls; we accept up to ~5 MB of allocator overhead. A real leak
    # would be tens of MB at minimum.
    assert growth < 5 * 1024 * 1024, f"allocated {growth} bytes across 100 extracts"


def test_store_cycle_with_metrics_stable_memory(tmp_path):
    """Tight loop of upsert + bump_metric + read — the path the run callbacks hit."""
    db = tmp_path / "tight.db"
    s = Store(db)
    s.open()

    tracemalloc.start()
    gc.collect()
    snap_before = tracemalloc.take_snapshot()

    for i in range(500):
        s.upsert_task(TaskRow(id=f"t{i % 10}", agent="coder", status="running",
                              started_at=i, retry_count=0))
        s.bump_metric("tokens_in", 100)
        s.bump_metric("files_made", 1)
        s.get_task(f"t{i % 10}")
        s.all_tasks()

    gc.collect()
    snap_after = tracemalloc.take_snapshot()
    stats = snap_after.compare_to(snap_before, "filename")
    growth = sum(st.size_diff for st in stats if st.size_diff > 0)
    tracemalloc.stop()

    s.close()
    assert growth < 5 * 1024 * 1024, f"allocated {growth} bytes across 500 store ops"
