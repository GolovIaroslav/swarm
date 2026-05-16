"""Unit tests for the hang detector (watchdog.py).

We bypass the polling loop and call Watchdog._check() directly with a
stale log file so the test runs in milliseconds.
"""

from __future__ import annotations

import os
import time

import pytest

from config import Config
from state import Store, TaskRow
from watchdog import Watchdog


def _make_cfg(timeout_min: int = 1, max_retry: int = 3) -> Config:
    cfg = Config()
    cfg.execution.task_timeout_minutes = timeout_min
    cfg.execution.max_retry = max_retry
    return cfg


def test_watchdog_does_nothing_when_no_running_tasks(tmp_path):
    log = tmp_path / "run.log"
    log.write_text("ok")

    store = Store(tmp_path / "s.db")
    store.open()
    store.upsert_task(TaskRow(id="t1", agent="coder", status="done"))

    hang_calls: list[str] = []
    wd = Watchdog(
        log_path=log,
        store=store,
        cfg=_make_cfg(),
        on_hang=lambda tid: hang_calls.append(tid),
        on_exhausted=lambda tid: "abort",
    )

    wd._check(timeout_secs=1)
    assert hang_calls == []
    store.close()


def test_watchdog_does_nothing_when_log_fresh(tmp_path):
    log = tmp_path / "run.log"
    log.write_text("fresh")

    store = Store(tmp_path / "s.db")
    store.open()
    store.upsert_task(TaskRow(id="t1", agent="coder", status="running",
                              started_at=int(time.time())))

    hang_calls: list[str] = []
    wd = Watchdog(
        log_path=log,
        store=store,
        cfg=_make_cfg(),
        on_hang=lambda tid: hang_calls.append(tid),
        on_exhausted=lambda tid: "abort",
    )

    # log mtime is just now, timeout_secs is 60 — nothing should fire
    wd._check(timeout_secs=60)
    assert hang_calls == []
    store.close()


def test_watchdog_tier1_marks_pending_and_bumps_retry(tmp_path):
    log = tmp_path / "run.log"
    log.write_text("stale")
    # backdate the log so it looks idle
    old = time.time() - 600
    os.utime(log, (old, old))

    store = Store(tmp_path / "s.db")
    store.open()
    store.upsert_task(TaskRow(
        id="t1", agent="coder", status="running",
        started_at=int(time.time()) - 600, retry_count=0,
    ))

    hang_calls: list[str] = []
    wd = Watchdog(
        log_path=log,
        store=store,
        cfg=_make_cfg(max_retry=3),
        on_hang=lambda tid: hang_calls.append(tid),
        on_exhausted=lambda tid: "abort",
    )

    wd._check(timeout_secs=60)

    assert hang_calls == ["t1"]
    row = store.get_task("t1")
    assert row.status == "pending"
    assert row.retry_count == 1
    store.close()


def test_watchdog_tier2_skip_marks_failed(tmp_path):
    log = tmp_path / "run.log"
    log.write_text("stale")
    old = time.time() - 600
    os.utime(log, (old, old))

    store = Store(tmp_path / "s.db")
    store.open()
    # retry already at max
    store.upsert_task(TaskRow(
        id="t1", agent="coder", status="running",
        started_at=int(time.time()) - 600, retry_count=3,
    ))

    exhausted_calls: list[str] = []
    def on_exhausted(tid: str) -> str:
        exhausted_calls.append(tid)
        return "skip"

    wd = Watchdog(
        log_path=log,
        store=store,
        cfg=_make_cfg(max_retry=3),
        on_hang=lambda tid: None,
        on_exhausted=on_exhausted,
    )

    wd._check(timeout_secs=60)

    assert exhausted_calls == ["t1"]
    row = store.get_task("t1")
    assert row.status == "failed"
    assert "skipped" in (row.error or "")
    store.close()


def test_watchdog_tier2_abort_stops_loop(tmp_path):
    log = tmp_path / "run.log"
    log.write_text("stale")
    old = time.time() - 600
    os.utime(log, (old, old))

    store = Store(tmp_path / "s.db")
    store.open()
    store.upsert_task(TaskRow(
        id="t1", agent="coder", status="running",
        started_at=int(time.time()) - 600, retry_count=3,
    ))

    wd = Watchdog(
        log_path=log,
        store=store,
        cfg=_make_cfg(max_retry=3),
        on_hang=lambda tid: None,
        on_exhausted=lambda tid: "abort",
    )

    wd._check(timeout_secs=60)
    assert wd._stop.is_set()  # abort signals the polling loop to exit
    store.close()


def test_watchdog_handles_missing_log(tmp_path):
    # log doesn't exist yet — watchdog must not crash
    log = tmp_path / "never_created.log"

    store = Store(tmp_path / "s.db")
    store.open()
    store.upsert_task(TaskRow(id="t1", agent="coder", status="running"))

    wd = Watchdog(
        log_path=log,
        store=store,
        cfg=_make_cfg(),
        on_hang=lambda tid: pytest.fail("should not fire when log missing"),
        on_exhausted=lambda tid: "abort",
    )

    wd._check(timeout_secs=60)
    store.close()
