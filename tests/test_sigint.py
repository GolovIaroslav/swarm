"""Signal-handler tests: SIGINT mid-run must leave a resumable state.

We don't actually fork a child process — instead we install the same
handler `swarm._cmd_run` would install, then invoke it directly with a
fake signal. This avoids subprocess flakiness and runs in milliseconds.
"""

from __future__ import annotations

import signal
import sys

import pytest

from state import Store, TaskRow
from swarm import install_signal_handlers


@pytest.fixture
def restore_signal():
    """Save and restore SIGINT/SIGTERM handlers so tests don't bleed."""
    sigint = signal.getsignal(signal.SIGINT)
    sigterm = signal.getsignal(signal.SIGTERM) if hasattr(signal, "SIGTERM") else None
    yield
    signal.signal(signal.SIGINT, sigint)
    if sigterm is not None:
        signal.signal(signal.SIGTERM, sigterm)


def test_install_signal_handler_calls_cleanup(monkeypatch, restore_signal):
    cleanup_calls: list[int] = []
    exit_codes: list[int] = []

    monkeypatch.setattr(sys, "exit", lambda code=0: exit_codes.append(code))
    install_signal_handlers(lambda: cleanup_calls.append(1))

    handler = signal.getsignal(signal.SIGINT)
    handler(signal.SIGINT, None)

    assert cleanup_calls == [1]
    assert exit_codes == [0]


def test_install_signal_handler_swallows_cleanup_exceptions(monkeypatch, restore_signal):
    exit_codes: list[int] = []

    def bad_cleanup() -> None:
        raise RuntimeError("cleanup blew up")

    monkeypatch.setattr(sys, "exit", lambda code=0: exit_codes.append(code))
    install_signal_handlers(bad_cleanup)

    handler = signal.getsignal(signal.SIGINT)
    # must not raise: the handler swallows cleanup errors so the process
    # still exits cleanly even if state-flush fails partially
    handler(signal.SIGINT, None)
    assert exit_codes == [0]


def test_install_signal_handler_registers_sigterm(monkeypatch, restore_signal):
    if not hasattr(signal, "SIGTERM"):
        pytest.skip("SIGTERM not available on this platform")

    monkeypatch.setattr(sys, "exit", lambda code=0: None)
    install_signal_handlers(lambda: None)

    sigterm_handler = signal.getsignal(signal.SIGTERM)
    # should be our custom function, not SIG_DFL / SIG_IGN
    assert callable(sigterm_handler)
    assert sigterm_handler not in (signal.SIG_DFL, signal.SIG_IGN)


def test_sigint_during_run_leaves_resumable_state(tmp_path, monkeypatch, restore_signal):
    """End-to-end-ish: open a store, mark a task running, take SIGINT,
    reopen the store and prove the state is intact and resumable."""
    db = tmp_path / "_state.db"

    store = Store(db)
    store.open()
    store.upsert_task(TaskRow(id="research", agent="researcher", status="done",
                              started_at=100, finished_at=200))
    store.upsert_task(TaskRow(id="architecture", agent="architect",
                              status="running", started_at=300))
    store.upsert_task(TaskRow(id="implementation", agent="coder",
                              status="pending"))

    monkeypatch.setattr(sys, "exit", lambda code=0: None)
    install_signal_handlers(lambda: store.close())

    # simulate Ctrl+C: invoke the installed handler synchronously
    handler = signal.getsignal(signal.SIGINT)
    handler(signal.SIGINT, None)

    # store must have been closed by the cleanup; db file is on disk
    assert db.exists()

    # reopen and verify everything is there for resume
    s2 = Store(db)
    s2.open()
    rows = {r.id: r for r in s2.all_tasks()}
    assert rows["research"].status == "done"
    assert rows["architecture"].status == "running"  # was mid-flight
    assert rows["implementation"].status == "pending"

    unfinished = {r.id for r in s2.unfinished_tasks()}
    assert unfinished == {"architecture", "implementation"}
    s2.close()
