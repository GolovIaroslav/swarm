#!/usr/bin/env python3
"""swarm — entry point.

Boot order:
  1. config.load()
  2. setup_screen() — questionary TUI with resume detection
  3. backend.start() (spawns llama-server if configured)
  4. preflight() — ping LLM, JSON-capability test, web-search test
  5. Build crew from presets/custom + Store + tools
  6. Start Monitor + Watchdog
  7. crew.kickoff(); checkpoint after every task via callbacks
  8. On exit (clean or signal): stop watchdog, close store, backend.stop()

Ctrl+C anywhere is safe — SIGINT handler flushes the store and exits.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Disable third-party telemetry and tracing before importing crewai / litellm.
os.environ.setdefault("CREWAI_TELEMETRY_OPT_OUT", "true")
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")
os.environ.setdefault("LITELLM_TELEMETRY", "False")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "false")

import questionary
from crewai import Crew, Process
from rich import print as rprint
from rich.tree import Tree

import agents as ag
import presets as pr
from backend import Backend
from config import Config, load, projects_root
from extractor import extract
from monitor import Monitor, plain_stdout_monitor
from state import Store, TaskRow
from tasks import build_task
from tools import make_tools
from watchdog import Watchdog


# ---------------------------------------------------------------------------
# User choices from setup_screen
# ---------------------------------------------------------------------------

@dataclass
class Choices:
    project: str
    preset: str
    goal: str
    roles: list[str] = field(default_factory=list)
    process: str = "sequential"
    resume: bool = False


# ---------------------------------------------------------------------------
# Signal handling
# ---------------------------------------------------------------------------

_cleanup_fn = None


def install_signal_handlers(on_exit) -> None:
    """Register SIGINT/SIGTERM to call on_exit() then sys.exit(0)."""
    global _cleanup_fn
    _cleanup_fn = on_exit

    def _handler(sig, frame):
        print("\n[swarm] signal received — saving state and exiting...")
        try:
            if _cleanup_fn:
                _cleanup_fn()
        except Exception:
            pass
        sys.exit(0)

    signal.signal(signal.SIGINT, _handler)
    # SIGTERM is not available on Windows
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _handler)


# ---------------------------------------------------------------------------
# Setup screen
# ---------------------------------------------------------------------------

def setup_screen(cfg: Config) -> Choices:
    """Questionary flow — project name, preset, description, agents, process.

    Detects existing checkpoints and offers Resume/Start over/Delete.
    """
    proj_root = projects_root(cfg)
    proj_root.mkdir(parents=True, exist_ok=True)

    # Build autocomplete from existing project directories
    try:
        from prompt_toolkit.completion import WordCompleter
        existing = sorted(p.name for p in proj_root.iterdir() if p.is_dir())
        completer = WordCompleter(existing, ignore_case=True)
    except Exception:
        completer = None

    project = questionary.text(
        "Project name:",
        completer=completer,
    ).ask()
    if not project:
        sys.exit(0)
    project = project.strip()

    # Resume detection
    db_path = proj_root / project / "_state.db"
    resume = False
    if db_path.exists():
        s = Store(db_path)
        s.open()
        unfinished = s.unfinished_tasks()
        s.close()

        if unfinished:
            action = questionary.select(
                f"Found {len(unfinished)} unfinished task(s) in '{project}'. What to do?",
                choices=["Resume", "Start over", "Delete project"],
            ).ask()

            if action is None or action == "Start over":
                resume = False
            elif action == "Resume":
                resume = True
            elif action == "Delete project":
                import shutil
                shutil.rmtree(proj_root / project, ignore_errors=True)
                print(f"[swarm] Deleted project '{project}'.")
                resume = False

    # Preset selection
    preset_choices = list(pr.PIPELINES.keys()) + ["custom"]
    preset = questionary.select(
        "Project preset:",
        choices=preset_choices,
    ).ask()
    if not preset:
        sys.exit(0)

    # Goal description
    try:
        goal = questionary.text(
            "Describe the goal (Alt+Enter to finish multiline, or Enter for single line):",
            multiline=True,
        ).ask()
    except Exception:
        goal = questionary.text("Describe the goal:").ask()
    if not goal:
        sys.exit(0)
    goal = goal.strip()

    # Custom roles
    roles: list[str] = []
    if preset == "custom":
        roles = questionary.checkbox(
            "Select agents:",
            choices=list(ag.ROLES),
        ).ask() or []
        if not roles:
            print("[swarm] No agents selected. Exiting.")
            sys.exit(0)

    # Process mode
    process_choice = questionary.select(
        "Process mode:",
        choices=["sequential", "hierarchical"],
        default="sequential",
    ).ask() or "sequential"

    return Choices(
        project=project,
        preset=preset,
        goal=goal,
        roles=roles,
        process=process_choice,
        resume=resume,
    )


# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------

def preflight(backend: Backend, cfg: Config, choices: Choices) -> bool:
    """Ping LLM, JSON test, web-search test. Returns True if user confirms run."""
    print("\n[preflight] Checking environment...\n")

    # 1. Ping LLM
    try:
        backend.ping()
        print(f"  ✓  LLM: {backend.model_id}")
    except Exception as e:
        print(f"  ✗  LLM ping failed: {e}")
        return False

    # 2. JSON capability test (needed for hierarchical mode)
    json_ok = False
    try:
        import openai as _openai
        client = _openai.OpenAI(
            base_url=cfg.backend.url,
            api_key="lm-studio",
        )
        resp = client.chat.completions.create(
            model=backend.model_id,
            messages=[{
                "role": "user",
                "content": 'Respond with exactly this JSON and nothing else: {"status":"ok"}',
            }],
            max_tokens=30,
        )
        raw_text = (resp.choices[0].message.content or "").strip()
        # strip markdown fences if model wraps JSON
        raw_text = raw_text.strip("`")
        if raw_text.startswith("json"):
            raw_text = raw_text[4:].strip()
        json.loads(raw_text)
        json_ok = True
        print("  ✓  JSON output: OK")
    except json.JSONDecodeError:
        print(f"  ⚠  JSON output: model returned non-JSON ({raw_text!r})")
        if choices.process == "hierarchical":
            print("     Hierarchical mode needs reliable JSON — consider switching to sequential.")
            if questionary.confirm("Switch to sequential?", default=True).ask():
                choices.process = "sequential"
    except Exception as e:
        print(f"  ⚠  JSON test skipped: {e}")

    # 3. Web search test (only when researcher is in the pipeline)
    agent_names = _agent_names_for(choices)
    if "researcher" in agent_names and cfg.tools.web_search:
        print("  Checking web search (DDG)...", end=" ", flush=True)
        try:
            from tools import _ddg_search
            result = _ddg_search("python 3.12")
            print("✓") if result else print("⚠  (no results)")
        except Exception as e:
            print(f"⚠  ({e})")

    # 4. Summary
    specs = _specs_for(choices)
    est_min = len(specs) * 10
    print(f"\n  Model:   {backend.model_id}")
    print(f"  Preset:  {choices.preset}")
    print(f"  Agents:  {', '.join(agent_names)}")
    print(f"  Tasks:   {len(specs)}  (est. ≈{est_min} min)")
    print(f"  Process: {choices.process}")
    print()

    confirmed = questionary.confirm(
        "Start the crew? (May run for hours. Ctrl+C saves state.)",
        default=True,
    ).ask()
    return bool(confirmed)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _agent_names_for(choices: Choices) -> list[str]:
    seen: list[str] = []
    for s in _specs_for(choices):
        if s.agent_name not in seen:
            seen.append(s.agent_name)
    return seen


def _specs_for(choices: Choices):
    if choices.preset == "custom":
        return pr.custom(choices.goal, choices.roles)
    return pr.PIPELINES[choices.preset](choices.goal)


def _setup_file_logging(log_path: Path, debug: bool = False) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    # events.log — clean INFO+ log shown in the monitor panel
    events_path = log_path.parent / "events.log"
    events_h = logging.FileHandler(str(events_path), mode="a", encoding="utf-8")
    events_h.setLevel(logging.INFO)
    events_h.setFormatter(fmt)
    # suppress noisy third-party loggers in the clean log
    events_h.addFilter(_SwarmFilter())

    handlers: list[logging.Handler] = [events_h]

    if debug:
        # debug.log — everything including litellm internals
        debug_path = log_path.parent / "debug.log"
        debug_h = logging.FileHandler(str(debug_path), mode="a", encoding="utf-8")
        debug_h.setLevel(logging.DEBUG)
        debug_h.setFormatter(fmt)
        handlers.append(debug_h)
        print(f"  [debug] full log: {debug_path}")

    root = logging.getLogger()
    for h in handlers:
        root.addHandler(h)
    root.setLevel(logging.DEBUG)

    # keep litellm noise out of events.log (it goes to debug.log if --debug)
    for noisy in ("LiteLLM", "litellm", "httpx", "httpcore", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


class _SwarmFilter(logging.Filter):
    """Allow only our own log records into the clean events.log."""
    _blocked = ("LiteLLM", "litellm", "httpx", "httpcore", "openai", "crewai")

    def filter(self, record: logging.LogRecord) -> bool:
        return not any(record.name.startswith(n) for n in self._blocked)


def _extract_handoff(raw: str) -> str:
    """Pull the ## HANDOFF section from raw agent output (up to 2000 chars)."""
    upper = raw.upper()
    idx = upper.find("## HANDOFF")
    if idx >= 0:
        return raw[idx : idx + 2000]
    return raw[-500:] if len(raw) > 500 else raw


def _get_raw(output) -> str:
    """Normalise TaskOutput or plain string to a str."""
    if hasattr(output, "raw"):
        return str(output.raw)
    return str(output)


def _print_file_tree(proj_root: Path) -> None:
    src = proj_root / "src"
    if not src.exists():
        return
    tree = Tree(f"[bold]{proj_root.name}/src/[/bold]")
    for p in sorted(src.rglob("*")):
        if p.is_file():
            tree.add(str(p.relative_to(src)))
    rprint(tree)


# ---------------------------------------------------------------------------
# LiteLLM token tracking
# ---------------------------------------------------------------------------

def _register_litellm_callback(store: "Store") -> None:  # type: ignore[name-defined]
    """Hook into litellm to count tokens_in / tokens_out / llm_requests."""
    try:
        import litellm

        def _success(kwargs, response, start_time, end_time):
            try:
                usage = getattr(response, "usage", None)
                if usage:
                    store.bump_metric("tokens_in",  getattr(usage, "prompt_tokens",     0) or 0)
                    store.bump_metric("tokens_out", getattr(usage, "completion_tokens",  0) or 0)
                store.bump_metric("llm_requests")
            except Exception:
                pass

        litellm.success_callback = litellm.success_callback or []
        litellm.success_callback.append(_success)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Help text
# ---------------------------------------------------------------------------

def _print_help() -> None:
    print("""swarm — local multi-agent coding TUI

Usage:
  python swarm.py           launch interactive setup screen
  python swarm.py --debug   same but write full debug log to _logs/debug.log
  python swarm.py --help    show this message

Logs (written per project under projects/<name>/_logs/):
  events.log   clean INFO-level log — shown in the monitor panel
  debug.log    everything including LiteLLM internals (only with --debug)

Config:
  config.toml  copy from config.toml.example and edit
""")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    argv = list(argv or [])
    debug = "--debug" in argv
    if "--help" in argv or "-h" in argv:
        _print_help()
        return 0

    cfg = load()

    choices = setup_screen(cfg)

    print("\n[swarm] Starting backend...")
    backend = Backend(cfg=cfg)
    backend.start()
    print(f"[swarm] Backend ready — model: {backend.model_id}\n")

    if not preflight(backend, cfg, choices):
        backend.stop()
        return 0

    # ---- project directories ----
    proj_root = projects_root(cfg) / choices.project
    crew_dir = proj_root / "_crew"
    log_path = proj_root / "_logs" / "events.log"   # shown in monitor
    db_path = proj_root / "_state.db"

    for d in (crew_dir, log_path.parent):
        d.mkdir(parents=True, exist_ok=True)

    _setup_file_logging(log_path.parent / "run.log", debug=debug)
    logging.info(f"[swarm] starting project={choices.project} preset={choices.preset}")

    store = Store(db_path)
    store.open()

    _register_litellm_callback(store)
    install_signal_handlers(lambda: (store.close(), backend.stop()))

    # ---- build tools + LLM ----
    lm = backend.llm()
    tools_dict = make_tools(proj_root, store, cfg)

    # ---- task specs ----
    all_specs = _specs_for(choices)

    # ---- resume: collect already-done task IDs ----
    done_ids: set[str] = set()
    if choices.resume:
        for row in store.all_tasks():
            if row.status == "done":
                done_ids.add(row.id)
        print(f"[swarm] resuming — {len(done_ids)} task(s) already done")

    # ---- upsert pending rows for non-done tasks ----
    for spec in all_specs:
        if spec.id not in done_ids and not store.get_task(spec.id):
            store.upsert_task(TaskRow(id=spec.id, agent=spec.agent_name, status="pending"))

    # ---- build agents (only for tasks that will actually run) ----
    pending_specs = [s for s in all_specs if s.id not in done_ids]
    if not pending_specs:
        print("[swarm] All tasks already done. Nothing to run.")
        _print_file_tree(proj_root)
        store.close()
        backend.stop()
        return 0

    needed_roles = {s.agent_name for s in pending_specs}
    agents_built = {name: ag.build(name, lm, tools_dict, cfg) for name in needed_roles}

    # ---- build crewai Task objects with per-task callbacks ----
    crewai_tasks: list = []

    for i, spec in enumerate(pending_specs):
        agent = agents_built[spec.agent_name]
        task = build_task(spec, agent, crewai_tasks[:], output_dir=crew_dir)

        def _make_callback(s=spec, idx=i, specs=pending_specs):
            def _cb(output) -> None:
                raw = _get_raw(output)
                handoff = _extract_handoff(raw)

                extracted = extract(raw, proj_root)
                store.bump_metric("files_made", len(extracted))

                store.upsert_task(TaskRow(
                    id=s.id,
                    agent=s.agent_name,
                    status="done",
                    finished_at=int(time.time()),
                    output_file=str(crew_dir / (s.output_file or f"{s.id}.md")),
                    handoff=handoff,
                ))

                if extracted:
                    file_list = ", ".join(
                        str(f.path.relative_to(proj_root)) for f in extracted
                    )
                    logging.info(f"[{s.agent_name}] extracted {len(extracted)} file(s): {file_list}")
                logging.info(f"[{s.agent_name}] task {s.id!r} done")

                # advance status of the next task to "running"
                if idx + 1 < len(specs):
                    next_row = store.get_task(specs[idx + 1].id)
                    if next_row and next_row.status == "pending":
                        next_row.status = "running"
                        store.upsert_task(next_row)

            return _cb

        task.callback = _make_callback()
        crewai_tasks.append(task)

    # mark the first task as running before kickoff
    first_row = store.get_task(pending_specs[0].id)
    if first_row:
        first_row.status = "running"
        store.upsert_task(first_row)

    # ---- start monitor ----
    stop_event = threading.Event()
    if cfg.ui.live_monitor:
        monitor = Monitor(store, log_path, choices.project, backend.model_id, cfg)
        mon_thread = monitor.run(stop_event)
    else:
        mon_thread = threading.Thread(
            target=plain_stdout_monitor,
            args=(store, log_path, stop_event),
            daemon=True,
            name="monitor",
        )
        mon_thread.start()

    # ---- watchdog callbacks ----
    def _on_hang(task_id: str) -> None:
        print(f"\n[watchdog] Tier 1: task {task_id!r} hung — retry triggered", flush=True)
        logging.warning(f"[watchdog] task {task_id} hung")
        store.bump_metric("retries")

    def _on_exhausted(task_id: str) -> str:
        action = questionary.select(
            f"Task {task_id!r} exhausted {cfg.execution.max_retry} retries:",
            choices=["retry", "skip", "abort"],
        ).ask() or "abort"
        return action

    watchdog = Watchdog(log_path, store, cfg, _on_hang, _on_exhausted)
    watchdog.start()

    # ---- assemble Crew ----
    process = (
        Process.hierarchical
        if choices.process == "hierarchical"
        else Process.sequential
    )

    crew_kwargs: dict = dict(
        agents=list(agents_built.values()),
        tasks=crewai_tasks,
        process=process,
        verbose=True,
    )
    if choices.process == "hierarchical":
        crew_kwargs["manager_agent"] = ag.manager(lm, cfg)

    crew = Crew(**crew_kwargs)

    print(f"\n[swarm] Launching {len(crewai_tasks)} task(s) "
          f"({choices.process}) — Ctrl+C to save state\n")

    try:
        crew.kickoff()
    except KeyboardInterrupt:
        print("\n[swarm] Interrupted.")
    except Exception as exc:
        print(f"\n[swarm] Crew error: {exc}")
        logging.exception("crew.kickoff() raised an exception")
    finally:
        stop_event.set()
        watchdog.stop()
        mon_thread.join(timeout=3)
        store.close()
        backend.stop()

    print("\n[swarm] Run complete.")
    _print_file_tree(proj_root)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
