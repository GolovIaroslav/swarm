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

import argparse
import json
import logging
import os
import shutil
import signal
import sys
import threading
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Disable third-party telemetry and tracing before importing crewai / litellm.
os.environ.setdefault("CREWAI_TELEMETRY_OPT_OUT", "true")
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")
os.environ.setdefault("LITELLM_TELEMETRY", "False")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "false")

# Silence noisy third-party warnings before the TUI starts up.
# We use our own SQLite checkpoint, so CrewAI's pydantic callback warning
# is irrelevant. Deprecations come from inside CrewAI on every agent build.
# RuntimeWarning includes the duckduckgo_search -> ddgs rename notice.
for _cat in (DeprecationWarning, UserWarning, RuntimeWarning, FutureWarning):
    warnings.simplefilter("ignore", _cat)
# also catch the few warnings that slip through child processes
os.environ.setdefault("PYTHONWARNINGS", "ignore")

import questionary
from crewai import Crew, Process
from rich import print as rprint
from rich.table import Table
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
# Project name validation
# ---------------------------------------------------------------------------

def _validate_project_name(name: str) -> "str | None":
    """Return an error string if name is invalid, else None."""
    if not name:
        return "Project name cannot be empty."
    if "/" in name or "\\" in name:
        return "Project name must not contain '/' or '\\'."
    if ".." in name:
        return "Project name must not contain '..'."
    return None


def _warn_project_name(name: str) -> None:
    if " " in name:
        print(f"  [warn] project name contains spaces — directory will be '{name}'")
    if any(ord(c) > 127 for c in name):
        print(f"  [warn] project name contains non-ASCII characters")


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

    err = _validate_project_name(project)
    if err:
        print(f"[swarm] {err}")
        sys.exit(1)
    _warn_project_name(project)

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

def preflight(backend: Backend, cfg: Config, choices: Choices, skip_confirm: bool = False) -> bool:
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
    # For api/remote backends we trust the provider and skip the test.
    raw_text = ""
    if cfg.backend.type in ("lm_studio", "llama_cpp", "custom"):
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
            raw_text = raw_text.strip("`")
            if raw_text.startswith("json"):
                raw_text = raw_text[4:].strip()
            json.loads(raw_text)
            print("  ✓  JSON output: OK")
        except json.JSONDecodeError:
            print(f"  ⚠  JSON output: model returned non-JSON ({raw_text!r})")
            if choices.process == "hierarchical" and not skip_confirm:
                print("     Hierarchical mode needs reliable JSON — consider switching to sequential.")
                if questionary.confirm("Switch to sequential?", default=True).ask():
                    choices.process = "sequential"
        except Exception as e:
            print(f"  ⚠  JSON test skipped: {e}")
    else:
        print("  ·  JSON output: skipped (remote api backend)")

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

    if skip_confirm:
        return True
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


_HANDOFF_TOKEN_BUDGET = 500


def _extract_handoff(raw: str) -> str:
    """Pull the ## HANDOFF section from raw agent output, capped to ~500 tokens.

    Falls back to the trailing 2000 chars if no HANDOFF marker is found.
    Uses tiktoken when available, otherwise rough char-count heuristic.
    """
    upper = raw.upper()
    idx = upper.find("## HANDOFF")
    text = raw[idx:] if idx >= 0 else (raw[-2000:] if len(raw) > 2000 else raw)

    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        tokens = enc.encode(text)
        if len(tokens) <= _HANDOFF_TOKEN_BUDGET:
            return text
        return enc.decode(tokens[:_HANDOFF_TOKEN_BUDGET])
    except Exception:
        # ~4 chars per token rule of thumb
        max_chars = _HANDOFF_TOKEN_BUDGET * 4
        return text[:max_chars]


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
# CLI parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="swarm",
        description=(
            "Local-first multi-agent TUI for autonomous coding. "
            "Talks to LM Studio / llama.cpp / any OpenAI-compat or LiteLLM provider."
        ),
    )
    sub = p.add_subparsers(dest="cmd")

    run_p = sub.add_parser("run", help="Run a crew (default if no subcommand).")
    _add_run_flags(run_p)

    sub.add_parser("list", help="List existing projects under projects/.")

    rm_p = sub.add_parser("rm", help="Delete a project directory.")
    rm_p.add_argument("name", help="Project name")
    rm_p.add_argument("-y", "--yes", action="store_true", help="Skip confirmation")

    sub.add_parser("presets", help="List available pipeline presets.")

    return p


def _add_run_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--project", help="Project name (skips the TUI prompt)")
    p.add_argument(
        "--preset",
        choices=list(pr.PIPELINES.keys()) + ["custom"],
        help="Pipeline preset (skips the TUI prompt)",
    )
    p.add_argument("--goal", help="What the crew should build (skips the TUI prompt)")
    p.add_argument(
        "--roles",
        help="Comma-separated agent roles (only used when --preset=custom)",
    )
    p.add_argument(
        "--process",
        choices=["sequential", "hierarchical"],
        help="Crew process mode (skips the TUI prompt)",
    )
    p.add_argument("--resume", action="store_true", help="Resume an existing project")
    p.add_argument("--no-resume", action="store_true", help="Start over even if a checkpoint exists")
    p.add_argument("--no-monitor", action="store_true", help="Disable the rich live panel")
    p.add_argument("--debug", action="store_true", help="Also write full LiteLLM trace to _logs/debug.log")
    p.add_argument("-y", "--yes", action="store_true", help="Skip 'Start?' confirmation")


# ---------------------------------------------------------------------------
# Subcommands: list / rm / presets
# ---------------------------------------------------------------------------

def _cmd_list(cfg: Config) -> int:
    root = projects_root(cfg)
    if not root.exists():
        print(f"No projects directory at {root}.")
        return 0

    table = Table(title="Projects", show_header=True, header_style="bold")
    table.add_column("Name", style="cyan")
    table.add_column("Tasks", justify="right")
    table.add_column("Done", justify="right")
    table.add_column("Files", justify="right")
    table.add_column("Modified")

    rows = sorted(root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    for proj in rows:
        if not proj.is_dir():
            continue
        db = proj / "_state.db"
        n_total = n_done = 0
        files = "—"
        if db.exists():
            try:
                s = Store(db)
                s.open()
                tasks = s.all_tasks()
                n_total = len(tasks)
                n_done = sum(1 for t in tasks if t.status == "done")
                src = proj / "src"
                if src.exists():
                    files = str(sum(1 for _ in src.rglob("*") if _.is_file()))
                s.close()
            except Exception:
                pass
        mtime = time.strftime("%Y-%m-%d %H:%M", time.localtime(proj.stat().st_mtime))
        table.add_row(proj.name, str(n_total), str(n_done), files, mtime)

    rprint(table)
    return 0


def _cmd_rm(cfg: Config, name: str, assume_yes: bool) -> int:
    target = projects_root(cfg) / name
    if not target.exists():
        print(f"No such project: {target}")
        return 1
    if not assume_yes:
        ok = questionary.confirm(f"Delete {target}? This cannot be undone.").ask()
        if not ok:
            print("aborted.")
            return 1
    shutil.rmtree(target, ignore_errors=True)
    print(f"Deleted {target}.")
    return 0


def _cmd_presets() -> int:
    print("Available presets:")
    for name, fn in pr.PIPELINES.items():
        specs = fn("<goal>")
        roles = ", ".join(s.agent_name for s in specs)
        print(f"  {name:22s}  {roles}")
    print(f"  {'custom':22s}  (pick agents by hand)")
    return 0


# ---------------------------------------------------------------------------
# Main / dispatch
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    cmd = args.cmd or "run"

    cfg = load()

    if cmd == "list":
        return _cmd_list(cfg)
    if cmd == "rm":
        return _cmd_rm(cfg, args.name, args.yes)
    if cmd == "presets":
        return _cmd_presets()

    return _cmd_run(cfg, args)


def _resolve_choices(cfg: Config, args: argparse.Namespace) -> Choices:
    """Build Choices from CLI args, falling back to the interactive TUI
    for any field the user didn't pre-fill."""
    proj_root = projects_root(cfg)
    proj_root.mkdir(parents=True, exist_ok=True)

    project = (args.project or questionary.text("Project name:").ask() or "").strip()
    if not project:
        sys.exit(0)

    err = _validate_project_name(project)
    if err:
        print(f"[swarm] {err}", file=sys.stderr)
        sys.exit(1)
    _warn_project_name(project)

    # resume detection (also handles --resume / --no-resume)
    db_path = proj_root / project / "_state.db"
    resume = False
    if db_path.exists():
        s = Store(db_path)
        s.open()
        unfinished = s.unfinished_tasks()
        s.close()
        if unfinished:
            if args.resume:
                resume = True
            elif args.no_resume:
                shutil.rmtree(proj_root / project, ignore_errors=True)
                resume = False
            else:
                action = questionary.select(
                    f"Found {len(unfinished)} unfinished task(s) in '{project}'. What to do?",
                    choices=["Resume", "Start over", "Delete project"],
                ).ask()
                if action == "Resume":
                    resume = True
                elif action == "Delete project":
                    shutil.rmtree(proj_root / project, ignore_errors=True)

    preset = args.preset or questionary.select(
        "Project preset:",
        choices=list(pr.PIPELINES.keys()) + ["custom"],
    ).ask()
    if not preset:
        sys.exit(0)

    goal = args.goal
    if not goal:
        try:
            goal = questionary.text(
                "Describe the goal (Alt+Enter to finish multiline, Enter for single line):",
                multiline=True,
            ).ask()
        except Exception:
            goal = questionary.text("Describe the goal:").ask()
    if not goal:
        sys.exit(0)
    goal = goal.strip()

    roles: list[str] = []
    if preset == "custom":
        if args.roles:
            roles = [r.strip() for r in args.roles.split(",") if r.strip()]
            unknown = [r for r in roles if r not in ag.ROLES]
            if unknown:
                print(f"Unknown role(s): {unknown}. Valid: {ag.ROLES}", file=sys.stderr)
                return 2  # type: ignore[return-value]
        else:
            roles = questionary.checkbox(
                "Select agents:",
                choices=list(ag.ROLES),
            ).ask() or []
        if not roles:
            print("[swarm] No agents selected. Exiting.")
            sys.exit(0)

    process_choice = args.process or questionary.select(
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


def _cmd_run(cfg: Config, args: argparse.Namespace) -> int:
    # If no CLI args at all, run the full interactive TUI for back-compat.
    fully_interactive = not any([args.project, args.preset, args.goal])
    choices = setup_screen(cfg) if fully_interactive else _resolve_choices(cfg, args)

    print("\n[swarm] Starting backend...")
    backend = Backend(cfg=cfg)
    backend.start()
    print(f"[swarm] Backend ready — model: {backend.model_id}\n")

    if not preflight(backend, cfg, choices, skip_confirm=args.yes):
        backend.stop()
        return 0

    # ---- project directories ----
    proj_root = projects_root(cfg) / choices.project
    crew_dir = proj_root / "_crew"
    log_path = proj_root / "_logs" / "events.log"   # shown in monitor
    db_path = proj_root / "_state.db"

    for d in (crew_dir, log_path.parent):
        d.mkdir(parents=True, exist_ok=True)

    _setup_file_logging(log_path.parent / "run.log", debug=args.debug)
    logging.info(f"[swarm] starting project={choices.project} preset={choices.preset}")

    store = Store(db_path)
    store.open()

    # Reset run-scoped metrics so tokens / file counts don't carry over from
    # previous runs of the same project. Resume keeps the task table intact.
    store.reset_metrics()

    # Try to discover the model's true context window from the server (LM
    # Studio's native API, llama.cpp /props). Fall back to whatever's in
    # config.toml. This is what powers the ctx% gauge in the header.
    detected_ctx = backend.detect_context_window()
    if detected_ctx and detected_ctx != cfg.execution.context_window:
        print(f"[swarm] context window detected from server: {detected_ctx:,} "
              f"(config had {cfg.execution.context_window:,}) — using detected value")
        cfg.execution.context_window = detected_ctx

    _register_litellm_callback(store)
    install_signal_handlers(lambda: (store.close(), backend.stop()))

    # ---- monitor on/off + verbose ----
    live_monitor = cfg.ui.live_monitor and not args.no_monitor
    # When the live panel is on we want a clean screen — silence CrewAI's
    # verbose console output (everything still goes to events.log / debug.log).
    crew_verbose = not live_monitor

    # ---- build tools + LLM(s) ----
    lm_default = backend.llm()
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
    # resolve per-role LLM override from cfg.backend.per_role
    role_llms: dict[str, object] = {}
    for role in needed_roles:
        override = cfg.backend.per_role.get(role)
        if override:
            try:
                role_llms[role] = backend.llm(model_override=override)
                logging.info(f"[swarm] role={role} uses override model={override!r}")
            except Exception as e:
                logging.warning(f"[swarm] role={role} override failed ({e}), using default")
                role_llms[role] = lm_default
        else:
            role_llms[role] = lm_default

    agents_built = {
        name: ag.build(name, role_llms[name], tools_dict, cfg, verbose=crew_verbose)
        for name in needed_roles
    }

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

                prev = store.get_task(s.id)
                started_at = prev.started_at if prev else None
                retry_count = prev.retry_count if prev else 0

                store.upsert_task(TaskRow(
                    id=s.id,
                    agent=s.agent_name,
                    status="done",
                    started_at=started_at,
                    finished_at=int(time.time()),
                    output_file=str(crew_dir / (s.output_file or f"{s.id}.md")),
                    handoff=handoff,
                    retry_count=retry_count,
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
                        next_row.started_at = int(time.time())
                        store.upsert_task(next_row)

            return _cb

        task.callback = _make_callback()
        crewai_tasks.append(task)

    # mark the first task as running before kickoff
    first_row = store.get_task(pending_specs[0].id)
    if first_row:
        first_row.status = "running"
        first_row.started_at = int(time.time())
        store.upsert_task(first_row)

    # ---- start monitor ----
    stop_event = threading.Event()
    if live_monitor:
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
        verbose=crew_verbose,
    )
    if choices.process == "hierarchical":
        crew_kwargs["manager_agent"] = ag.manager(lm_default, cfg, verbose=crew_verbose)

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
        # give the monitor 1s to render the final state after the last
        # task callback fires (refresh tick is 0.5s)
        time.sleep(1.0)
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
