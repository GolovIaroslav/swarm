# swarm

Local-first multi-agent TUI for autonomous coding. Built on [CrewAI](https://github.com/crewAIInc/crewAI), talks to local LLM servers ([LM Studio](https://lmstudio.ai/) or [`llama.cpp`](https://github.com/ggerganov/llama.cpp)) over the OpenAI-compatible API. Tell it what to build, walk away — come back to working code on disk.

![python](https://img.shields.io/badge/python-3.12-blue) ![license](https://img.shields.io/badge/license-MIT-green) ![status](https://img.shields.io/badge/status-beta-orange)

## Why

Most agent orchestrators (ccswarm, overstory, operator, agent-deck, …) are wired to Claude Code / Codex CLI and cloud APIs. None of them target a model behind LM Studio or `llama.cpp`. This one does — and it's built to run for hours without falling over.

It's also not a chat wrapper. It's a small TUI that walks you through setup, then a checkpointed pipeline of role-specialised agents that hand off through a SQLite store. Crash-resume works. Ctrl+C works.

## Features

- **One command** — `python swarm.py` — questionary TUI walks you through the project.
- **Two local backends** — LM Studio (running its server) or `llama.cpp` (we spawn `llama-server` for you and kill it on exit).
- **9-agent pool** — researcher, architect, coder, tester, reviewer, devops, security, docs, refactorer. Six presets or compose your own.
- **Hierarchical or sequential.** Pre-flight tests whether the model can produce clean JSON and warns down to sequential if it can't.
- **Resume after crashes.** SQLite checkpoint after every task. Pick up where it left off.
- **Three-tier watchdog.** Log-mtime hang detector → auto-retry → user-prompted escalation.
- **Context-budget aware.** Files on disk + per-task HANDOFF blocks + SQLite shared state + `respect_context_window=True`. Built for 32–60k context models.
- **Live monitor.** Rich panel with task table, token / file / search counters, log tail.
- **Web search.** DuckDuckGo by default (free, no key). Optional Tavily fallback. Cached in SQLite for 7 days.
- **Cross-platform.** Linux, macOS, Windows.

## How it looks

```
╭─ _test │ qwen2.5-coder-14b │ 12m 03s │ ctx 38% │ rpm 12.4 ─╮
├──────────────────────────┬───────────────────────────────────┤
│ TASKS                    │ STATS                             │
│ ✓ researcher  research   │ Tokens IN:   45,231               │
│ ✓ architect  arch (8m)   │ Tokens OUT:  18,400               │
│ ⚙ coder      impl (4m)   │ Files made:  17                   │
│ ⏳ tester    testing     │ Searches:    3                    │
│ ⏳ docs      docs        │ Retries:     0                    │
├──────────────────────────┴───────────────────────────────────┤
│ LOG (last 10 lines)                                          │
│ [coder] extracted 4 file(s): src/api/users.py, ...           │
╰──────────────────────────────────────────────────────────────╯
```

## Quick start

```bash
# 1. Python env (3.12 — 3.14 still breaks tiktoken)
python3.12 -m venv ~/progs/crewai-env
source ~/progs/crewai-env/bin/activate
pip install -r requirements.txt

# 2. Fire up LM Studio (load a model, start the server on :1234)
#    or point config.toml at your llama-server binary + a GGUF.

# 3. Config
cp config.toml.example config.toml
# edit if needed

# 4. Run
python swarm.py
```

On Windows / macOS the steps are the same — just swap the venv path.

## Output layout

Each run lives under `projects/<name>/`:

```
projects/my-app/
├── _state.db          SQLite — tasks, shared state, search cache, metrics
├── _crew/*.md         raw agent output (one file per task)
├── _logs/events.log   clean log shown in the monitor
├── _logs/run.log      same, plus debug.log if you pass --debug
└── src/               extracted, ready-to-run code
```

## Presets

| Preset | Pipeline |
|---|---|
| `python_lib` | researcher → architect → coder → tester → docs |
| `web_api` | architect → coder ║ devops → tester → security → reviewer |
| `cli_tool` | architect → coder → tester → docs |
| `refactor_existing` | refactorer → tester → reviewer |
| `research_prototype` | researcher → architect → coder |
| `custom` | pick agents by hand |

`║` means "parallel" — runs as `async_execution=True` where the tasks don't block each other.

## Configuration

All settings in `config.toml`. The interesting knobs:

- `[backend]` — `lm_studio`, `llama_cpp`, or `custom` OpenAI-compat URL.
- `[execution]` — `process` (`hierarchical`/`sequential`), `max_retry`, `max_rpm`, `task_timeout_minutes`, `context_window`.
- `[tools]` — web search provider + per-task cap + cache TTL.
- `[paths]` — where projects get written.

See `config.toml.example` for the full template with comments.

## CLI flags

```
python swarm.py             # interactive setup
python swarm.py --debug     # also write full LiteLLM trace to _logs/debug.log
python swarm.py --help      # show this
```

## Hardware

Tested on:
- Arch Linux, 32 GB RAM, RTX 3060 6 GB VRAM
- LM Studio, models in the 4–14B range (gemma, qwen2.5-coder, llama-3.1)

For hierarchical mode you want ≥7B with reliable JSON tool-calling. Smaller models work fine in sequential mode.

## What's planned / what's not

**Planned:** more presets, better resume UX (skip individual tasks), token-aware HANDOFF compaction, OpenRouter as another backend.

**Not planned:** git worktrees / multi-branch coordination, agent mailbox / mesh comms, MCP server integration. The current sequential+hierarchical + SQLite shared state is enough for one local model.

## License

[MIT](LICENSE).

## Credits

Ideas borrowed from [overstory](https://github.com/jayminwest/overstory) (SQLite state, tiered watchdog) and the rest of the [awesome-agent-orchestrators](https://github.com/andyrewlee/awesome-agent-orchestrators) crowd.
