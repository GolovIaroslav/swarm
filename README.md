# swarm

Local-first multi-agent TUI for autonomous coding. Built on CrewAI, talks to local LLM servers (LM Studio or llama.cpp) over the OpenAI-compatible API. Tell it what to build, walk away, come back to working code on disk.

![python](https://img.shields.io/badge/python-3.12-blue) ![license](https://img.shields.io/badge/license-MIT-green) ![status](https://img.shields.io/badge/status-beta-orange)

## Why

Most agent orchestrators out there (ccswarm, overstory, operator, agent-deck, and so on) are wired to Claude Code, Codex CLI and cloud APIs. None of them target a model behind LM Studio or llama.cpp. This one does, and it's built to run for hours without falling over.

It's also not a chat wrapper. It's a small TUI that walks you through setup, then a checkpointed pipeline of role-specialised agents that hand off through a SQLite store. Crash-resume works. Ctrl+C works.

## What it does

- One command, `python swarm.py`. A questionary TUI walks you through project name, preset, goal and process mode.
- Two local backends. LM Studio (you run its server) or llama.cpp (we spawn `llama-server` for you and kill it on exit). A `custom` backend type covers any OpenAI-compatible URL.
- Nine roles in the agent pool: researcher, architect, coder, tester, reviewer, devops, security, docs, refactorer. Six presets shipped, or compose your own.
- Hierarchical or sequential crews. The pre-flight tests whether the model can produce clean JSON and warns down to sequential if it can't.
- Resume after crashes. Checkpoint is written to SQLite after every task, not every kickoff. You can pick up where you left off.
- Three-tier watchdog. Log-mtime hang detector, then auto-retry, then a user prompt.
- Context-budget aware. Files go to disk, every task ends with a short HANDOFF block, shared state lives in SQLite, and `respect_context_window=True` is the last-resort net. Built for 32-60k context models.
- Live monitor with task table, token counter, file counter, search counter and log tail.
- Web search through DuckDuckGo by default (free, no key). Optional Tavily fallback. Cached in SQLite for 7 days.
- Cross-platform. Linux, macOS, Windows.

## Quick start

```bash
# Python env. 3.12 only for now, 3.14 still breaks tiktoken.
python3.12 -m venv ~/progs/crewai-env
source ~/progs/crewai-env/bin/activate
pip install -r requirements.txt

# Fire up LM Studio (load a model, start the server on port 1234),
# or point config.toml at your llama-server binary and a GGUF file.

# Config
cp config.toml.example config.toml

# Run
python swarm.py
```

On Windows and macOS the steps are the same, just swap the venv path.

## Output layout

Each run lives under `projects/<name>/`:

```
projects/my-app/
    _state.db      SQLite with tasks, shared state, search cache and metrics
    _crew/         raw agent output, one markdown file per task
    _logs/         events.log shown in the monitor, debug.log if --debug
    src/           extracted, ready-to-run code
```

## Presets

- `python_lib` — researcher, architect, coder, tester, docs (in order)
- `web_api` — architect, then coder and devops in parallel, then tester, security, reviewer
- `cli_tool` — architect, coder, tester, docs
- `refactor_existing` — refactorer, tester, reviewer (over an existing src tree)
- `research_prototype` — researcher, architect, coder
- `custom` — pick agents by hand from a checkbox list

## Configuration

All knobs in `config.toml`. The interesting ones:

- `[backend]` — `lm_studio`, `llama_cpp`, or `custom` OpenAI-compatible URL.
- `[execution]` — `process` (`hierarchical` or `sequential`), `max_retry`, `max_rpm`, `task_timeout_minutes`, `context_window`.
- `[tools]` — web search provider, per-task cap, cache TTL.
- `[paths]` — where projects get written.

See `config.toml.example` for the full template with comments.

## CLI flags

```
python swarm.py             interactive setup
python swarm.py --debug     also write full LiteLLM trace to _logs/debug.log
python swarm.py --help      show usage
```

## Hardware

Tested on Arch Linux, 32 GB RAM, RTX 3060 6 GB VRAM. LM Studio with models in the 4-14B range (gemma, qwen2.5-coder, llama-3.1).

For hierarchical mode you want at least a 7B model with reliable JSON tool-calling. Smaller models work fine in sequential mode.

## What's planned

More presets (data science, API client). Better resume UX (skip individual tasks, not just the whole pipeline). Token-aware HANDOFF compaction. OpenRouter as another backend. Maybe CLI flags to skip the TUI entirely for scripting.

## What's not planned

Git worktrees and multi-branch coordination. Agent mailbox or mesh comms. MCP server integration. Sequential plus hierarchical plus SQLite shared state is enough for one local model.

## License

MIT. See `LICENSE`.

## Credits

Ideas borrowed from [overstory](https://github.com/jayminwest/overstory) (SQLite state, tiered watchdog) and the rest of the [awesome-agent-orchestrators](https://github.com/andyrewlee/awesome-agent-orchestrators) crowd. Built on top of [CrewAI](https://github.com/crewAIInc/crewAI).
