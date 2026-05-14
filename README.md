# swarm

Local-first multi-agent TUI for autonomous coding. Built on [CrewAI](https://github.com/crewAIInc/crewAI), talks to local LLM servers (LM Studio or `llama.cpp`) over the OpenAI-compatible API. Tell it what to build, walk away — come back to working code on disk.

## Why

Existing agent orchestrators (ccswarm, overstory, operator, etc.) are wired to Claude Code / Codex CLI and cloud APIs. None of them target a local model behind LM Studio or `llama.cpp`. This one does — and it's designed to run for days without falling over.

## Features

- **One command** — `python swarm.py` — TUI walks you through project setup.
- **Two local backends out of the box** — LM Studio and `llama.cpp` (auto-spawns `llama-server`).
- **9-agent pool** — researcher, architect, coder, tester, reviewer, devops, security, docs, refactorer. Pick a preset or compose your own.
- **Hierarchical or sequential** crews with a pre-flight model capability test.
- **Resume after crashes** — SQLite checkpoints after every task, not every kickoff. Ctrl+C is safe.
- **Three-tier watchdog** — log-mtime hang detector, auto-retry, then user-prompted escalation.
- **Context budget aware** — artifacts on disk, HANDOFF blocks, SQLite shared state, and `respect_context_window=True`. Built for models that cap at 60k context.
- **Live monitor** — `rich`-powered panel with tasks, stats, and tail-of-log.
- **Web search** — DuckDuckGo by default (free, no key), Tavily fallback if you supply a key. Cached for 7 days.

## Status

**Pre-alpha — under active development.** Skeleton in place; module implementations landing across the next few iterations. See [crew.py](crew.py) for the early sequential prototype that already runs against LM Studio.

## Requirements

- Linux (tested on Arch). Should work anywhere `uv` and a local OpenAI-compat server run.
- Python **3.12** (3.14 still breaks `tiktoken` at the time of writing).
- LM Studio **or** a `llama.cpp` build with `llama-server`.

## Quick start

```bash
# 1. Python env (recommended via uv)
uv venv --python 3.12 ~/progs/crewai-env
source ~/progs/crewai-env/bin/activate
uv pip install -r requirements.txt

# 2. Fire up LM Studio (load a model, start the server on :1234)
#    or point config.toml at your llama-server binary + GGUF.

# 3. Copy the template config and edit if needed
cp config.toml.example config.toml

# 4. Run
python swarm.py
```

## Configuration

All settings live in `config.toml`. Key sections:

- `[backend]` — `lm_studio`, `llama_cpp`, or `custom` OpenAI-compat URL.
- `[execution]` — `process` (`hierarchical`/`sequential`), retries, RPM, timeouts, context window.
- `[tools]` — web search provider and limits.
- `[paths]` — where projects get written.

See `config.toml.example` for the full template.

## Project layout

Each run lives under `projects/<name>/`:

```
projects/my-app/
├── _state.db        SQLite — tasks, shared state, search cache, metrics
├── _crew/*.md       Raw agent output (one file per task)
├── _logs/run.log    tail -f for live logs
└── src/             Extracted, ready-to-run code
```

## Presets

- **Python lib** — researcher → architect → coder → tester → docs
- **Web API** — architect → coder ║ devops (parallel) → tester → security → reviewer
- **CLI tool** — architect → coder → tester → docs
- **Refactor existing** — refactorer + tester + reviewer
- **Research + prototype** — researcher → architect → coder
- **Custom** — pick agents by hand

## License

[MIT](LICENSE).

## Acknowledgements

Ideas stolen, with thanks, from [overstory](https://github.com/jayminwest/overstory) (SQLite state, multi-tier watchdog) and the broader [awesome-agent-orchestrators](https://github.com/andyrewlee/awesome-agent-orchestrators) ecosystem.
