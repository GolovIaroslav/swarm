# swarm

Local-first multi-agent TUI for autonomous coding. Built on CrewAI, talks to local LLM servers (LM Studio or llama.cpp) over the OpenAI-compatible API. Tell it what to build, walk away, come back to working code on disk.

![python](https://img.shields.io/badge/python-3.12-blue) ![license](https://img.shields.io/badge/license-MIT-green) ![status](https://img.shields.io/badge/status-beta-orange)

## Why

Most agent orchestrators out there (ccswarm, overstory, operator, agent-deck, and so on) are wired to Claude Code, Codex CLI and cloud APIs. None of them target a model behind LM Studio or llama.cpp. This one does, and it's built to run for hours without falling over.

It's also not a chat wrapper. It's a small TUI that walks you through setup, then a checkpointed pipeline of role-specialised agents that hand off through a SQLite store. Crash-resume works. Ctrl+C works.

## What it does

- One command, `python swarm.py`. A questionary TUI walks you through backend, project name, preset, goal and process mode. Or pass everything as CLI flags and skip the TUI.
- Four backend types: LM Studio, llama.cpp (we spawn `llama-server` for you, including any custom path / `LD_LIBRARY_PATH` / extra flags), any OpenAI-compatible `custom` URL, and an `api` mode that lets you point at any provider LiteLLM supports — OpenRouter, NVIDIA NIM, Groq, OpenAI, Anthropic, Gemini, Together, DeepInfra, and so on. One env var with the key, one model string, done.
- **Backend wizard, no TOML editing required.** The first time you run, a TUI wizard walks you through binary paths, env vars, model strings — your answers are saved as a named profile under `~/.config/swarm/backends.json`. Next time, pick from the list (`gemma_local`, `openrouter_claude`, ...). Manage with `swarm.py backends list / add / rm / show`.
- Per-role model override. Strong model for the architect, fast/cheap for the coder, etc.
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

# Optional — only [execution], [tools], [paths] sections matter now.
# The [backend] section is legacy; the TUI wizard replaces it.
cp config.toml.example config.toml

# Run. First time you'll get a wizard for your backend (LM Studio,
# llama.cpp with arbitrary paths and flags, or any cloud provider).
python swarm.py
```

On Windows and macOS the steps are the same, just swap the venv path.

### Backend examples the wizard handles

- **LM Studio** — boot LM Studio, load a model, start its local server. Pick "LM Studio" in the wizard, default URL is fine.
- **llama.cpp** — point at your `llama-server` binary, the GGUF file, context size, GPU layers and port. Add extra flags (`-fa on`, `-t 6`, `--jinja`, ...) and `LD_LIBRARY_PATH` for builds outside system paths.
- **Remote API** — pick a provider (OpenRouter, Anthropic, OpenAI, Groq, Gemini, NVIDIA NIM), fill in the model string and the env var that holds your API key.
- **Custom OpenAI-compatible URL** — vLLM, TGI, or anything that speaks `/v1/chat/completions`.

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

Two places things live:

- `~/.config/swarm/backends.json` — your named backends (managed by the wizard; never edit by hand).
- `config.toml` — everything else:
  - `[execution]` — `process` (`hierarchical` or `sequential`), `max_retry`, `max_rpm`, `task_timeout_minutes`, `context_window`, `max_iter`.
  - `[tools]` — web search provider, per-task cap, cache TTL.
  - `[paths]` — where projects get written.

The `[backend]` section in `config.toml.example` is legacy: it still works as a fallback, but new users get walked through the wizard instead.

## CLI

```
python swarm.py                          interactive: backend → project → preset → goal → run
python swarm.py run --backend NAME \     non-interactive run with a saved backend
        --project foo \
        --preset cli_tool \
        --goal "build a JSON-to-CSV CLI" \
        --process sequential -y
python swarm.py run --dry-run            print the plan and exit, no backend started

python swarm.py backends list            show saved backends
python swarm.py backends add             interactive wizard for a new backend
python swarm.py backends show NAME       print the full JSON entry
python swarm.py backends rm NAME         delete a saved backend

python swarm.py list                     show existing projects + checkpoint status
python swarm.py rm <name>                delete a project directory
python swarm.py presets                  show available pipelines
python swarm.py --help                   full usage
```

Flags for `run`: `--backend`, `--project`, `--preset`, `--goal`, `--roles a,b,c` (for `--preset custom`), `--process sequential|hierarchical`, `--resume`, `--no-resume`, `--no-monitor`, `--debug`, `-y/--yes`, `--dry-run`.

## Hardware

Tested on Arch Linux, 32 GB RAM, RTX 3060 6 GB VRAM. LM Studio with models in the 4-14B range (gemma, qwen2.5-coder, llama-3.1).

For hierarchical mode you want at least a 7B model with reliable JSON tool-calling. Smaller models work fine in sequential mode.

## What's planned

More presets (data science, API client). Better resume UX (skip individual tasks, not just the whole pipeline). Token-aware HANDOFF compaction. A backend wizard step that lets you set per-role overrides without dropping to TOML.

## What's not planned

Git worktrees and multi-branch coordination. Agent mailbox or mesh comms. MCP server integration. Sequential plus hierarchical plus SQLite shared state is enough for one local model.

## License

MIT. See `LICENSE`.

## Credits

Ideas borrowed from [overstory](https://github.com/jayminwest/overstory) (SQLite state, tiered watchdog) and the rest of the [awesome-agent-orchestrators](https://github.com/andyrewlee/awesome-agent-orchestrators) crowd. Built on top of [CrewAI](https://github.com/crewAIInc/crewAI).
