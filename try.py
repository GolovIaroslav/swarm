"""Acceptance test for session 2.

Runs a minimal architect -> coder crew on a toy goal ("write a hello() function"),
writes a real .py file to projects/_scratch/src/, leaves checkpoint rows in tasks.
No TUI, no monitor, no watchdog.

Usage: python try.py
"""

import time
from pathlib import Path

from crewai import Crew, Process

from agents import build
from backend import Backend
from config import load, projects_root
from extractor import extract
from state import TaskRow, open_store
from tasks import build_task, for_architect, for_coder
from tools import make_tools

GOAL = "Write a hello() function that returns the string 'hello world'."


def _scratch_run():
    cfg = load()

    # project paths
    root = projects_root(cfg)
    proj_dir = root / "_scratch"
    proj_dir.mkdir(parents=True, exist_ok=True)
    src_dir = proj_dir / "src"
    crew_dir = proj_dir / "_crew"
    crew_dir.mkdir(parents=True, exist_ok=True)

    db_path = proj_dir / "_state.db"

    with open_store(db_path) as store:
        # backend + LLM
        backend = Backend(cfg=cfg)
        backend.start()
        lm = backend.llm()
        print(f"[try] model: {backend.model_id}")

        # tools + agents
        tools = make_tools(proj_dir, store, cfg)
        arch_agent = build("architect", lm, tools, cfg)
        coder_agent = build("coder", lm, tools, cfg)

        # task specs -> crewai tasks
        arch_spec = for_architect(GOAL)
        coder_spec = for_coder(GOAL)

        arch_task = build_task(arch_spec, arch_agent, [], output_dir=crew_dir)
        coder_task = build_task(coder_spec, coder_agent, [arch_task], output_dir=crew_dir)

        # checkpoint: mark as running
        for spec, task in [(arch_spec, arch_task), (coder_spec, coder_task)]:
            store.upsert_task(TaskRow(
                id=spec.id, agent=spec.agent_name, status="pending",
                started_at=int(time.time()),
            ))

        # run
        crew = Crew(
            agents=[arch_agent, coder_agent],
            tasks=[arch_task, coder_task],
            process=Process.sequential,
            verbose=True,
        )

        print("[try] kicking off crew...")
        t0 = time.time()
        result = crew.kickoff()
        elapsed = time.time() - t0
        print(f"[try] crew done in {elapsed:.0f}s")

        # save coder output to _crew/
        raw_output = str(result)
        (crew_dir / "coder_raw.md").write_text(raw_output)

        # extract files (pass proj_dir so "src/foo.py" resolves correctly)
        extracted = extract(raw_output, proj_dir)
        print(f"[try] extracted {len(extracted)} file(s):")
        for f in extracted:
            print(f"  {f.path.relative_to(proj_dir)}  ({f.bytes_written} bytes)")

        # checkpoint: done
        store.upsert_task(TaskRow(
            id="architecture", agent="architect", status="done",
            finished_at=int(time.time()),
        ))
        store.upsert_task(TaskRow(
            id="implementation", agent="coder", status="done",
            finished_at=int(time.time()),
        ))

        tasks_in_db = store.all_tasks()
        print(f"[try] checkpoint rows: {[t.id for t in tasks_in_db]}")
        print("[try] DONE — check projects/_scratch/")


if __name__ == "__main__":
    _scratch_run()
