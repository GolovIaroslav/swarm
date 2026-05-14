"""Task templates and the HANDOFF contract.

Every task description ends with the same instruction: produce a
`## HANDOFF (<=500 tokens)` section that the next agent will receive in lieu
of the raw output. This is the central token-saving mechanism (see CLAUDE.md
section "The 4 context-saving tricks").
"""

from __future__ import annotations

from dataclasses import dataclass


HANDOFF_INSTRUCTION = """
End your output with a `## HANDOFF` section, no more than 500 tokens, that
contains:
  - what you accomplished (one or two bullets)
  - paths of files you created or modified
  - anything the next agent should look out for (constraints, contracts, TODOs)
Do NOT dump full code into HANDOFF; the next agent will read files directly.
""".strip()


@dataclass
class TaskSpec:
    """Pre-Task description that build_task() consumes."""
    id: str
    agent_name: str              # one of agents.ROLES
    description: str
    expected_output: str
    depends_on: tuple[str, ...] = ()
    async_execution: bool = False
    output_file: str = ""        # relative to projects/<name>/_crew/


def build_task(spec: TaskSpec, agent, context_tasks: list):
    """Wrap a TaskSpec into a crewai.Task, appending HANDOFF_INSTRUCTION."""
    raise NotImplementedError("session 2")


# Specific task builders — keyed off agent role and the active preset.
def for_researcher(goal: str) -> TaskSpec: raise NotImplementedError("session 2")
def for_architect(goal: str) -> TaskSpec:  raise NotImplementedError("session 2")
def for_coder(goal: str) -> TaskSpec:      raise NotImplementedError("session 2")
def for_tester(goal: str) -> TaskSpec:     raise NotImplementedError("session 2")
def for_reviewer(goal: str) -> TaskSpec:   raise NotImplementedError("session 2")
def for_devops(goal: str) -> TaskSpec:     raise NotImplementedError("session 2")
def for_security(goal: str) -> TaskSpec:   raise NotImplementedError("session 2")
def for_docs(goal: str) -> TaskSpec:       raise NotImplementedError("session 2")
def for_refactorer(goal: str) -> TaskSpec: raise NotImplementedError("session 2")
