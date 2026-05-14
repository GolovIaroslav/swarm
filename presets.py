"""Project-type presets — declarative pipelines of agent roles + task specs.

Each preset is a function `(goal: str) -> list[TaskSpec]`. presets.PIPELINES
maps the user-visible preset name to its builder.

`║` in ARCHITECTURE.md means "parallel via async_execution=True". In code,
that's just two TaskSpecs with overlapping `depends_on` and async_execution=True.
"""

from __future__ import annotations

from typing import Callable

from tasks import TaskSpec


def python_lib(goal: str) -> list[TaskSpec]:
    """researcher -> architect -> coder -> tester -> docs"""
    raise NotImplementedError("session 2")


def web_api(goal: str) -> list[TaskSpec]:
    """architect -> (coder || devops) -> tester -> security -> reviewer"""
    raise NotImplementedError("session 2")


def cli_tool(goal: str) -> list[TaskSpec]:
    """architect -> coder -> tester -> docs"""
    raise NotImplementedError("session 2")


def refactor_existing(goal: str) -> list[TaskSpec]:
    """refactorer + tester + reviewer over an existing src/ tree"""
    raise NotImplementedError("session 2")


def research_prototype(goal: str) -> list[TaskSpec]:
    """researcher -> architect -> coder"""
    raise NotImplementedError("session 2")


def custom(goal: str, roles: list[str]) -> list[TaskSpec]:
    """Build a pipeline from a hand-picked list of agent roles."""
    raise NotImplementedError("session 2")


PIPELINES: dict[str, Callable[[str], list[TaskSpec]]] = {
    "python_lib":          python_lib,
    "web_api":             web_api,
    "cli_tool":            cli_tool,
    "refactor_existing":   refactor_existing,
    "research_prototype":  research_prototype,
}
