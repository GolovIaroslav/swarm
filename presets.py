"""Project-type presets — declarative pipelines of agent roles + task specs.

Each preset is a function `(goal: str) -> list[TaskSpec]`. presets.PIPELINES
maps the user-visible preset name to its builder.

`║` in ARCHITECTURE.md means "parallel via async_execution=True". In code,
that's just two TaskSpecs with overlapping `depends_on` and async_execution=True.
"""

from __future__ import annotations

from typing import Callable

from tasks import (
    TaskSpec,
    for_architect,
    for_coder,
    for_devops,
    for_docs,
    for_refactorer,
    for_researcher,
    for_reviewer,
    for_security,
    for_tester,
)


def python_lib(goal: str) -> list[TaskSpec]:
    """researcher -> architect -> coder -> tester -> docs"""
    return [
        for_researcher(goal),
        for_architect(goal),
        for_coder(goal),
        for_tester(goal),
        for_docs(goal),
    ]


def web_api(goal: str) -> list[TaskSpec]:
    """architect -> coder || devops (parallel) -> tester -> security -> reviewer"""
    arch = for_architect(goal)

    coder_spec = for_coder(goal)
    coder_spec.depends_on = ("architecture",)

    devops_spec = for_devops(goal)
    devops_spec.depends_on = ("architecture",)
    devops_spec.async_execution = True   # runs parallel with coder

    tester_spec = for_tester(goal)
    tester_spec.depends_on = ("implementation", "devops")

    sec_spec = for_security(goal)
    sec_spec.depends_on = ("testing",)

    rev_spec = for_reviewer(goal)
    rev_spec.depends_on = ("security",)

    return [arch, coder_spec, devops_spec, tester_spec, sec_spec, rev_spec]


def cli_tool(goal: str) -> list[TaskSpec]:
    """architect -> coder -> tester -> docs"""
    return [
        for_architect(goal),
        for_coder(goal),
        for_tester(goal),
        for_docs(goal),
    ]


def refactor_existing(goal: str) -> list[TaskSpec]:
    """refactorer -> tester -> reviewer over an existing src/ tree"""
    return [
        for_refactorer(goal),
        for_tester(goal),
        for_reviewer(goal),
    ]


def research_prototype(goal: str) -> list[TaskSpec]:
    """researcher -> architect -> coder"""
    return [
        for_researcher(goal),
        for_architect(goal),
        for_coder(goal),
    ]


def custom(goal: str, roles: list[str]) -> list[TaskSpec]:
    """Build a pipeline from a hand-picked list of agent roles."""
    _builders: dict[str, Callable[[str], TaskSpec]] = {
        "researcher": for_researcher,
        "architect":  for_architect,
        "coder":      for_coder,
        "tester":     for_tester,
        "reviewer":   for_reviewer,
        "devops":     for_devops,
        "security":   for_security,
        "docs":       for_docs,
        "refactorer": for_refactorer,
    }
    specs = []
    for role in roles:
        if role not in _builders:
            raise ValueError(f"Unknown role: {role!r}")
        specs.append(_builders[role](goal))
    return specs


PIPELINES: dict[str, Callable[[str], list[TaskSpec]]] = {
    "python_lib":          python_lib,
    "web_api":             web_api,
    "cli_tool":            cli_tool,
    "refactor_existing":   refactor_existing,
    "research_prototype":  research_prototype,
}
