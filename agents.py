"""The 9-role agent pool.

Every agent is built with the mandatory knobs documented in CLAUDE.md:
  respect_context_window=True, max_iter, max_retry_limit, max_execution_time,
  max_rpm, allow_delegation=False (True only for hierarchical manager),
  verbose=True, llm=lm.

`build(name, llm, tools_by_name, cfg)` is the single entry point. Other
modules import roles by string name ("researcher", "architect", ...).
"""

from __future__ import annotations

from typing import Callable

from config import Config


ROLES = (
    "researcher",
    "architect",
    "coder",
    "tester",
    "reviewer",
    "devops",
    "security",
    "docs",
    "refactorer",
)

# Which tool names each role needs. tools.make_tools() returns a dict keyed
# by these names; agents pull only what they're listed for.
ROLE_TOOLS: dict[str, tuple[str, ...]] = {
    "researcher": ("web_search", "fetch_url", "get_state"),
    "architect":  ("get_state", "set_state"),
    "coder":      ("read_file", "write_file", "get_state"),
    "tester":     ("read_file", "write_file", "run_command"),
    "reviewer":   ("read_file", "write_file", "get_state"),
    "devops":     ("read_file", "write_file"),
    "security":   ("read_file", "web_search"),
    "docs":       ("read_file", "write_file"),
    "refactorer": ("read_file", "write_file"),
}


def build(name: str, llm, tools_by_name: dict, cfg: Config):
    """Construct one crewai.Agent for the given role name.

    Pulls role description from ROLE_PROMPTS, picks tools per ROLE_TOOLS.
    """
    raise NotImplementedError("session 2")


def manager(llm, cfg: Config):
    """Construct the hierarchical manager agent (allow_delegation=True)."""
    raise NotImplementedError("session 2")
