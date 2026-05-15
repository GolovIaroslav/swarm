"""Task templates and the HANDOFF contract.

Every task description ends with the same instruction: produce a
`## HANDOFF (<=500 tokens)` section that the next agent will receive in lieu
of the raw output. This is the central token-saving mechanism (see CLAUDE.md
section "The 4 context-saving tricks").
"""

from __future__ import annotations

from dataclasses import dataclass, field

from crewai import Task


HANDOFF_INSTRUCTION = """
End your output with a `## HANDOFF` section, no more than 500 tokens, that
contains:
  - what you accomplished (one or two bullets)
  - paths of files you created or modified
  - anything the next agent should look out for (constraints, contracts, TODOs)
Do NOT dump full code into HANDOFF; the next agent will read files directly.
""".strip()

_CODE_FORMAT_INSTRUCTION = """
CRITICAL: When writing code, output EVERY file using EXACTLY this format:

## src/filename.py
```python
<code here>
```

Rules you MUST follow:
- The heading MUST use the exact relative path starting with src/ (e.g. `## src/main.py`)
- NEVER use generic headings like `## Solution`, `## Implementation`, `## Code`
- Every fenced code block MUST be immediately preceded by its path heading
- If you skip the path heading, the file will NOT be saved to disk
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


def build_task(spec: TaskSpec, agent, context_tasks: list, output_dir=None) -> Task:
    """Wrap a TaskSpec into a crewai.Task, appending HANDOFF_INSTRUCTION."""
    description = spec.description + "\n\n" + HANDOFF_INSTRUCTION

    kwargs = dict(
        description=description,
        expected_output=spec.expected_output,
        agent=agent,
        async_execution=spec.async_execution,
    )

    if context_tasks:
        kwargs["context"] = context_tasks

    if spec.output_file and output_dir:
        from pathlib import Path
        out_path = Path(output_dir) / spec.output_file
        out_path.parent.mkdir(parents=True, exist_ok=True)
        kwargs["output_file"] = str(out_path)

    return Task(**kwargs)


# ---------------------------------------------------------------------------
# Per-role task builders
# ---------------------------------------------------------------------------

def for_researcher(goal: str) -> TaskSpec:
    return TaskSpec(
        id="research",
        agent_name="researcher",
        description=(
            f"Research the following goal and gather all relevant technical information:\n\n"
            f"{goal}\n\n"
            "Find: relevant libraries and their current versions, best practices, "
            "common pitfalls, and any security considerations. "
            "Store key findings in shared state using set_state."
        ),
        expected_output=(
            "A concise technical brief covering libraries, approaches, and constraints. "
            "Ends with a ## HANDOFF section."
        ),
        output_file="01_research.md",
    )


def for_architect(goal: str) -> TaskSpec:
    return TaskSpec(
        id="architecture",
        agent_name="architect",
        description=(
            f"Design the software architecture to achieve:\n\n{goal}\n\n"
            "Define: module structure, data models, public interfaces, "
            "key dependencies, and any important design decisions. "
            "Use set_state to store: 'architecture_summary', 'data_models', "
            "'api_contracts' so downstream agents can fetch them without context bloat."
        ),
        expected_output=(
            "An architecture document with module layout, data models, and API contracts. "
            "Ends with a ## HANDOFF section."
        ),
        output_file="02_architecture.md",
    )


def for_coder(goal: str) -> TaskSpec:
    return TaskSpec(
        id="implementation",
        agent_name="coder",
        description=(
            f"Implement the code to achieve:\n\n{goal}\n\n"
            "Read the architecture from shared state (get_state). "
            f"{_CODE_FORMAT_INSTRUCTION}\n\n"
            "Write all files to src/. Include a main entry point if applicable."
        ),
        expected_output=(
            "All source files written under src/. "
            "Each file as a heading + fenced code block. "
            "Ends with a ## HANDOFF section listing created files."
        ),
        output_file="03_implementation.md",
    )


def for_tester(goal: str) -> TaskSpec:
    return TaskSpec(
        id="testing",
        agent_name="tester",
        description=(
            f"Write and run tests for the implementation of:\n\n{goal}\n\n"
            "Read source files from src/. "
            f"{_CODE_FORMAT_INSTRUCTION}\n\n"
            "Write tests to src/tests/ or test_*.py. "
            "Run them with run_command('python -m pytest src/ -v --tb=short'). "
            "Fix obvious failures. Report remaining issues in HANDOFF."
        ),
        expected_output=(
            "Test files written and pytest output showing pass/fail. "
            "Ends with a ## HANDOFF section summarising test results."
        ),
        output_file="04_testing.md",
    )


def for_reviewer(goal: str) -> TaskSpec:
    return TaskSpec(
        id="review",
        agent_name="reviewer",
        description=(
            f"Review the implementation for:\n\n{goal}\n\n"
            "Check: correctness, clarity, edge cases, architecture adherence. "
            "Apply small fixes in-place using write_file. "
            "Document significant issues in HANDOFF."
        ),
        expected_output=(
            "A code review report with issues found, fixes applied, and remaining concerns. "
            "Ends with a ## HANDOFF section."
        ),
        output_file="05_review.md",
    )


def for_devops(goal: str) -> TaskSpec:
    return TaskSpec(
        id="devops",
        agent_name="devops",
        description=(
            f"Create deployment configuration for:\n\n{goal}\n\n"
            f"{_CODE_FORMAT_INSTRUCTION}\n\n"
            "Write: Dockerfile, docker-compose.yml (if needed), and a CI pipeline "
            "(.github/workflows/ci.yml or equivalent). "
            "Make sure 'docker build' and 'docker run' work from project root."
        ),
        expected_output=(
            "Deployment files written. Ends with a ## HANDOFF section."
        ),
        output_file="06_devops.md",
    )


def for_security(goal: str) -> TaskSpec:
    return TaskSpec(
        id="security",
        agent_name="security",
        description=(
            f"Perform a security audit of the implementation for:\n\n{goal}\n\n"
            "Check: OWASP top-10, injection risks, hardcoded secrets, "
            "insecure dependencies. Search for CVEs in major dependencies. "
            "Report findings with severity (critical/high/medium/low) and fix suggestion."
        ),
        expected_output=(
            "A security report listing vulnerabilities with severity and remediation. "
            "Ends with a ## HANDOFF section."
        ),
        output_file="07_security.md",
    )


def for_docs(goal: str) -> TaskSpec:
    return TaskSpec(
        id="docs",
        agent_name="docs",
        description=(
            f"Write documentation for:\n\n{goal}\n\n"
            f"{_CODE_FORMAT_INSTRUCTION}\n\n"
            "Write: README.md (usage, installation, examples), "
            "and docstrings for all public functions/classes. "
            "The README should get a developer started in under 5 minutes."
        ),
        expected_output=(
            "README.md and updated source with docstrings. "
            "Ends with a ## HANDOFF section."
        ),
        output_file="08_docs.md",
    )


def for_refactorer(goal: str) -> TaskSpec:
    return TaskSpec(
        id="refactor",
        agent_name="refactorer",
        description=(
            f"Refactor the existing code to improve it for:\n\n{goal}\n\n"
            "Read existing files from src/. "
            "Reduce duplication, clarify names, flatten unnecessary abstractions. "
            "Run tests after each significant change to verify nothing broke. "
            f"{_CODE_FORMAT_INSTRUCTION}"
        ),
        expected_output=(
            "Refactored source files and passing tests. "
            "Ends with a ## HANDOFF section listing changed files."
        ),
        output_file="09_refactor.md",
    )
