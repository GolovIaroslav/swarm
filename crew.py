#!/usr/bin/env python3
"""
CrewAI мультиагентный кодинг под LM Studio
Запуск: python crew.py "твоя задача"
"""

import sys
import os
from crewai import Agent, Task, Crew, Process, LLM

# ─── LM Studio ────────────────────────────────────────────────────────────────
lm = LLM(
    model="openai/local",  # CrewAI требует префикс openai/
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",  # любая строка
    temperature=0.3,
)

# ─── Агенты ───────────────────────────────────────────────────────────────────
architect = Agent(
    role="Software Architect",
    goal="Decompose the task into clear implementation plan with file structure",
    backstory="Senior architect who plans before coding. Always thinks about structure, modularity and testability.",
    llm=lm,
    verbose=True,
    allow_delegation=False,
)

coder = Agent(
    role="Senior Python Developer",
    goal="Implement clean, working code based on the architect's plan",
    backstory="Expert developer who writes production-quality code. Follows the plan exactly, uses best practices.",
    llm=lm,
    verbose=True,
    allow_delegation=False,
)

tester = Agent(
    role="QA Engineer",
    goal="Write comprehensive pytest tests covering all functionality",
    backstory="QA expert who writes tests for every edge case. Aims for 90%+ coverage. Uses pytest fixtures and parametrize.",
    llm=lm,
    verbose=True,
    allow_delegation=False,
)

reviewer = Agent(
    role="Code Reviewer",
    goal="Review code and tests, find bugs, suggest improvements, write final README",
    backstory="Principal engineer who catches bugs before production. Reviews for correctness, security, and maintainability.",
    llm=lm,
    verbose=True,
    allow_delegation=False,
)

# ─── Задача из аргумента ──────────────────────────────────────────────────────
goal = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Create a REST API with FastAPI for a todo app with CRUD operations"

output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# ─── Таски ────────────────────────────────────────────────────────────────────
task_plan = Task(
    description=f"""
Analyse this task and create a detailed implementation plan:

TASK: {goal}

Your output must include:
1. File structure (list every file that needs to be created)
2. For each file: purpose and key functions/classes
3. Dependencies (requirements.txt content)
4. Data models and interfaces
5. API endpoints (if applicable)
""",
    expected_output="Detailed implementation plan with file structure, dependencies, and component descriptions",
    agent=architect,
    output_file=f"{output_dir}/01_plan.md",
)

task_code = Task(
    description=f"""
Implement the full application based on the architect's plan.

TASK: {goal}

Requirements:
- Write COMPLETE, RUNNABLE code (no placeholders, no '...')
- Every file from the plan must be implemented
- Include all imports
- Add docstrings to all functions and classes
- Handle errors properly
- Output each file as a separate code block with the filename as header

Format:
## filename.py
```python
<complete code>
```
""",
    expected_output="Complete implementation of all files with full working code",
    agent=coder,
    context=[task_plan],
    output_file=f"{output_dir}/02_code.md",
)

task_tests = Task(
    description=f"""
Write a comprehensive pytest test suite for the implemented code.

Requirements:
- Test every function and endpoint
- Include edge cases and error cases
- Use pytest fixtures
- Use parametrize for multiple test cases
- Aim for 90%+ coverage
- Tests must actually run (no mocks where real implementation can be used)
- Include a conftest.py if needed

Output all test files as code blocks with filenames.
""",
    expected_output="Complete pytest test suite covering all functionality",
    agent=tester,
    context=[task_plan, task_code],
    output_file=f"{output_dir}/03_tests.md",
)

task_review = Task(
    description=f"""
Review the implementation and tests. Then produce the final deliverable.

Your output must contain:

1. **Code Review** — bugs found, security issues, improvements made
2. **Final corrected code** — all files with any fixes applied
3. **README.md** — setup instructions, how to run, how to test, API docs if applicable

Be thorough. If you find bugs, fix them in the final code.
""",
    expected_output="Code review report + final corrected code + README.md",
    agent=reviewer,
    context=[task_plan, task_code, task_tests],
    output_file=f"{output_dir}/04_final.md",
)

# ─── Запуск ───────────────────────────────────────────────────────────────────
crew = Crew(
    agents=[architect, coder, tester, reviewer],
    tasks=[task_plan, task_code, task_tests, task_review],
    process=Process.sequential,
    verbose=True,
)

print(f"\n{'='*60}")
print(f"ЗАДАЧА: {goal}")
print(f"Результаты будут в папке: {output_dir}/")
print(f"{'='*60}\n")

result = crew.kickoff()

print(f"\n{'='*60}")
print("ГОТОВО! Файлы:")
for f in sorted(os.listdir(output_dir)):
    size = os.path.getsize(f"{output_dir}/{f}")
    print(f"  {output_dir}/{f}  ({size} bytes)")
print(f"{'='*60}")
