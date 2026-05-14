"""Extract code blocks from agent markdown output into real files on disk.

Agents are instructed to emit code like:

    ## relative/path/to/file.py
    ```python
    <code>
    ```

This module walks that markdown and writes each block under projects/<name>/src/.
Refuses to escape the project root. Run after every coder/tester/reviewer task.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class ExtractedFile:
    path: Path
    bytes_written: int
    language: str


def extract(markdown: str, dest_root: Path) -> list[ExtractedFile]:
    """Parse markdown and write each fenced code block to its named file.

    Filename comes from the heading immediately preceding the block (## name).
    Returns a list of written files. Skips blocks with no filename heading.
    """
    raise NotImplementedError("session 2")
