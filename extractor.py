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

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ExtractedFile:
    path: Path
    bytes_written: int
    language: str


# Match a heading that looks like a file path (has an extension or a slash)
_HEADING = re.compile(r"^#{1,3}\s+(.+)$", re.MULTILINE)
_FENCE_OPEN = re.compile(r"^```(\w*)$", re.MULTILINE)
_FENCE_CLOSE = re.compile(r"^```$", re.MULTILINE)

_PATH_LIKE = re.compile(r"^[\w./-]+\.\w+$")


def extract(markdown: str, dest_root: Path) -> list[ExtractedFile]:
    """Parse markdown and write each fenced code block to its named file.

    Filename comes from the heading immediately preceding the block (## name).
    Returns a list of written files. Skips blocks with no filename heading.
    """
    results: list[ExtractedFile] = []
    lines = markdown.splitlines()
    last_heading: str = ""
    i = 0

    while i < len(lines):
        line = lines[i]

        # Track headings that look like file paths
        m = _HEADING.match(line)
        if m:
            candidate = m.group(1).strip()
            if _PATH_LIKE.match(candidate):
                last_heading = candidate
            i += 1
            continue

        # Detect opening fence
        mf = _FENCE_OPEN.match(line)
        if mf:
            lang = mf.group(1) or ""
            # collect body until closing ```
            body_lines = []
            i += 1
            while i < len(lines):
                if lines[i].strip() == "```":
                    break
                body_lines.append(lines[i])
                i += 1
            i += 1  # skip closing fence

            if not last_heading:
                continue  # no filename heading — skip

            code = "\n".join(body_lines)
            if not code.strip():
                continue

            try:
                target = _safe_path(dest_root, last_heading)
            except ValueError:
                continue

            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(code, encoding="utf-8")
            results.append(ExtractedFile(
                path=target,
                bytes_written=len(code.encode()),
                language=lang,
            ))
            last_heading = ""  # consume heading
            continue

        i += 1

    return results


def _safe_path(root: Path, rel: str) -> Path:
    target = (root / rel).resolve()
    if not str(target).startswith(str(root.resolve())):
        raise ValueError(f"Path escapes dest_root: {rel}")
    return target
