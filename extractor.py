"""Extract code blocks from agent markdown output into real files on disk.

Agents are instructed to emit code like:

    ## relative/path/to/file.py
    ```python
    <code>
    ```

This module walks that markdown and writes each block under projects/<name>/src/.
Refuses to escape the project root. Run after every coder/tester/reviewer task.

Blocks without a path heading fall back to src/snippet_N.<ext>.
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

_LANG_EXT: dict[str, str] = {
    "python": "py", "py": "py",
    "javascript": "js", "js": "js", "jsx": "jsx",
    "typescript": "ts", "ts": "ts", "tsx": "tsx",
    "shell": "sh", "bash": "sh", "sh": "sh", "zsh": "sh",
    "rust": "rs",
    "go": "go",
    "java": "java",
    "c": "c",
    "cpp": "cpp", "c++": "cpp",
    "html": "html",
    "css": "css",
    "json": "json",
    "yaml": "yaml", "yml": "yaml",
    "toml": "toml",
    "sql": "sql",
}


def _lang_to_ext(lang: str) -> str:
    return _LANG_EXT.get(lang.lower(), "txt")


def extract(markdown: str, dest_root: Path) -> list[ExtractedFile]:
    """Parse markdown and write each fenced code block to its named file.

    Filename comes from the heading immediately preceding the block (## name).
    Falls back to src/snippet_N.<ext> when no path heading is present.
    Returns a list of written files.
    """
    results: list[ExtractedFile] = []
    lines = markdown.splitlines()
    last_heading: str = ""
    snippet_n = 0
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
            body_lines = []
            i += 1
            while i < len(lines):
                if lines[i].strip() == "```":
                    break
                body_lines.append(lines[i])
                i += 1
            i += 1  # skip closing fence

            code = "\n".join(body_lines)
            if not code.strip():
                continue

            if last_heading:
                rel = last_heading
                last_heading = ""  # consume heading
            else:
                snippet_n += 1
                ext = _lang_to_ext(lang)
                rel = f"src/snippet_{snippet_n}.{ext}"

            try:
                target = _safe_path(dest_root, rel)
            except ValueError:
                continue

            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(code, encoding="utf-8")
            results.append(ExtractedFile(
                path=target,
                bytes_written=len(code.encode()),
                language=lang,
            ))
            continue

        i += 1

    return results


def _safe_path(root: Path, rel: str) -> Path:
    target = (root / rel).resolve()
    if not str(target).startswith(str(root.resolve())):
        raise ValueError(f"Path escapes dest_root: {rel}")
    return target
