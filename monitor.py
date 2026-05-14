"""Rich live monitor — the dashboard the user watches while crew runs.

Layout (see ARCHITECTURE.md):
  header  — project name | model | uptime | ctx % | rpm
  tasks   — per-task status & duration
  stats   — tokens in/out, files written, searches, retries
  log     — tail of _logs/run.log (~20 lines)
  footer  — q/p/r shortcuts

Runs in a background thread via threading.Thread. Reads everything from the
Store + log file — pulls nothing from the LLM directly.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Optional

from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from config import Config
from state import Store


_STATUS_ICONS = {
    "done":    "✓",
    "running": "⚙",
    "pending": "⏳",
    "failed":  "✗",
}

_STATUS_STYLES = {
    "done":    "green",
    "running": "yellow",
    "pending": "dim",
    "failed":  "red bold",
}


class Monitor:
    def __init__(
        self,
        store: Store,
        log_path: Path,
        project_name: str,
        model_id: str,
        cfg: Optional[Config] = None,
    ):
        self.store = store
        self.log_path = log_path
        self.project_name = project_name
        self.model_id = model_id
        self.cfg = cfg
        self._start_time: float = time.time()

    def render(self) -> Layout:
        """Build the rich.layout.Layout tree for the current frame."""
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="log", size=12),
            Layout(name="footer", size=3),
        )
        layout["body"].split_row(
            Layout(name="tasks"),
            Layout(name="stats", minimum_size=30),
        )

        layout["header"].update(self._render_header())
        layout["tasks"].update(self._render_tasks())
        layout["stats"].update(self._render_stats())
        layout["log"].update(self._render_log())
        layout["footer"].update(self._render_footer())

        return layout

    def run(self, stop_event: threading.Event) -> threading.Thread:
        """Spawn a background daemon thread running rich.live.Live.

        Returns the thread so callers can join() it on shutdown.
        """
        self._start_time = time.time()

        def _live_loop() -> None:
            with Live(
                self.render(),
                refresh_per_second=2,
                screen=False,
                transient=False,
            ) as live:
                while not stop_event.is_set():
                    try:
                        live.update(self.render())
                    except Exception:
                        pass
                    time.sleep(0.5)

        t = threading.Thread(target=_live_loop, daemon=True, name="monitor")
        t.start()
        return t

    # ------------------------------------------------------------------
    def _render_header(self) -> Panel:
        elapsed = int(time.time() - self._start_time)
        h, m, s = elapsed // 3600, (elapsed % 3600) // 60, elapsed % 60
        uptime = f"{h}h {m:02d}m" if h else f"{m}m {s:02d}s"

        # context usage: absolute numbers + percent, coloured by severity
        tokens_in = int(self.store.get_metric("tokens_in"))
        limit = self.cfg.execution.context_window if self.cfg else 0
        if limit > 0:
            pct = tokens_in / limit * 100
            ctx_str = f"ctx {tokens_in:,}/{limit:,} ({pct:.0f}%)"
            if pct >= 95:
                ctx_color = "red bold"
            elif pct >= 75:
                ctx_color = "yellow bold"
            else:
                ctx_color = "bold"
        else:
            ctx_str = f"ctx {tokens_in:,}"
            ctx_color = "bold"

        # rpm = completed LLM calls / elapsed minutes
        elapsed_min = max((time.time() - self._start_time) / 60, 0.01)
        requests = self.store.get_metric("llm_requests")
        rpm = requests / elapsed_min

        header = Text()
        header.append(f" {self.project_name}", style="bold cyan")
        header.append("  │  ", style="dim")
        header.append(self.model_id, style="bold")
        header.append("  │  ", style="dim")
        header.append(uptime, style="bold")
        header.append("  │  ", style="dim")
        header.append(ctx_str, style=ctx_color)
        header.append("  │  ", style="dim")
        header.append(f"rpm {rpm:.1f}", style="bold")
        return Panel(header, style="cyan")

    def _render_tasks(self) -> Panel:
        table = Table(box=None, show_header=True, header_style="bold", expand=True)
        table.add_column("", width=2)
        table.add_column("Agent")
        table.add_column("Task")
        table.add_column("Duration", justify="right")

        try:
            rows = self.store.all_tasks()
        except Exception:
            rows = []

        for row in rows:
            icon = _STATUS_ICONS.get(row.status, "?")
            style = _STATUS_STYLES.get(row.status, "")

            if row.status == "running" and row.started_at:
                secs = int(time.time() - row.started_at)
                dur = f"{secs // 60}m {secs % 60:02d}s"
            elif row.started_at and row.finished_at:
                secs = row.finished_at - row.started_at
                dur = f"{secs // 60}m {secs % 60:02d}s"
            else:
                dur = ""

            table.add_row(
                Text(icon, style=style),
                Text(row.agent, style=style),
                Text(row.id),
                Text(dur, style=style),
            )

        return Panel(table, title="TASKS", border_style="blue")

    def _render_stats(self) -> Panel:
        try:
            tokens_in  = int(self.store.get_metric("tokens_in"))
            tokens_out = int(self.store.get_metric("tokens_out"))
            files_made = int(self.store.get_metric("files_made"))
            searches   = int(self.store.get_metric("searches"))
            retries    = int(self.store.get_metric("retries"))
        except Exception:
            tokens_in = tokens_out = files_made = searches = retries = 0

        lines = [
            f"Tokens IN:   {tokens_in:,}",
            f"Tokens OUT:  {tokens_out:,}",
            f"Files made:  {files_made}",
            f"Searches:    {searches}",
            f"Retries:     {retries}",
        ]
        return Panel("\n".join(lines), title="STATS", border_style="blue")

    def _render_log(self) -> Panel:
        text = ""
        if self.log_path.exists():
            try:
                lines = self.log_path.read_text(
                    encoding="utf-8", errors="replace"
                ).splitlines()
                text = "\n".join(lines[-10:])
            except OSError:
                text = "(log unavailable)"
        return Panel(text or "(no log yet)", title="LOG (last 10 lines)", border_style="dim")

    def _render_footer(self) -> Panel:
        from rich.markup import escape
        log_hint = f" │  log: {self.log_path}" if self.log_path else ""
        content = escape("[q]uit (Ctrl+C)  [h]elp: see README  [d]ebug: " + str(self.log_path))
        return Panel(content, style="dim")


def plain_stdout_monitor(
    store: Store,
    log_path: Path,
    stop_event: Optional[threading.Event] = None,
) -> None:
    """Fallback when `[ui].live_monitor = false`. Prints task transitions only."""
    known: dict[str, str] = {}
    while stop_event is None or not stop_event.is_set():
        try:
            for row in store.all_tasks():
                if known.get(row.id) != row.status:
                    known[row.id] = row.status
                    icon = _STATUS_ICONS.get(row.status, "?")
                    print(f"  {icon} [{row.agent}] {row.id}: {row.status}", flush=True)
        except Exception:
            pass
        time.sleep(5)
