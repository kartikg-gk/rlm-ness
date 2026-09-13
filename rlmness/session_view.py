"""A session read back in the terminal: its questions, their runs, what it kept.

Like the dashboard this only reads. It opens a session's state and the traces
its answers link to, and never touches a run. Textual is imported here, and
only `rlmness-viewlog --tui` reaches this module.
"""

from __future__ import annotations

from pathlib import Path

from rich.syntax import Syntax
from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import Horizontal, VerticalScroll
from textual.screen import Screen
from textual.widgets import DataTable, Footer, Static, Tree

from .dashboard import STYLE, RunPanes, _clip
from .events import RunTree
from .session import Session
from .viewlog import _locate, _records, _totals, _tree_of

BAR = 20

EXTRA = """
#session-head, #memory-head {
    height: 5;
    border: solid #21262d;
    border-title-color: #58a6ff;
    padding: 0 1;
}
#questions { height: 1fr; border: solid #21262d; }
#question-detail {
    height: 9;
    border: solid #21262d;
    border-title-color: #7d8590;
    padding: 0 1;
}
#memory { height: 1fr; layout: horizontal; }
#memory-names { width: 48; border: solid #21262d; }
#memory-detail {
    width: 1fr;
    border: solid #21262d;
    border-title-color: #7d8590;
    padding: 0 1;
}
"""


def _spend(run: dict | None) -> str:
    if run is None:
        return "—"
    return f"${run['cost']:.4f}" if run["priced"] else "unknown"


class RunScreen(RunPanes, Screen):
    """One question's run, drawn from its trace with the dashboard's panes."""

    BINDINGS = [
        ("escape", "app.pop_screen", "Back"),
        ("left_square_bracket", "step(-1)", "Previous question"),
        ("right_square_bracket", "step(1)", "Next question"),
        ("f", "follow", "Follow latest"),
        ("q", "app.quit", "Quit"),
    ]

    def __init__(self, tree: RunTree, heading: str, index: int | None = None):
        super().__init__()
        self._init_panes(tree)
        self.heading = heading
        self.index = index

    def compose(self) -> ComposeResult:
        yield Static(id="query")
        yield from self._compose_panes()
        yield Footer()

    def on_mount(self) -> None:
        self._setup_panes()
        self.query_one("#query").border_title = "RUN"
        self.query_one("#query", Static).update(self.heading)
        self._sync_tree()
        self._draw_detail()

    def on_tree_node_selected(self, event) -> None:
        self._select_node(event)

    def action_step(self, offset: int) -> None:
        if self.index is None:
            return
        self.app.open_run(self.index + offset, replace=True)


class SessionScreen(Screen):
    """Every question in order, with what each one's run cost."""

    BINDINGS = [
        ("m", "memory", "Session memory"),
        ("q", "app.quit", "Quit"),
    ]

    def compose(self) -> ComposeResult:
        yield Static(id="session-head")
        yield DataTable(id="questions")
        with VerticalScroll(id="question-detail"):
            yield Static(id="question-body")
        yield Footer()

    def on_mount(self) -> None:
        browser = self.app
        book = browser.book
        head = self.query_one("#session-head", Static)
        head.border_title = "SESSION"
        tokens = sum(run["tokens"] for run in browser.runs if run)
        priced = all(run["priced"] for run in browser.runs if run)
        cost = sum(run["cost"] for run in browser.runs if run)
        linked = sum(1 for run in browser.runs if run)
        unfinished = (
            f"\n[yellow]unfinished[/] {_clip(book.asking, 100)}" if book.asking else ""
        )
        head.update(
            f"[b]{browser.state_path.parent}[/b]  [dim]{browser.state_path.name} "
            f"v{book.version}[/dim]\n"
            f"{len(book.answered)} answered  ·  {len(book.variables)} variables  "
            f"{len(book.functions)} functions  {len(book.dropped)} dropped  ·  "
            f"{tokens} tokens  cost {f'${cost:.4f}' if priced else 'unknown'}  "
            f"[dim]across {linked} linked runs[/dim]{unfinished}"
        )
        table = self.query_one("#questions", DataTable)
        table.cursor_type = "row"
        table.add_columns("#", "question", "answer", "steps", "tokens", "cost", "share")
        largest = max((run["tokens"] for run in browser.runs if run), default=0)
        for number, (answered, run) in enumerate(zip(book.answered, browser.runs), 1):
            share = (
                "█" * max(1, round(BAR * run["tokens"] / largest))
                if run and largest
                else ("" if run else "not linked" if not answered.trace else "missing")
            )
            table.add_row(
                str(number),
                _clip(answered.question, 60),
                _clip(repr(answered.answer), 40),
                str(run["steps"]) if run else "—",
                str(run["tokens"]) if run else "—",
                _spend(run),
                share,
            )
        self.query_one("#question-detail").border_title = "Question"
        table.focus()
        self._describe(0)

    def _describe(self, index: int) -> None:
        body = self.query_one("#question-body", Static)
        answered = self.app.book.answered
        if not 0 <= index < len(answered):
            body.update("[dim]no questions answered yet[/dim]")
            return
        entry = answered[index]
        trace = self.app.traces[index]
        where = (
            str(trace) if trace else
            f"{entry.trace}  (missing: moved or deleted)" if entry.trace else "(not linked)"
        )
        body.update(
            Text.assemble(
                ("question  ", "#7d8590"), (str(entry.question), "#c9d1d9"), "\n",
                ("FINAL     ", "#7d8590"), (repr(entry.answer), "green"), "\n",
                ("trace     ", "#7d8590"), (where, "#58a6ff"), "\n",
                ("enter opens the run, m shows what the session kept", "#7d8590"),
            )
        )

    def on_data_table_row_highlighted(self, event) -> None:
        self._describe(event.cursor_row)

    def on_data_table_row_selected(self, event) -> None:
        self.app.open_run(event.cursor_row)

    def action_memory(self) -> None:
        self.app.push_screen(MemoryScreen())


class MemoryScreen(Screen):
    """What a resumed question inherits: kept names on the left, one in full on the right."""

    BINDINGS = [
        ("escape", "app.pop_screen", "Back"),
        ("q", "app.quit", "Quit"),
    ]

    def compose(self) -> ComposeResult:
        yield Static(id="memory-head")
        with Horizontal(id="memory"):
            yield DataTable(id="memory-names")
            with VerticalScroll(id="memory-detail"):
                yield Static(id="memory-body")
        yield Footer()

    def entries(self) -> list[tuple[str, str, str, object]]:
        """Committed values first, since those are what the model chose to keep."""
        book = self.app.book
        variables = sorted(
            book.variables.items(), key=lambda item: (not item[1].get("committed"), item[0])
        )
        found = [
            ("★" if meta.get("committed") else " ", "variable", name, meta)
            for name, meta in variables
        ]
        found += [(" ", "function", name, source) for name, source in sorted(book.functions.items())]
        found += [(" ", "dropped", name, reason) for name, reason in sorted(book.dropped.items())]
        return found

    def on_mount(self) -> None:
        book = self.app.book
        head = self.query_one("#memory-head", Static)
        head.border_title = "SESSION MEMORY"
        head.update(
            f"{len(book.variables)} variables  "
            f"({sum(1 for meta in book.variables.values() if meta.get('committed'))} committed)  "
            f"{len(book.functions)} functions  {len(book.dropped)} dropped\n"
            f"[dim]★ committed  ·  escape goes back[/dim]"
        )
        self._rows = self.entries()
        table = self.query_one("#memory-names", DataTable)
        table.cursor_type = "row"
        table.add_columns("", "kind", "name", "type")
        for mark, kind, name, value in self._rows:
            kind_of = value.get("type", "") if isinstance(value, dict) else ""
            table.add_row(mark, kind, name, kind_of)
        self.query_one("#memory-detail").border_title = "Detail"
        table.focus()
        self._show(0)

    def _show(self, index: int) -> None:
        body = self.query_one("#memory-body", Static)
        if not 0 <= index < len(self._rows):
            body.update("[dim]nothing kept[/dim]")
            return
        mark, kind, name, value = self._rows[index]
        if kind == "function":
            body.update(Syntax(str(value), "python", theme="github-dark",
                               background_color="#0d1117", word_wrap=True))
            return
        if kind == "dropped":
            body.update(Text.assemble((f"{name}\n\n", "b"), ("not kept: ", "#7d8590"), str(value)))
            return
        parts = [(f"{name}", "b"), (f"  {value.get('type', '')}", "#7d8590")]
        if value.get("committed"):
            parts.append(("  committed", "yellow"))
        parts.append("\n")
        if value.get("note"):
            parts += [("\nnote     ", "#7d8590"), (str(value["note"]), "#c9d1d9")]
        if value.get("comment"):
            parts += [("\ncomment  ", "#7d8590"), (str(value["comment"]), "#c9d1d9")]
        parts += [("\n\n", ""), (str(value.get("preview", "")), "#c9d1d9")]
        body.update(Text.assemble(*parts))

    def on_data_table_row_highlighted(self, event) -> None:
        if event.data_table.id == "memory-names":
            self._show(event.cursor_row)


class SessionBrowser(App):
    CSS = STYLE + EXTRA

    def __init__(self, state: Path | str):
        super().__init__()
        self.state_path = Path(state)
        self.book = Session.load(self.state_path)
        self.traces: list[Path | None] = []
        self.runs: list[dict | None] = []
        self._records: dict[Path, list[dict]] = {}
        for answered in self.book.answered:
            found = _locate(answered.trace, self.state_path) if answered.trace else None
            self.traces.append(found)
            self.runs.append(
                _totals(_tree_of(self._read(found), answered.run_id)) if found else None
            )

    def _read(self, path: Path) -> list[dict]:
        if path not in self._records:
            self._records[path] = _records(path)
        return self._records[path]

    def on_mount(self) -> None:
        self.push_screen(SessionScreen())

    def open_run(self, index: int, replace: bool = False) -> None:
        answered = self.book.answered
        if not 0 <= index < len(answered):
            return
        found = self.traces[index]
        if found is None:
            self.notify("this question has no trace to open", severity="warning")
            return
        entry = answered[index]
        tree = RunTree.replay(_tree_of(self._read(found), entry.run_id))
        tree.query = str(entry.question)
        heading = (
            f"[b][{index + 1}/{len(answered)}][/b] {_clip(entry.question, 150)}\n"
            f"[green]FINAL[/] {_clip(repr(entry.answer), 150)}  "
            f"[dim]· [ and ] move between questions, escape goes back[/dim]"
        )
        if replace:
            self.pop_screen()
        self.push_screen(RunScreen(tree, heading, index))


class TraceBrowser(App):
    CSS = STYLE

    def __init__(self, path: Path | str):
        super().__init__()
        self.path = Path(path)

    def on_mount(self) -> None:
        tree = RunTree.replay(_records(self.path))
        screen = RunScreen(tree, f"[b]{self.path}[/b]  [dim]escape or q quits[/dim]")
        screen.BINDINGS = [("escape", "app.quit", "Quit"), ("q", "app.quit", "Quit")]
        self.push_screen(screen)


def browse_session(state: Path | str) -> None:
    SessionBrowser(state).run()


def browse_trace(path: Path | str) -> None:
    TraceBrowser(path).run()
