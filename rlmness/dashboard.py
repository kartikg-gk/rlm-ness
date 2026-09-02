"""A live view of a run, over the events the engine already emits.

This is a viewer and nothing else. It starts no agents, holds no policy and
makes no model calls: it subscribes a `RunTree` to the run and draws whatever
that tree says. Deleting this file would change nothing about how a run
behaves, which is the property that keeps the engine usable headless.

Textual is imported here and nowhere else, so the core install does not carry
it. `rlmness --dashboard` is the only thing that reaches this module.
"""

from __future__ import annotations

import threading
from typing import Any

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.widgets import DataTable, Footer, Static, Tree

from .events import DONE, FAILED, RUNNING, RunTree

SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

# Redraws are driven off a clock rather than off events. A fan-out can post
# hundreds of changes in a second and the terminal cannot show more than a few;
# polling a version counter collapses a burst into one repaint.
TICK = 0.1

MARKS = {RUNNING: "·", DONE: "✓", FAILED: "✗"}


def _cell(value, unknown: str = "—") -> str:
    return unknown if value is None else str(value)


def _money(value) -> str:
    return "unknown" if value is None else f"${value:.4f}"


def _clip(text: str, width: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= width else text[: width - 1] + "…"


class Dashboard(App):
    """The whole screen. One `RunTree`, four panes over it."""

    CSS = """
    Screen { layout: vertical; }

    #query {
        height: 3;
        border: round $accent;
        padding: 0 1;
        content-align: left middle;
    }

    #body { height: 1fr; layout: horizontal; }

    /* Narrow left column: the tree above, the namespace below. */
    #left { width: 34; layout: vertical; }
    #tree { height: 1fr; border: round $primary; padding: 0 1; }
    #namespace { height: 1fr; border: round $primary; padding: 0 1; }

    #right { width: 1fr; layout: vertical; }

    /* Compact by construction: a fixed height, so it cannot grow into the
       space the code and output need. */
    #info { height: 7; border: round $secondary; padding: 0 1; }

    #panes { height: 1fr; layout: horizontal; }
    #code { width: 1fr; border: round $success; padding: 0 1; }
    #output { width: 1fr; border: round $warning; padding: 0 1; }

    DataTable { height: 1fr; }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("f", "follow", "Follow latest"),
        ("ctrl+c", "quit", "Quit"),
    ]

    def __init__(self, tree: RunTree, runner=None, title: str = "rlm-ness"):
        super().__init__()
        self.state = tree
        self.runner = runner
        self._title = title
        self.selected: tuple[str, int | None] | None = None
        self.following = True
        self._agent_nodes: dict[str, Any] = {}
        self._step_nodes: dict[tuple[str, int], Any] = {}
        self._drawn_version = -1
        self._frame = 0
        self.result: Any = None
        self.failure: BaseException | None = None
        self.running = False

    # -- layout -----------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield Static(id="query")
        with Horizontal(id="body"):
            with Vertical(id="left"):
                yield Tree("run", id="tree")
                with VerticalScroll(id="namespace"):
                    yield DataTable(id="variables")
            with Vertical(id="right"):
                yield Static(id="info")
                with Horizontal(id="panes"):
                    with VerticalScroll(id="code"):
                        yield Static(id="code-body")
                    with VerticalScroll(id="output"):
                        yield Static(id="output-body")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#variables", DataTable)
        table.add_columns("name", "type", "size", "preview")
        table.cursor_type = "row"
        self.query_one("#tree", Tree).root.expand()
        # Draw once before the clock starts, so a run already part-way through
        # is on screen immediately rather than after the first tick.
        self.refresh_view()
        self.set_interval(TICK, self._tick)
        if self.runner is not None:
            self._start(self.runner)

    def _start(self, work) -> None:
        """Run the engine on its own thread; the UI thread only ever draws."""
        self.running = True

        def body():
            try:
                self.result = work()
            except BaseException as error:
                self.failure = error
            finally:
                self.running = False

        threading.Thread(target=body, daemon=True).start()

    # -- drawing ----------------------------------------------------------

    def _tick(self) -> None:
        self._frame += 1
        self.refresh_view()

    def refresh_view(self) -> None:
        """Draw what has changed since the last draw, and nothing else.

        The header moves every tick because the spinner has to. The rest is
        gated on a version counter, so a burst of events from a fan-out costs
        one repaint rather than one per event.

        The clock keeps ticking while the app is being torn down, by which
        point the widgets it would draw into are gone. A tick that finds
        nothing to draw into has nothing to say, so it returns rather than
        raising into the event loop.
        """
        version = self.state.version
        try:
            self._draw_query()
            if version == self._drawn_version:
                return
            self._drawn_version = version
            self._sync_tree()
            self._draw_detail()
        except NoMatches:
            return

    def _draw_query(self) -> None:
        status = self.state.status
        if status == RUNNING:
            mark = SPINNER[self._frame % len(SPINNER)]
            colour, word = "yellow", "running"
        elif status == DONE:
            mark, colour, word = "✓", "green", "done"
        else:
            mark, colour, word = "✗", "red", "failed"
        agents = len(self.state.agents)
        steps = sum(len(agent.steps) for agent in self.state.agents.values())
        follow = "follow" if self.following else "pinned"
        self.query_one("#query", Static).update(
            f"[b]{self._title}[/b]  [{colour}]{mark} {word}[/]  "
            f"agents {agents}  steps {steps}  [dim]({follow})[/dim]\n"
            f"[dim]{_clip(self.state.query, 200)}[/dim]"
        )

    def _sync_tree(self) -> None:
        """Add what is new, relabel what changed, and touch nothing else.

        Rebuilding would drop the expansion state and the cursor on every
        repaint. Nodes are keyed by run identity so two siblings in one
        `gather_rlm` land on separate branches however their steps interleave.
        """
        widget = self.query_one("#tree", Tree)
        for agent in self.state.ordered():
            node = self._agent_nodes.get(agent.run_id)
            if node is None:
                if agent.parent_run_id is None:
                    node = widget.root
                else:
                    parent = self._agent_nodes.get(agent.parent_run_id)
                    if parent is None:
                        continue
                    node = parent.add("", expand=True)
                self._agent_nodes[agent.run_id] = node
            node.set_label(self._agent_label(agent))
            node.data = (agent.run_id, None)
            for state in agent.steps:
                key = (agent.run_id, state.step)
                step_node = self._step_nodes.get(key)
                if step_node is None:
                    step_node = node.add_leaf("")
                    step_node.data = key
                    self._step_nodes[key] = step_node
                step_node.set_label(self._step_label(state))
        if self.following:
            latest = self._latest()
            if latest is not None and latest != self.selected:
                self.selected = latest

    def _agent_label(self, agent) -> str:
        mark = MARKS.get(agent.status, " ")
        name = "root" if agent.parent_run_id is None else f"agent {agent.run_id[:6]}"
        return f"{mark} {name}  d{agent.depth}"

    def _step_label(self, state) -> str:
        # No square brackets: a label is parsed as markup, so "[error]" would
        # be read as a style tag and disappear instead of printing.
        mark = MARKS.get(state.status, " ")
        note = "  error" if state.error else ""
        return f"{mark} step {state.step}{note}"

    def _latest(self) -> tuple[str, int] | None:
        """The newest step anywhere, so an idle viewer tracks the work."""
        newest = None
        for agent in self.state.ordered():
            for state in agent.steps:
                if newest is None or state.started is None:
                    newest = (agent.run_id, state.step, state.started)
                elif newest[2] is None or state.started >= newest[2]:
                    newest = (agent.run_id, state.step, state.started)
        return (newest[0], newest[1]) if newest else None

    def _draw_detail(self) -> None:
        self._draw_info()
        self._draw_code_and_output()
        self._draw_namespace()

    def _current(self):
        if self.selected is None:
            return None, None
        run_id, step = self.selected
        agent = self.state.agents.get(run_id)
        if agent is None:
            return None, None
        if step is None:
            return agent, None
        return agent, self.state.find_step(run_id, step)

    def _draw_info(self) -> None:
        agent, state = self._current()
        panel = self.query_one("#info", Static)
        if agent is None:
            panel.update("[dim]no step selected[/dim]")
            return
        usage = state.usage if state is not None else agent.usage
        where = f"step {state.step}" if state is not None else "agent"
        status = state.status if state is not None else agent.status
        if agent.status == DONE and agent.result is not None:
            outcome = f"[green]result[/] {_clip(repr(agent.result), 70)}"
        elif agent.status == FAILED:
            outcome = f"[red]failed[/] {_clip(agent.error or '', 70)}"
        else:
            outcome = f"[yellow]{agent.status}[/]"
        panel.update(
            f"{self._agent_label(agent)}  ·  {where} [{status}]  ·  {agent.model}\n"
            f"{outcome}\n"
            f"prompt {usage.prompt_tokens}  completion {usage.completion_tokens}  "
            f"total {usage.total_tokens}\n"
            f"cached {_cell(usage.cached_tokens)}  "
            f"reasoning {_cell(usage.reasoning_tokens)}  "
            f"cost {_money(usage.cost)}"
        )

    def _draw_code_and_output(self) -> None:
        _, state = self._current()
        code = self.query_one("#code-body", Static)
        output = self.query_one("#output-body", Static)
        if state is None:
            code.update("[dim]—[/dim]")
            output.update("[dim]—[/dim]")
            return
        # Code and output are read off one StepState, so a repaint can never
        # pair one agent's code with another agent's output.
        code.update(state.code or "[dim]waiting for the model…[/dim]")
        output.update(state.output or "[dim]—[/dim]")

    def _draw_namespace(self) -> None:
        _, state = self._current()
        table = self.query_one("#variables", DataTable)
        table.clear()
        if state is None:
            return
        for row in state.variables:
            table.add_row(
                row.get("name", ""),
                row.get("type", ""),
                _cell(row.get("size")),
                _clip(row.get("preview", ""), 60),
            )

    # -- input ------------------------------------------------------------

    def on_tree_node_selected(self, event) -> None:
        data = event.node.data
        if data is None:
            return
        # Choosing a step pins the view: an operator reading one step should
        # not have it yanked away by a sibling finishing.
        self.selected = data
        self.following = False
        self._draw_detail()

    def action_follow(self) -> None:
        self.following = not self.following
        self._drawn_version = -1


def show(tree: RunTree, runner=None, title: str = "rlm-ness") -> Dashboard:
    """Run the dashboard until the operator quits, and hand it back."""
    app = Dashboard(tree, runner=runner, title=title)
    app.run()
    return app
