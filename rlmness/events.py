"""What a run is doing, while it is doing it.

The journal answers "what happened" after the fact. A live view needs the same
information as it arrives, split finer: a step that has asked the model but not
yet run its code is a state the journal never records, because by the time a
step is written it is over.

Nothing here imports a UI. `RunTree` is a plain object that folds events into
the shape a viewer wants, so the same events drive a dashboard, a test, or
nothing at all. The engine holds a sink and calls methods on it if they exist,
which is how a `Journal` that knows only `step` and `final` keeps working
unchanged beside a sink that wants everything.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

from .providers import Spend, combine

RUNNING = "running"
DONE = "done"
FAILED = "failed"


def emit(sink, name: str, /, **fields) -> None:
    """Tell the sink, if the sink cares.

    A sink is anything with some of these methods. One that predates an event
    simply does not have it, and silence is the right response — an older sink
    is not broken, it is narrower.
    """
    if sink is None:
        return
    method = getattr(sink, name, None)
    if method is None:
        return
    method(**fields)


@dataclass
class StepState:
    step: int
    status: str = RUNNING
    code: str | None = None
    output: str = ""
    error: bool = False
    usage: Spend = field(default_factory=Spend)
    variables: list[dict] = field(default_factory=list)
    started: str | None = None
    ended: str | None = None

    @property
    def label(self) -> str:
        mark = {RUNNING: "·", DONE: " ", FAILED: "!"}.get(self.status, " ")
        return f"{mark} step {self.step}"


@dataclass
class AgentState:
    """One agent: its steps, its place in the tree, and how it ended."""

    run_id: str
    parent_run_id: str | None
    depth: int
    model: str = ""
    instruction: str | None = None
    prompt_type: str = ""
    prompt_size: int = 0
    status: str = RUNNING
    result: Any = None
    error: str | None = None
    steps: list[StepState] = field(default_factory=list)
    children: list[str] = field(default_factory=list)

    def step(self, number: int) -> StepState:
        for existing in self.steps:
            if existing.step == number:
                return existing
        created = StepState(step=number)
        self.steps.append(created)
        return created

    @property
    def usage(self) -> Spend:
        total = Spend()
        for one in self.steps:
            total = combine(total, one.usage)
        return total

    @property
    def label(self) -> str:
        name = "root" if self.parent_run_id is None else f"agent {self.run_id[:6]}"
        mark = {RUNNING: "·", DONE: "✓", FAILED: "✗"}.get(self.status, " ")
        return f"{mark} {name} (d{self.depth})"


class RunTree:
    """Every agent in one run, keyed by the identity it was given.

    Keyed by `run_id` and never by depth. Two children of one `gather_rlm`
    share a depth and interleave their steps; only their own identity keeps
    them apart, so that is the only thing indexed on. A lock guards the maps
    because a fan-out writes from one thread per child.
    """

    def __init__(self):
        self.agents: dict[str, AgentState] = {}
        self.root_id: str | None = None
        self.query: str = ""
        self.status: str = RUNNING
        self._lock = threading.RLock()
        self._version = 0

    @property
    def version(self) -> int:
        """Bumped on every change, so a viewer can skip an unchanged redraw."""
        with self._lock:
            return self._version

    def _touch(self):
        self._version += 1

    def reset(self) -> None:
        """Forget the previous run, keeping the version counter moving.

        The counter is not reset with the rest: a viewer decides whether to
        redraw by comparing against the last version it drew, so restarting
        the count at zero would read as "nothing has changed".
        """
        with self._lock:
            self.agents.clear()
            self.root_id = None
            self.query = ""
            self.status = RUNNING
            self._touch()

    # -- sink methods: the engine calls these ------------------------------

    def run_started(self, *, run_id, parent_run_id, depth, model, instruction,
                    prompt_type, prompt_size):
        with self._lock:
            agent = AgentState(
                run_id=run_id,
                parent_run_id=parent_run_id,
                depth=depth,
                model=model,
                instruction=instruction,
                prompt_type=prompt_type,
                prompt_size=prompt_size,
            )
            self.agents[run_id] = agent
            if parent_run_id is None:
                self.root_id = run_id
            elif parent_run_id in self.agents:
                self.agents[parent_run_id].children.append(run_id)
            self._touch()

    def step_started(self, *, run_id, step, started=None):
        with self._lock:
            agent = self.agents.get(run_id)
            if agent is None:
                return
            state = agent.step(step)
            state.status = RUNNING
            state.started = started
            self._touch()

    def code_generated(self, *, run_id, step, code):
        with self._lock:
            agent = self.agents.get(run_id)
            if agent is None:
                return
            agent.step(step).code = code
            self._touch()

    def output_received(self, *, run_id, step, output, error):
        with self._lock:
            agent = self.agents.get(run_id)
            if agent is None:
                return
            state = agent.step(step)
            state.output = output
            state.error = bool(error)
            self._touch()

    def step_completed(self, *, run_id, step, usage, error=False, ended=None):
        with self._lock:
            agent = self.agents.get(run_id)
            if agent is None:
                return
            state = agent.step(step)
            state.usage = usage
            state.error = bool(error)
            state.status = FAILED if error else DONE
            state.ended = ended
            self._touch()

    def namespace_changed(self, *, run_id, step, variables):
        with self._lock:
            agent = self.agents.get(run_id)
            if agent is None:
                return
            agent.step(step).variables = list(variables)
            self._touch()

    def run_completed(self, *, run_id, result):
        with self._lock:
            agent = self.agents.get(run_id)
            if agent is not None:
                agent.status = DONE
                agent.result = result
            if run_id == self.root_id:
                self.status = DONE
            self._touch()

    def run_failed(self, *, run_id, error):
        with self._lock:
            agent = self.agents.get(run_id)
            if agent is not None:
                agent.status = FAILED
                agent.error = error
            if run_id == self.root_id:
                self.status = FAILED
            self._touch()

    # -- reading -----------------------------------------------------------

    def children_of(self, run_id: str) -> list[AgentState]:
        with self._lock:
            agent = self.agents.get(run_id)
            if agent is None:
                return []
            return [self.agents[child] for child in agent.children if child in self.agents]

    def ordered(self) -> list[AgentState]:
        """Depth-first from the root, parents before their children."""
        with self._lock:
            if self.root_id is None:
                return []
            found: list[AgentState] = []
            stack = [self.root_id]
            while stack:
                current = stack.pop()
                agent = self.agents.get(current)
                if agent is None:
                    continue
                found.append(agent)
                stack.extend(reversed(agent.children))
            return found

    @property
    def usage(self) -> Spend:
        with self._lock:
            total = Spend()
            for agent in self.agents.values():
                total = combine(total, agent.usage)
            return total

    def find_step(self, run_id: str, step: int) -> StepState | None:
        with self._lock:
            agent = self.agents.get(run_id)
            if agent is None:
                return None
            for existing in agent.steps:
                if existing.step == step:
                    return existing
            return None


class Broadcast:
    """Send every event to several sinks.

    The journal and a live view want the same events for different reasons,
    and neither should have to know the other exists. A sink that raises is
    dropped rather than allowed to take the run down with it: a viewer is not
    worth failing a run over.
    """

    def __init__(self, *sinks):
        self.sinks = [sink for sink in sinks if sink is not None]

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)

        def fan(**fields):
            for sink in self.sinks:
                try:
                    emit(sink, name, **fields)
                except Exception:
                    continue

        return fan
