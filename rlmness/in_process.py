"""A runtime that shares this interpreter.

The cell runs here, in the process that started it, so a tool is the object
itself rather than text that rebuilds one. Lambdas, closures, builtins, live
handles and mutable data all work, and a tool may return anything because the
value never leaves.

The price is the whole of the isolation. Code the model writes runs with
everything this process can reach: it can read and change any state, touch the
filesystem and network, and take the process down with it. Use it where the
generated code is trusted — a fixture, a local experiment, a tool that must
hold a real connection — and use one of the process-backed runtimes otherwise.
"""

from __future__ import annotations

import asyncio
import ast
import threading
import io
import traceback
from contextlib import redirect_stdout
from typing import Any, Callable, Mapping

from .namespace import summarise
from .runtime import SESSION as _GUEST_SOURCE, CellOutcome


def _awaitable(function: Callable) -> Callable:
    """Keep the bridge contract the prompt describes.

    A bridge needs no round trip here, so it could be called outright. It stays
    awaited so one prompt describes every runtime and a cell written for one
    runs on the others.
    """

    async def bridge(*args, **kwargs):
        return function(*args, **kwargs)

    return bridge


class InProcessRuntime:
    #: Shares this interpreter, so it reaches whatever this process can.
    SEALED = False
    #: Tools arrive as objects, so nothing has to be rebuilt from text.
    NEEDS_SOURCE = False
    #: No process is started at all, so nothing here is worth capping.
    MAX_LIVE = 64

    def __init__(
        self,
        prompt,
        bridges: Mapping[str, Callable] = (),
        timeout: float | None = None,
        tools=(),
        session=None,
    ):
        # Kept for a uniform signature. There is no process to interrupt, so a
        # cell that never returns holds this thread; the process-backed
        # runtimes are the ones that can be timed out.
        self.timeout = timeout
        self._closed = False
        # Reset at the start of every execute() call. FINAL writes here
        # instead of raising, so code the model writes after FINAL runs the
        # same as it would in any other cell -- matching the other two
        # runtimes, rather than cutting the cell short.
        self._outcome = {"has_final": False, "final": None}
        self.namespace: dict[str, Any] = {
            "__name__": "__rlm_cell__",
            "PROMPT": prompt,
            "FINAL": self._final,
        }
        for name, function in dict(bridges).items():
            self.namespace[name] = _awaitable(function)
        # Executed into a private dictionary per runtime rather than imported.
        # There is no boundary to ship it across here, but the module keeps
        # state between steps -- committed names, harvested comments -- and a
        # single imported copy would share that state between every agent in
        # this interpreter, which the process-backed runtimes never do.
        self._session: dict[str, Any] = {}
        exec(_GUEST_SOURCE, self._session)
        # The caller's names, not the model's. A session must neither restore
        # over them nor sweep them up as though the model had written them.
        self._theirs = {tool.name for tool in tools}
        self.restore_failed = set()
        if session is not None:
            self.namespace["commit"] = self._session["commit"]
            self.restore_failed = set(
                self._session["restore"](self.namespace, session, self._theirs)
            )
        for tool in tools:
            self.namespace[tool.name] = tool.value
            # A stable handle for asserting identity without going through the
            # name the model sees.
            self.namespace[f"_tool_{tool.name}"] = tool.value

    def _final(self, value=None):
        self._outcome["has_final"] = True
        self._outcome["final"] = value

    def _drive(self, pending):
        """Run the cell's coroutine, even inside a loop that is already running.

        A child's cell runs inside its parent's await, on the parent's thread,
        so a second `asyncio.run` here refuses outright: a loop is already
        running on it. That denied a child every helper its parent had, and
        denied it as a cell error, so the parent read a traceback as though it
        were an answer and carried on with it.

        The nested case gets a loop of its own on a thread of its own, and
        this thread waits for it. The common case -- a root cell, no loop
        running -- is left exactly as it was.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(pending)

        outcome = {}

        def run():
            try:
                outcome["value"] = asyncio.run(pending)
            except BaseException as failure:  # carried back to the real cell
                outcome["error"] = failure

        worker = threading.Thread(target=run, daemon=True)
        worker.start()
        worker.join()
        if "error" in outcome:
            raise outcome["error"]
        return outcome.get("value")

    def execute(self, code: str) -> CellOutcome:
        buffer = io.StringIO()
        self._outcome["has_final"] = False
        self._outcome["final"] = None
        error = None
        try:
            compiled = compile(
                code, "<cell>", "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT
            )
            with redirect_stdout(buffer):
                pending = eval(compiled, self.namespace)
                if pending is not None:
                    self._drive(pending)
        except BaseException:
            error = traceback.format_exc()
        return CellOutcome(
            stdout=buffer.getvalue(),
            final=self._outcome["final"],
            has_final=self._outcome["has_final"],
            error=error,
        )

    def sweep(self, code: str | None = None) -> dict:
        return self._session["sweep"](self.namespace, code, self._theirs)

    def snapshot(self) -> list[dict]:
        """The same summary the other runtimes build, over the same names.

        There is no boundary to cross here, so the summariser is imported
        rather than shipped. It is still the summariser that runs, so one
        variable reads identically whichever runtime is underneath.
        """
        try:
            return summarise(self.namespace)
        except Exception:
            return []

    def close(self):
        self._closed = True
