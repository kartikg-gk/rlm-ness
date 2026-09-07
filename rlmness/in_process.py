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
import io
import traceback
from contextlib import redirect_stdout
from typing import Any, Callable, Mapping

from . import session_guest
from .namespace import summarise
from .runtime import CellOutcome


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
        # runtimes, and the reference, rather than cutting the cell short.
        self._outcome = {"has_final": False, "final": None}
        self.namespace: dict[str, Any] = {
            "__name__": "__rlm_cell__",
            "PROMPT": prompt,
            "FINAL": self._final,
        }
        for name, function in dict(bridges).items():
            self.namespace[name] = _awaitable(function)
        # Imported rather than shipped as source: there is no boundary to ship
        # it across here, and the same functions do the same work either way.
        self.restore_failed = set()
        if session is not None:
            self.namespace["commit"] = session_guest.commit
            self.restore_failed = set(session_guest.restore(self.namespace, session))
        for tool in tools:
            self.namespace[tool.name] = tool.value
            # A stable handle for asserting identity without going through the
            # name the model sees.
            self.namespace[f"_tool_{tool.name}"] = tool.value

    def _final(self, value=None):
        self._outcome["has_final"] = True
        self._outcome["final"] = value

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
                    asyncio.run(pending)
        except BaseException:
            error = traceback.format_exc()
        return CellOutcome(
            stdout=buffer.getvalue(),
            final=self._outcome["final"],
            has_final=self._outcome["has_final"],
            error=error,
        )

    def sweep(self, code: str | None = None) -> dict:
        return session_guest.sweep(self.namespace, code)

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
