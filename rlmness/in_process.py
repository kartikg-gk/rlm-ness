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

from .runtime import CellOutcome


class _Answered(Exception):
    def __init__(self, value=None):
        super().__init__(value)
        self.value = value


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
    #: Tools arrive as objects, so nothing has to be rebuilt from text.
    NEEDS_SOURCE = False

    def __init__(
        self,
        prompt,
        bridges: Mapping[str, Callable] = (),
        timeout: float | None = None,
        tools=(),
    ):
        # Kept for a uniform signature. There is no process to interrupt, so a
        # cell that never returns holds this thread; the process-backed
        # runtimes are the ones that can be timed out.
        self.timeout = timeout
        self._closed = False
        self.namespace: dict[str, Any] = {
            "__name__": "__rlm_cell__",
            "PROMPT": prompt,
            "FINAL": self._final,
        }
        for name, function in dict(bridges).items():
            self.namespace[name] = _awaitable(function)
        for tool in tools:
            self.namespace[tool.name] = tool.value
            # A stable handle for asserting identity without going through the
            # name the model sees.
            self.namespace[f"_tool_{tool.name}"] = tool.value

    @staticmethod
    def _final(value=None):
        raise _Answered(value)

    def execute(self, code: str) -> CellOutcome:
        buffer = io.StringIO()
        final, has_final, error = None, False, None
        try:
            compiled = compile(
                code, "<cell>", "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT
            )
            with redirect_stdout(buffer):
                pending = eval(compiled, self.namespace)
                if pending is not None:
                    asyncio.run(pending)
        except _Answered as signal:
            has_final, final = True, signal.value
        except BaseException:
            error = traceback.format_exc()
        return CellOutcome(
            stdout=buffer.getvalue(),
            final=final,
            has_final=has_final,
            error=error,
        )

    def close(self):
        self._closed = True
