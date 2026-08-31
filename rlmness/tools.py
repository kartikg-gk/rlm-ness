"""Custom tools, prepared on the host so a runtime can define them.

A tool becomes an ordinary Python function inside the cell namespace, called
the way any function is called. Nothing about a call crosses the process
boundary: only the function's source does, once, when the runtime starts.

The cost of that is what a tool may be. It has to be a plain function whose
source can be read and which stands on its own, because the runtime rebuilds
it from text rather than receiving the object. A lambda, a builtin, or a
function closing over something on the host cannot be rebuilt, and each is
refused here rather than failing later inside a cell.
"""

from __future__ import annotations

import inspect
import textwrap
from dataclasses import dataclass
from typing import Callable, Mapping

# Names the runtime binds itself. Handing one to a tool would either shadow a
# helper or be shadowed by it, and both are worse than refusing.
RESERVED = frozenset(
    {"llm", "rlm", "gather_llm", "gather_rlm", "FINAL", "PROMPT"}
)


class ToolError(Exception):
    pass


@dataclass(frozen=True)
class Tool:
    name: str
    signature: str
    doc: str
    source: str


UNAVAILABLE = (
    "source is unavailable. A tool has to be a plain function defined in a "
    "file — not a lambda, a builtin, or something defined in a REPL."
)


def _source_of(name: str, function: Callable) -> str:
    # A lambda's source is whatever line it appeared on, which is not a
    # definition the runtime can execute. Catch it before reading anything.
    if getattr(function, "__name__", "") == "<lambda>":
        raise ToolError(f"tool {name!r}: {UNAVAILABLE}")
    try:
        return textwrap.dedent(inspect.getsource(function))
    except (OSError, TypeError) as missing:
        raise ToolError(f"tool {name!r}: {UNAVAILABLE}") from missing


def _reject_closures(name: str, function: Callable) -> None:
    captured = getattr(function, "__code__", None)
    if captured is not None and captured.co_freevars:
        names = ", ".join(captured.co_freevars)
        raise ToolError(
            f"tool {name!r}: closes over {names}. The runtime rebuilds a tool "
            f"from its source, where a free variable of the host does not exist. "
            f"Pass the value as an argument or read it inside the function."
        )


def describe(tools: Mapping[str, Callable] | None) -> list[Tool]:
    """Check the tools and prepare what a runtime needs to define them."""
    if not tools:
        return []

    prepared = []
    for name, function in tools.items():
        if not str(name).isidentifier():
            raise ToolError(f"tool name {name!r} is not a Python identifier")
        if name in RESERVED:
            raise ToolError(
                f"tool name {name!r} is already bound by the runtime; choose another"
            )
        if not callable(function):
            raise ToolError(f"tool {name!r} is not callable")

        _reject_closures(name, function)
        source = _source_of(name, function)

        # The key is what the model calls, which need not be what the function
        # was defined as.
        defined_as = getattr(function, "__name__", name)
        if defined_as != name:
            source = f"{source}\n{name} = {defined_as}\n"

        prepared.append(
            Tool(
                name=name,
                signature=str(inspect.signature(function)),
                doc=inspect.getdoc(function) or "",
                source=source,
            )
        )
    return prepared
