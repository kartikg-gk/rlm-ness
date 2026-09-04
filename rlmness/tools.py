"""Custom tools, prepared for whichever runtime will hold them.

A tool becomes an ordinary name in the cell namespace — a function called the
way any function is called, or a value read the way any value is read. Nothing
about using one crosses a process boundary.

What a tool may be depends on where the namespace lives. A runtime sharing this
interpreter receives the object itself, so anything goes: a lambda, a closure, a
builtin, a live handle. A runtime in its own process cannot be handed an object
at all, only the text that rebuilds one, and each thing that text cannot carry
is refused here rather than failing later inside a cell.
"""

from __future__ import annotations

import inspect
import json
import textwrap
from dataclasses import dataclass
from typing import Any, Callable, Mapping

# Names the runtime binds itself. Handing one to a tool would either shadow a
# helper or be shadowed by it, and both are worse than refusing.
RESERVED = frozenset({"llm", "rlm", "gather_llm", "gather_rlm", "FINAL", "PROMPT"})

UNAVAILABLE = (
    "source is unavailable. For a runtime in its own process a tool has to be "
    "a plain function defined in a file — not a lambda, a builtin, or something "
    "defined in a REPL."
)


class ToolError(Exception):
    pass


@dataclass(frozen=True)
class Tool:
    name: str
    value: Any
    description: str
    signature: str
    callable: bool
    #: Text that rebuilds the tool. None where the runtime takes the object.
    source: str | None = None


def _unpack(entry: Any) -> tuple[Any, str | None]:
    """Accept a bare value or {"tool": ..., "description": ...}."""
    if isinstance(entry, Mapping) and "tool" in entry:
        return entry["tool"], entry.get("description")
    return entry, None


def _reject_closures(name: str, function: Callable) -> None:
    code = getattr(function, "__code__", None)
    if code is not None and code.co_freevars:
        names = ", ".join(code.co_freevars)
        raise ToolError(
            f"tool {name!r}: closes over {names}. A runtime in its own process "
            f"rebuilds a tool from its source, where a free variable of the "
            f"caller does not exist. Pass the value as an argument, read it "
            f"inside the function, or use a runtime that shares this process."
        )


def _function_source(name: str, function: Callable) -> str:
    # A lambda's source is whatever line it appeared on, which is not a
    # definition a runtime can execute. Catch it before reading anything.
    if getattr(function, "__name__", "") == "<lambda>":
        raise ToolError(f"tool {name!r}: {UNAVAILABLE}")
    _reject_closures(name, function)
    try:
        text = textwrap.dedent(inspect.getsource(function))
    except (OSError, TypeError) as missing:
        raise ToolError(f"tool {name!r}: {UNAVAILABLE}") from missing

    # The key is what the model calls, which need not be what the function was
    # defined as.
    defined_as = getattr(function, "__name__", name)
    return text if defined_as == name else f"{text}\n{name} = {defined_as}\n"


def _value_source(name: str, value: Any) -> str:
    try:
        return f"{name} = {json.dumps(value)}\n"
    except (TypeError, ValueError) as unserialisable:
        raise ToolError(
            f"tool {name!r}: a {type(value).__name__} cannot be carried as JSON, "
            f"so a runtime in its own process cannot be given it. Use plain data, "
            f"or a runtime that shares this process."
        ) from unserialisable


def describe(
    tools: Mapping[str, Any] | None, *, need_source: bool = True
) -> list[Tool]:
    """Check the tools and prepare what a runtime needs to install them.

    `need_source` is what the runtime asks for. A runtime rebuilding tools from
    text needs it and accepts less; one taking the objects does not and accepts
    anything.
    """
    if not tools:
        return []

    prepared = []
    for name, entry in tools.items():
        value, described = _unpack(entry)

        if not str(name).isidentifier():
            raise ToolError(f"tool name {name!r} is not a Python identifier")
        if name in RESERVED:
            raise ToolError(
                f"tool name {name!r} is already bound by the runtime; choose another"
            )

        is_callable = callable(value)
        if is_callable:
            signature = str(inspect.signature(value))
            description = described or inspect.getdoc(value) or ""
        else:
            signature = ""
            description = described or ""

        source = None
        if need_source:
            source = (
                _function_source(name, value)
                if is_callable
                else _value_source(name, value)
            )

        prepared.append(
            Tool(
                name=name,
                value=value,
                description=description,
                signature=signature,
                callable=is_callable,
                source=source,
            )
        )
    return prepared
