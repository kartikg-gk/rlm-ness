"""The subprocess that executes cells."""

import ast
import asyncio
import io
import json
import sys
import traceback
from contextlib import redirect_stdout

_OUT = sys.__stdout__
_IN = sys.__stdin__


def _write(obj):
    _OUT.write(json.dumps(obj, default=str) + "\n")
    _OUT.flush()


def _read():
    line = _IN.readline()
    if not line:
        raise EOFError("host closed the connection")
    return json.loads(line)


class _Answered(Exception):
    def __init__(self, value):
        super().__init__(value)
        self.value = value


_next_id = 0


def _make_proxy(name):
    async def proxy(*args, **kwargs):
        global _next_id
        _next_id += 1
        _write(
            {
                "op": "bridge",
                "name": name,
                "args": list(args),
                "kwargs": kwargs,
                "_id": _next_id,
            }
        )
        reply = _read()
        while reply.get("op") != "bridge_result":
            reply = _read()
        if not reply.get("ok"):
            raise RuntimeError(reply.get("error", f"{name}() failed on the host"))
        return reply.get("value")

    return proxy


def _install_tools(specs, namespace):
    """Define each tool here, so calling one never leaves this process.

    The result is checked against what JSON can carry. Nothing forces that —
    the value goes straight to code in this same namespace — but a tool that
    hands back something only one runtime could produce would behave
    differently depending on which is underneath, and that is worse than a
    refusal the model can read.
    """
    for spec in specs:
        name = spec["name"]
        exec(spec["source"], namespace)
        rebuilt = namespace[name]
        # Only a function has a result to check; a data tool is just a value.
        if callable(rebuilt):
            namespace[name] = _checked(name, rebuilt)


def _checked(name, function):
    def tool(*args, **kwargs):
        value = function(*args, **kwargs)
        try:
            json.dumps(value)
        except (TypeError, ValueError):
            raise TypeError(
                f"tool {name!r} returned {type(value).__name__}, which cannot be "
                f"carried as JSON. Return plain data — a string, number, list, "
                f"dict, bool or None."
            ) from None
        return value

    tool.__name__ = getattr(function, "__name__", name)
    tool.__doc__ = function.__doc__
    tool.__wrapped__ = function
    return tool


def _exec_cell(code, namespace):
    buffer = io.StringIO()
    final, has_final, error = None, False, None
    try:
        compiled = compile(code, "<cell>", "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
        with redirect_stdout(buffer):
            pending = eval(compiled, namespace)
            if pending is not None:
                asyncio.run(pending)
    except _Answered as signal:
        has_final, final = True, signal.value
    except BaseException:
        error = traceback.format_exc()
    return {
        "op": "result",
        "stdout": buffer.getvalue(),
        "final": final,
        "has_final": has_final,
        "error": error,
    }


def main():
    init = _read()
    if init.get("op") != "init":
        raise RuntimeError(f"expected init, got {init!r}")

    namespace = {"__name__": "__rlm_cell__", "PROMPT": init.get("prompt", "")}
    for name in init.get("bridges", []):
        namespace[name] = _make_proxy(name)

    def FINAL(answer=None):
        raise _Answered(answer)

    namespace["FINAL"] = FINAL
    _install_tools(init.get("tools", []), namespace)

    # Kept out of the cell's namespace: the model must not see a name it did
    # not bind, and the summary is the host's question, not the model's tool.
    summariser = {}
    if init.get("summariser"):
        exec(init["summariser"], summariser)
    _write({"op": "ready"})

    while True:
        command = _read()
        operation = command.get("op")
        if operation == "shutdown":
            return
        if operation == "snapshot":
            describe = summariser.get("summarise")
            try:
                variables = describe(namespace) if describe else []
            except Exception:
                variables = []
            _write({"op": "namespace", "variables": variables})
            continue
        if operation != "exec":
            continue
        _write(_exec_cell(command.get("code", ""), namespace))


if __name__ == "__main__":
    main()
