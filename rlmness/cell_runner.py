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
    _write({"op": "ready"})

    while True:
        command = _read()
        operation = command.get("op")
        if operation == "shutdown":
            return
        if operation != "exec":
            continue
        _write(_exec_cell(command.get("code", ""), namespace))


if __name__ == "__main__":
    main()
