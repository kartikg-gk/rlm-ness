"""The subprocess that executes cells."""

import ast
import asyncio
import io
import json
import queue
import sys
import threading
import traceback
from contextlib import redirect_stdout

_OUT = sys.__stdout__
_IN = sys.__stdin__


_WRITING = threading.Lock()


def _write(obj):
    # Several helper calls can be in flight at once, each writing its own
    # request. The lock keeps two of them from interleaving into one line
    # that neither side can parse.
    with _WRITING:
        _OUT.write(json.dumps(obj, default=str) + '\n')
        _OUT.flush()


def _read():
    line = _IN.readline()
    if not line:
        raise EOFError("host closed the connection")
    return json.loads(line)


# Replies are sorted by a reader thread rather than by whoever happens to be
# waiting. A call that read the wire itself would consume the reply meant for
# another call still awaiting, which is what made a second concurrent call
# impossible.
_COMMANDS = queue.Queue()
_PENDING = {}
_PENDING_LOCK = threading.Lock()
_LOOP = None


def _resolve(future, ok, payload):
    if future.done():
        return
    if ok:
        future.set_result(payload)
    else:
        future.set_exception(RuntimeError(payload))


def _settle(future, ok, payload):
    """Finish a call from the reader thread, on the loop that is awaiting it."""
    loop = _LOOP
    if loop is None:
        return
    loop.call_soon_threadsafe(_resolve, future, ok, payload)


def _pump():
    """Read the wire forever, routing each line to whoever wants it."""
    while True:
        try:
            message = _read()
        except (EOFError, ValueError):
            break
        if message.get("op") == "bridge_result":
            with _PENDING_LOCK:
                future = _PENDING.pop(message.get("_id"), None)
            if future is not None:
                _settle(
                    future,
                    bool(message.get("ok")),
                    message.get("value") if message.get("ok")
                    else message.get("error", "the call failed on the host"),
                )
        else:
            _COMMANDS.put(message)
    # The host is gone. Nothing more will arrive, so anything still waiting
    # has to be told rather than left hanging until the cell times out.
    _COMMANDS.put(None)
    with _PENDING_LOCK:
        stranded = list(_PENDING.values())
        _PENDING.clear()
    for future in stranded:
        _settle(future, False, "host closed the connection")


# The cell's own record of whether FINAL was called this turn. FINAL used to
# raise and abort the rest of the block; now it just marks the outcome, so
# code the model writes after FINAL runs exactly as it would in any other
# cell -- a self-correction (a later FINAL call) overwrites this the same way
# reassigning any other variable would, and a mistake after FINAL is reported
# as an error without losing the answer that was already given.
_outcome = {"has_final": False, "final": None}


_next_id = 0


def _make_proxy(name):
    """A helper that really yields while the host works on it.

    The body used to write its request and then spin on the wire until its own
    reply came back. That made the `async def` a promise the call could not
    keep: awaiting it never returned to the event loop, so a second call could
    not start until the first had finished, and gathering several of them ran
    them one after another. Registering a future and awaiting it lets the loop
    get on with the rest.
    """
    async def proxy(*args, **kwargs):
        global _next_id
        _next_id += 1
        call = _next_id
        future = asyncio.get_running_loop().create_future()
        with _PENDING_LOCK:
            _PENDING[call] = future
        try:
            _write(
                {
                    "op": "bridge",
                    "name": name,
                    "args": list(args),
                    "kwargs": kwargs,
                    "_id": call,
                }
            )
        except BaseException:
            with _PENDING_LOCK:
                _PENDING.pop(call, None)
            raise
        return await future

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


async def _drive(pending):
    """Run the cell's coroutine, telling the reader thread where to deliver.

    A fresh event loop is made for every cell, so the thread that settles
    replies has to be told which one is current rather than capturing one at
    import time.
    """
    global _LOOP
    _LOOP = asyncio.get_running_loop()
    try:
        return await pending
    finally:
        _LOOP = None


def _exec_cell(code, namespace):
    buffer = io.StringIO()
    _outcome["has_final"] = False
    _outcome["final"] = None
    error = None
    try:
        compiled = compile(code, "<cell>", "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
        with redirect_stdout(buffer):
            pending = eval(compiled, namespace)
            if pending is not None:
                asyncio.run(_drive(pending))
    except BaseException:
        error = traceback.format_exc()
    # Read after the try, not out of it: a mistake in code written after FINAL
    # still lands here as an error, but the answer already given is not lost
    # to it.
    return {
        "op": "result",
        "stdout": buffer.getvalue(),
        "final": _outcome["final"],
        "has_final": _outcome["has_final"],
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
        _outcome["has_final"] = True
        _outcome["final"] = answer

    namespace["FINAL"] = FINAL
    _install_tools(init.get("tools", []), namespace)

    # Kept out of the cell's namespace: the model must not see a name it did
    # not bind, and the summary is the host's question, not the model's tool.
    summariser = {}
    if init.get("summariser"):
        exec(init["summariser"], summariser)

    # Same treatment, and one exception: `commit` is bound across because the
    # model is meant to call it. The rest stays out here, so a sweep can never
    # pick up its own machinery and a restored name can never collide with it.
    session = {}
    restored = []
    if init.get("session"):
        exec(init["session"], session)
        namespace["commit"] = session["commit"]
        saved = init.get("restore")
        if saved:
            restored = session["restore"](namespace, saved)
    threading.Thread(target=_pump, daemon=True).start()
    _write({"op": "ready", "restore_failed": restored})

    while True:
        command = _COMMANDS.get()
        if command is None:
            return
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
        if operation == "sweep":
            gather = session.get("sweep")
            try:
                swept = gather(namespace, command.get("code")) if gather else {}
            except Exception as failure:
                swept = {"variables": {}, "functions": {},
                         "dropped": {"*": f"the sweep failed: {failure}"}}
            _write({"op": "swept", **swept})
            continue
        if operation != "exec":
            continue
        _write(_exec_cell(command.get("code", ""), namespace))


if __name__ == "__main__":
    main()
