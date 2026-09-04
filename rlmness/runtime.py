"""The runtime interface, and the subprocess that satisfies it."""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

_RUNNER = str(Path(__file__).with_name("cell_runner.py"))

# Read once. The summary of a namespace is built where the namespace is, so
# the code that builds it has to travel there — the same one-way trip a tool
# makes, and for the same reason: nothing comes back but plain data.
SUMMARISER = Path(__file__).with_name("namespace.py").read_text(encoding="utf-8")


class CellTimeout(Exception):
    pass


class RuntimeGone(Exception):
    pass


@dataclass
class CellOutcome:
    stdout: str = ""
    final: Any = None
    has_final: bool = False
    error: str | None = None


class ProcessChannel:
    """One process, one sandbox, talking over its own pipes."""

    def __init__(self, process):
        self.process = process
        self.inbox: queue.Queue = queue.Queue()
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self):
        for line in self.process.stdout:
            self.inbox.put(line)
        self.inbox.put(None)

    def send(self, message):
        try:
            self.process.stdin.write(json.dumps(message, default=str) + "\n")
            self.process.stdin.flush()
        except (BrokenPipeError, ValueError, OSError) as exc:
            raise RuntimeGone("runtime is not accepting input") from exc

    def kill(self):
        try:
            self.process.kill()
        except OSError:
            pass

    def shutdown(self):
        try:
            self.send({"op": "shutdown"})
            self.process.wait(timeout=5)
        except (RuntimeGone, subprocess.TimeoutExpired):
            self.kill()


class ProtocolRuntime:
    """The wire protocol, over whatever channel carries it.

    The channel is a seam rather than a process, because a sandbox does not
    have to own one: several can share a single interpreter host and still be
    as separate from each other as they would be in separate processes.
    """

    def __init__(self, channel, prompt, bridges, timeout, tools=()):
        self.channel = channel
        self.timeout = timeout
        self.bridges = dict(bridges)
        self._closed = False

        # Tools travel as source and are defined inside the namespace, so a
        # call to one never reaches back across this boundary.
        self._write(
            {
                "op": "init",
                "prompt": prompt,
                "bridges": list(self.bridges),
                "tools": [
                    {"name": tool.name, "source": tool.source} for tool in tools
                ],
                "summariser": SUMMARISER,
            }
        )
        ready = self._receive()
        if ready.get("op") != "ready":
            raise RuntimeGone(f"runtime failed to start: {ready!r}")

    @property
    def process(self):
        """The process behind the channel, when there is one to itself."""
        return getattr(self.channel, "process", None)

    def _write(self, message):
        self.channel.send(message)

    def _receive(self):
        try:
            line = self.channel.inbox.get(timeout=self.timeout)
        except queue.Empty:
            self._kill()
            raise CellTimeout(f"no response within {self.timeout}s")
        if line is None:
            raise RuntimeGone("runtime exited")
        return line if isinstance(line, dict) else json.loads(line)

    def _kill(self):
        self._closed = True
        self.channel.kill()

    def _serve_bridge(self, message) -> None:
        name = message.get("name")
        call_id = message.get("_id")
        bridge = self.bridges.get(name)
        if bridge is None:
            self._reply(call_id, False, error=f"no bridge named {name!r}")
            return
        try:
            value = bridge(*message.get("args", []), **message.get("kwargs", {}))
        except Exception as error:
            self._reply(call_id, False, error=f"{type(error).__name__}: {error}")
            return
        self._reply(call_id, True, value=value)

    def _reply(self, call_id, ok: bool, *, value=None, error=None) -> None:
        message = {"op": "bridge_result", "ok": ok, "_id": call_id}
        if ok:
            message["value"] = value
        else:
            message["error"] = error
        self._write(message)

    def execute(self, code: str) -> CellOutcome:
        self._write({"op": "exec", "code": code})
        while True:
            message = self._receive()
            operation = message.get("op")
            if operation == "result":
                return CellOutcome(
                    stdout=message.get("stdout", ""),
                    final=message.get("final"),
                    has_final=bool(message.get("has_final")),
                    error=message.get("error"),
                )
            if operation == "bridge":
                self._serve_bridge(message)

    def snapshot(self) -> list[dict]:
        """What is bound in the cell's namespace, as plain data.

        A read-only question asked of the runtime, answered on the same wire
        the protocol already uses. It carries no objects, so a runtime stays
        exactly as reachable as it was — this is not a bridge.
        """
        if self._closed:
            return []
        try:
            self._write({"op": "snapshot"})
            while True:
                message = self._receive()
                if message.get("op") == "namespace":
                    return message.get("variables", [])
                if message.get("op") == "bridge":
                    self._serve_bridge(message)
        except (RuntimeGone, CellTimeout):
            # A snapshot is for looking at a run, never part of running one.
            return []

    def close(self):
        if self._closed:
            return
        self._closed = True
        self.channel.shutdown()


class SubprocessRuntime(ProtocolRuntime):
    #: Whether the cell runs without syscalls.
    SEALED = False
    #: Its own process, so a tool has to be rebuilt from text.
    NEEDS_SOURCE = True
    #: A bare interpreter, measured at about 4MB resident, so a wide tree
    #: costs little.
    MAX_LIVE = 32

    def __init__(
        self,
        prompt,
        bridges: Mapping[str, Callable] | Sequence[str] = (),
        timeout: float = 120.0,
        tools=(),
    ):
        environment = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
        process = subprocess.Popen(
            [sys.executable, "-I", _RUNNER],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            bufsize=1,
            env=environment,
        )
        super().__init__(ProcessChannel(process), prompt, bridges, timeout, tools)
