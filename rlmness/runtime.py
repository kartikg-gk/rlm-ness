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


class ProtocolRuntime:
    def __init__(self, process, prompt, bridges, timeout, tools=()):
        self.process = process
        self.timeout = timeout
        self.bridges = dict(bridges)
        self._closed = False
        self._inbox: queue.Queue = queue.Queue()
        threading.Thread(target=self._pump, daemon=True).start()

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
            }
        )
        ready = self._receive()
        if ready.get("op") != "ready":
            raise RuntimeGone(f"runtime failed to start: {ready!r}")

    def _pump(self):
        for line in self.process.stdout:
            self._inbox.put(line)
        self._inbox.put(None)

    def _write(self, message):
        try:
            self.process.stdin.write(json.dumps(message, default=str) + "\n")
            self.process.stdin.flush()
        except (BrokenPipeError, ValueError, OSError) as exc:
            raise RuntimeGone("runtime is not accepting input") from exc

    def _receive(self):
        try:
            line = self._inbox.get(timeout=self.timeout)
        except queue.Empty:
            self._kill()
            raise CellTimeout(f"no response within {self.timeout}s")
        if line is None:
            raise RuntimeGone("runtime exited")
        return json.loads(line)

    def _kill(self):
        self._closed = True
        try:
            self.process.kill()
        except OSError:
            pass

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

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            self._write({"op": "shutdown"})
            self.process.wait(timeout=5)
        except (RuntimeGone, subprocess.TimeoutExpired):
            try:
                self.process.kill()
            except OSError:
                pass


class SubprocessRuntime(ProtocolRuntime):
    #: Its own process, so a tool has to be rebuilt from text.
    NEEDS_SOURCE = True

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
        super().__init__(process, prompt, bridges, timeout, tools)
