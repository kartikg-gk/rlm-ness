"""The runtime interface, and the subprocess that satisfies it."""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

_RUNNER = str(Path(__file__).with_name("cell_runner.py"))

# Read once. The summary of a namespace is built where the namespace is, so
# the code that builds it has to travel there — the same one-way trip a tool
# makes, and for the same reason: nothing comes back but plain data.
SUMMARISER = Path(__file__).with_name("namespace.py").read_text(encoding="utf-8")

# Ships the same way and for the same reason: the values are over there, so
# the packing happens over there and only plain data comes back.
SESSION = Path(__file__).with_name("session_guest.py").read_text(encoding="utf-8")


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
        # A cell runs on one thread while close() may arrive on another,
        # so a line has to go out whole rather than interleaved with a
        # shutdown.
        self._writing = threading.Lock()
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self):
        for line in self.process.stdout:
            self.inbox.put(line)
        self.inbox.put(None)

    def send(self, message):
        with self._writing:
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

    def __init__(self, channel, prompt, bridges, timeout, tools=(), session=None):
        self.channel = channel
        self.timeout = timeout
        self.bridges = dict(bridges)
        self._closed = False
        # Bridges run off the receive loop so several can be outstanding at
        # once. Serving them inline meant the loop was busy inside the first
        # call when the second arrived, which made a concurrent fan-out
        # impossible however the cell was written. One pool per runtime, so a
        # child's bridges never queue behind its parent's.
        self._bridge_workers = ThreadPoolExecutor(
            max_workers=32, thread_name_prefix="bridge"
        )
        self._writing = threading.Lock()
        # How many sub-agent calls are running inside this runtime's current
        # cell. While any is, the cell is waiting rather than hung.
        self._bridges_in_flight = 0
        self._bridge_count_lock = threading.Lock()

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
                # Only when the caller is running a session. A run that is not
                # pays nothing: no source shipped, no `commit` bound, no sweep.
                "session": SESSION if session is not None else None,
                "restore": session,
            }
        )
        ready = self._receive()
        if ready.get("op") != "ready":
            raise RuntimeGone(f"runtime failed to start: {ready!r}")
        #: Names the saved state held that would not come back. The caller
        #: keeps them rather than letting a sweep that cannot see them report
        #: them as deleted.
        self.restore_failed = set(ready.get("restore_failed") or ())

    @property
    def process(self):
        """The process behind the channel, when there is one to itself."""
        return getattr(self.channel, "process", None)

    def _write(self, message):
        # Bridge replies come from pool threads, so two can be ready at once.
        with self._writing:
            self.channel.send(message)

    def _receive(self):
        while True:
            try:
                line = self.channel.inbox.get(timeout=self.timeout)
            except queue.Empty:
                # A cell that has handed work to a sub-agent is not hung, it is
                # waiting — and a sub-agent's whole run happens inside its
                # parent's cell, which takes as long as a run takes. Timing the
                # parent out here killed the tree the moment delegation started
                # working. The run as a whole is still bounded, by the
                # allowance's wall clock and by each child's own limits.
                if self._outstanding_bridges():
                    continue
                self._kill()
                raise CellTimeout(f"no response within {self.timeout}s")
            if line is None:
                raise RuntimeGone("runtime exited")
            return line if isinstance(line, dict) else json.loads(line)

    def _outstanding_bridges(self) -> int:
        with self._bridge_count_lock:
            return self._bridges_in_flight

    def _kill(self):
        self._closed = True
        # Not waited on: a bridge in flight is usually blocked on a model call,
        # and the caller killing this runtime is not obliged to sit through it.
        self._bridge_workers.shutdown(wait=False)
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
        finally:
            with self._bridge_count_lock:
                self._bridges_in_flight -= 1
        self._reply(call_id, True, value=value)

    def _serve_bridge_async(self, message) -> None:
        """Start a bridge without waiting for it.

        A runtime that has been closed under us would raise from the pool with
        nobody to catch it, so the shut case is answered on the spot.
        """
        if self._closed:
            self._reply(message.get("_id"), False, error="the runtime is closed")
            return
        # Counted before the work is queued, so the receive loop never sees a
        # gap where the call is neither in flight nor finished.
        with self._bridge_count_lock:
            self._bridges_in_flight += 1
        try:
            self._bridge_workers.submit(self._serve_bridge, message)
        except BaseException:
            with self._bridge_count_lock:
                self._bridges_in_flight -= 1
            raise

    def _reply(self, call_id, ok: bool, *, value=None, error=None) -> None:
        message = {"op": "bridge_result", "ok": ok, "_id": call_id}
        if ok:
            message["value"] = value
        else:
            message["error"] = error
        try:
            self._write(message)
        except Exception:
            # The runtime went away while this call was being served. There is
            # nobody left to tell, and raising here would only strand the
            # exception in a pool thread.
            pass

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
                # Handed to the pool rather than run here: the guest can have
                # several calls outstanding, and the receive loop has to stay
                # free to collect the rest of them. Replies carry the call id,
                # so they may return in any order.
                self._serve_bridge_async(message)

    def sweep(self, code: str | None = None) -> dict:
        """Everything in the namespace worth carrying into the next run."""
        if self._closed:
            return {}
        try:
            self._write({"op": "sweep", "code": code})
            while True:
                message = self._receive()
                if message.get("op") == "swept":
                    return {
                        "variables": message.get("variables", {}),
                        "functions": message.get("functions", {}),
                        "modules": message.get("modules", {}),
                        "dropped": message.get("dropped", {}),
                    }
                if message.get("op") == "bridge":
                    self._serve_bridge_async(message)
        except (RuntimeGone, CellTimeout):
            return {}

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
                    self._serve_bridge_async(message)
        except (RuntimeGone, CellTimeout):
            # A snapshot is for looking at a run, never part of running one.
            return []

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._bridge_workers.shutdown(wait=False)
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
        session=None,
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
        super().__init__(ProcessChannel(process), prompt, bridges, timeout, tools, session)
