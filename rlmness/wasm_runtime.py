"""CPython compiled to WebAssembly, several sandboxes to one Node process.

A sandbox costs about 122MB when it owns a whole Node process and about 45MB
when it is the second interpreter inside one — the runtime around it is the
expensive part, and it buys nothing per agent. A fan-out of eight is the
difference between roughly a gigabyte and roughly four hundred megabytes.

The isolation is unchanged: each sandbox is its own Pyodide interpreter with
its own globals, so two agents can no more see each other than if they lived
in separate processes. What they share is the host that carries their
messages, and the wire between them stays plain JSON.
"""

from __future__ import annotations

import atexit
import itertools
import json
import queue
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Callable, Mapping

from .runtime import ProtocolRuntime, RuntimeGone

_WORKER = str(Path(__file__).with_name("wasm_guest.mjs"))
_ROOT = str(Path(__file__).resolve().parents[1])


class _Host:
    """The one Node process, and the demultiplexer in front of it.

    Every sandbox gets a name and its own queue. A single reader thread sorts
    arriving messages by that name, so a reply meant for one agent can never
    surface in another's inbox — the property that makes sharing a process
    safe to do at all.
    """

    def __init__(self, node: str = "node"):
        self.process = subprocess.Popen(
            [node, _WORKER],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            bufsize=1,
            cwd=_ROOT,
        )
        self._inboxes: dict[str, queue.Queue] = {}
        self._lock = threading.Lock()
        # Several agents write down one pipe from several threads, so a line
        # has to be written whole or not at all.
        self._writing = threading.Lock()
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self):
        for line in self.process.stdout:
            try:
                message = json.loads(line)
            except ValueError:
                continue
            with self._lock:
                inbox = self._inboxes.get(message.get("sid"))
            if inbox is not None:
                inbox.put(message)
        # The host is gone; wake everyone still waiting rather than let them
        # sit until their timeout.
        with self._lock:
            for inbox in self._inboxes.values():
                inbox.put(None)

    def open(self, sid: str) -> queue.Queue:
        inbox: queue.Queue = queue.Queue()
        with self._lock:
            self._inboxes[sid] = inbox
        return inbox

    def release(self, sid: str) -> None:
        with self._lock:
            self._inboxes.pop(sid, None)

    def send(self, message: dict) -> None:
        with self._writing:
            try:
                self.process.stdin.write(json.dumps(message, default=str) + "\n")
                self.process.stdin.flush()
            except (BrokenPipeError, ValueError, OSError) as exc:
                raise RuntimeGone("runtime is not accepting input") from exc

    def stop(self) -> None:
        try:
            self.send({"op": "shutdown"})
            self.process.wait(timeout=5)
        except Exception:
            try:
                self.process.kill()
            except OSError:
                pass


_host: _Host | None = None
_host_lock = threading.Lock()
_names = itertools.count(1)


def _shared_host(node: str = "node") -> _Host:
    global _host
    with _host_lock:
        if _host is None or _host.process.poll() is not None:
            _host = _Host(node)
            atexit.register(_host.stop)
        return _host


def reset_host() -> None:
    """Drop the shared host, so the next sandbox starts a fresh one."""
    global _host
    with _host_lock:
        if _host is not None:
            _host.stop()
            _host = None


class _Session:
    """One sandbox's end of the shared host, shaped like a private channel."""

    def __init__(self, host: _Host, sid: str):
        self.host = host
        self.sid = sid
        self.inbox = host.open(sid)

    @property
    def process(self):
        return self.host.process

    def send(self, message):
        self.host.send({**message, "sid": self.sid})

    def kill(self):
        # One sandbox failing is not a reason to take down the others sharing
        # this process, so only its own registration goes.
        self.host.release(self.sid)

    def shutdown(self):
        try:
            self.send({"op": "close"})
        except RuntimeGone:
            pass
        self.host.release(self.sid)


class WasmRuntime(ProtocolRuntime):
    #: Whether the cell runs without syscalls.
    SEALED = True
    #: Its own interpreter, so a tool has to be rebuilt from text.
    NEEDS_SOURCE = True
    #: Sandboxes share one host process now, so an extra one costs about 45MB
    #: rather than 122MB. Still not free, so still bounded.
    MAX_LIVE = 16

    def __init__(
        self,
        prompt,
        bridges: Mapping[str, Callable] = (),
        timeout: float = 180.0,
        tools=(),
        node: str = "node",
    ):
        if shutil.which(node) is None:
            raise RuntimeError(f"{node!r} is not on PATH")
        host = _shared_host(node)
        session = _Session(host, f"s{next(_names)}")
        super().__init__(session, prompt, bridges, timeout, tools)


def wasm_available(node: str = "node") -> bool:
    if shutil.which(node) is None:
        return False
    try:
        probe = subprocess.run(
            [node, "-e", "require.resolve('pyodide')"],
            cwd=_ROOT,
            capture_output=True,
            timeout=30,
        )
        return probe.returncode == 0
    except Exception:
        return False
