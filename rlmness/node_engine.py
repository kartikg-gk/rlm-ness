"""The launcher for the Node engine, which owns the loop for wasm runs.

Node owns the run. This side starts it, answers what only Python can answer
(frontends, sessions on disk, schema checks, a model route with no native client
over there) and raises whatever the run ended in. It keeps no budget and runs no
loop: the allowance a caller passes is copied in at the start and written back
from the outcome.
"""

from __future__ import annotations

import atexit
import dataclasses
import itertools
import json
import os
import queue
import shutil
import subprocess
import threading
import time
import uuid
import weakref
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

from .background import PRELUDE
from .config import Config
from .events import emit
from .home import node_root, pyodide_entry
from .limits import Abandoned, Allowance, AllowanceSpent, TimeUp
from .providers import NATIVE_CALL, AnthropicClient, ChatClient, Spend, native_call
from .runtime import SESSION, SUMMARISER, CellTimeout, RuntimeGone
from .schema import Shape
from .session import Session
from .tools import describe
from .wasm_runtime import WasmRuntime, wasm_available
from .environment import execution_environment

_HERE = Path(__file__).parent
_ENGINE = str(_HERE / "node_engine.mjs")
PROTOCOL = 1
#: Quiet this long past the cell timeout means the event loop is stuck.
WATCHDOG_GRACE = 15.0

_SOURCES = {
    "briefing": (_HERE / "briefing.py").read_text(encoding="utf-8"),
    "loop_text": (_HERE / "loop_text.py").read_text(encoding="utf-8"),
    "summariser": SUMMARISER,
    "session": SESSION,
    "prelude": PRELUDE,
}


class _Run:
    """What this side holds for one run while Node is doing the work."""

    def __init__(self, run_id, backend, trace, sessions, background, host_functions, timeout):
        self.id = run_id
        self.backend = backend
        self.trace = trace
        self.sessions = sessions
        self.background = background
        self.host_functions = dict(host_functions or {})
        self.timeout = timeout
        self.outcome: queue.Queue = queue.Queue()
        self.failures: dict[int, BaseException] = {}
        self.sink_failure: BaseException | None = None
        self.counts = {"events": 0, "callbacks": 0}


class _Host:
    """The one engine process, kept warm between runs."""

    def __init__(self, node: str = "node"):
        self.process = subprocess.Popen(
            [node, _ENGINE],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            bufsize=1,
            cwd=str(node_root()),
            env=execution_environment({"RLMNESS_PYODIDE": pyodide_entry().as_uri()}),
        )
        self.runs: dict[str, _Run] = {}
        self.allowances = weakref.WeakValueDictionary()
        self.workers = ThreadPoolExecutor(max_workers=64, thread_name_prefix="node-callback")
        self._writing = threading.Lock()
        self._lock = threading.Lock()
        self._last_heard = time.monotonic()
        self._stopped = False
        self._lost: Exception | None = None
        threading.Thread(target=self._pump, daemon=True).start()
        threading.Thread(target=self._watchdog, daemon=True).start()

    def send(self, **record) -> None:
        with self._writing:
            try:
                self.process.stdin.write(json.dumps({"v": PROTOCOL, **record}, default=str) + "\n")
                self.process.stdin.flush()
            except (BrokenPipeError, ValueError, OSError) as exc:
                raise RuntimeGone("the engine is not accepting input") from exc

    def _pump(self) -> None:
        for line in self.process.stdout:
            self._last_heard = time.monotonic()
            try:
                record = json.loads(line)
            except ValueError:
                continue
            kind = record.get("type")
            with self._lock:
                run = self.runs.get(record.get("runId"))
                allowance = self.allowances.get(record.get("allowanceId"))
            if kind == "budget":
                if allowance is not None:
                    with allowance._lock:
                        for name, delta in record["delta"].items():
                            setattr(allowance, name, getattr(allowance, name) + delta)
                continue
            if run is None:
                continue
            if kind == "event":
                run.counts["events"] += 1
                self._event(run, record)
            elif kind == "callback":
                run.counts["callbacks"] += 1
                self.workers.submit(self._serve, run, record)
            elif kind == "outcome":
                run.outcome.put(record)
            elif kind == "released":
                with self._lock:
                    self.runs.pop(run.id, None)
        self.process.wait()
        self._host_lost(self._lost or RuntimeGone("the engine process exited"))

    def _watchdog(self) -> None:
        # Worker deadlines are local. This watchdog covers coordinator failure.
        while not self._stopped and self.process.poll() is None:
            time.sleep(0.5)
            with self._lock:
                limit = max((run.timeout for run in self.runs.values()), default=None)
            if limit is not None and time.monotonic() - self._last_heard > limit + WATCHDOG_GRACE:
                # Named before the kill, because the reader sees the pipe close at
                # once and tells every run what ended it.
                self._lost = CellTimeout(
                    f"the engine went quiet for over {limit}s, so its process was ended; "
                    "every agent in it stopped and only saved checkpoints remain"
                )
                self.process.kill()
                return

    def _host_lost(self, failure: Exception) -> None:
        with self._lock:
            active = list(self.runs.values())
            self.runs.clear()
        for run in active:
            run.outcome.put(failure)

    def _event(self, run: _Run, record: dict) -> None:
        fields = dict(record.get("fields") or {})
        if isinstance(fields.get("usage"), dict):
            fields["usage"] = Spend(**fields["usage"])
        try:
            emit(run.trace, record["name"], **fields)
        except BaseException as failure:  # noqa: BLE001 - raised from the run that owns the sink
            run.sink_failure = run.sink_failure or failure
            self.cancel(run.id, "a trace sink failed")

    def _serve(self, run: _Run, record: dict) -> None:
        try:
            context = getattr(run.backend, "context", None)
            if context is not None:
                context.agent = record["agentId"]
            reply = {"ok": True, "value": _answer(run, record["kind"],
                record.get("args") or {}, record["agentId"])}
        except BaseException as failure:  # noqa: BLE001 - every failure belongs to the run
            ref = record["requestId"]
            run.failures[ref] = failure
            reply = {"ok": False, "error": {
                "name": type(failure).__name__, "message": str(failure), "ref": ref,
            }}
        try:
            self.send(type="callback_result", runId=run.id, requestId=record["requestId"], **reply)
        except RuntimeGone:
            pass

    def cancel(self, run_id: str, message: str, reason: str = "token") -> None:
        try:
            self.send(type="cancel", runId=run_id, message=message, reason=reason)
        except RuntimeGone:
            pass

    def stop(self) -> None:
        self._stopped = True
        try:
            self.send(type="shutdown")
            self.process.wait(timeout=5)
        except Exception:
            try:
                self.process.kill()
            except OSError:
                pass
        self.workers.shutdown(wait=False)


_host: _Host | None = None
_host_lock = threading.Lock()
_session_ids = itertools.count(1)


class _NodeRuntime:
    """Ownership of a resident interpreter, without executing cells in Python."""

    def __init__(self, host, session_id):
        self.host, self.id = host, session_id
        self._closed = False
        self.sessions = {}

    def close(self):
        if self._closed:
            return
        self._closed = True
        for state in list(self.sessions.values()):
            other = state._runtime
            if isinstance(other, _NodeRuntime) and other is not self:
                other.close()
        self.sessions.clear()
        try:
            self.host.send(type="session_closed", sessionId=self.id)
        except RuntimeGone:
            pass


def _shared_host() -> _Host:
    global _host
    with _host_lock:
        if _host is None or _host.process.poll() is not None:
            _host = _Host()
            atexit.register(_host.stop)
        return _host


def reset_engine() -> None:
    """Drop the engine process, so the next run starts a fresh one."""
    global _host
    with _host_lock:
        if _host is not None:
            _host.stop()
            _host = None


def _answer(run: _Run, kind: str, args: dict, agent_id=None):
    if kind == "runtime_prepare":
        if args.get("owner") is not None and run.background is not None:
            run.background.owners[agent_id] = args["owner"]
        prepared = run.runtime_factory.NODE_PREPARE(
            args["prompt"], args["bridges"], run.timeout,
            session=run.sessions.get(args.get("sessionId")), agent=agent_id)
        functions = prepared.get("host_functions", {})
        run.agent_functions[agent_id] = functions
        return {"hostFunctions": sorted(functions), "startup": prepared.get("startup", ""),
                "cellPolicy": prepared.get("cell_policy")}
    if kind == "model":
        answer = run.backend.complete(args["messages"], model=args["model"])
        text, usage, reasoning = answer if len(answer) == 3 else (*answer, None)
        return {"text": text, "usage": dataclasses.asdict(usage), "reasoning": reasoning}
    if kind == "schema_text":
        return Shape(args["schema"]).text
    if kind == "schema_problems":
        return Shape(args["schema"]).problems(args["value"])
    if kind == "native_calls":
        return "".join(native_call(name, arguments) for name, arguments in args["calls"])
    if kind == "role":
        if run.background is None or run.background.resolve_role is None:
            raise ValueError("no role resolver configured")
        return run.background.resolve_role(args["role"])
    if kind == "host":
        functions = {**run.host_functions, **run.agent_functions.get(agent_id, {})}
        return functions[args["name"]](*args.get("args", []), **args.get("kwargs", {}))
    if kind == "background_record":
        save = getattr(run.background, "node_record", None)
        if save is not None:
            save(args["key"], args["record"])
        return None
    if kind == "session_new":
        loader = getattr(run.background, "node_state", None)
        if loader is not None:
            state = loader(args["key"])
        else:
            state = run.background.node_states.setdefault(args["key"], Session())
        runtime = state._runtime
        if not isinstance(runtime, _NodeRuntime) or runtime._closed or runtime.host is not run.host:
            state.close_runtime()
            runtime = state._runtime = _NodeRuntime(run.host, next(_session_ids))
        run.sessions[runtime.id] = state
        return runtime.id
    if kind == "session":
        return _session_call(run, args)
    raise ValueError(f"unknown callback {kind!r}")


def _session_call(run: _Run, args: dict):
    """Everything saved goes through the Session the other engine uses."""
    session = run.sessions[args["id"]]
    op = args["op"]
    if op == "state_for_guest":
        return session.state_for_guest()
    if op == "preamble":
        return session.preamble()
    if op == "ask":
        session.asking = args["question"] if isinstance(args["question"], str) else str(args["question"])
        session.save()
        return None
    if op == "probe":
        return session.probe(args["restore_failed"], args["taken"])
    if op == "absorb":
        cell = {"question": len(session.answered), **args["cell"]}
        session.absorb(args["swept"], cell, set(args["kept"]))
        session.save()
        return None
    if op == "settled":
        question = args["question"] if isinstance(args["question"], str) else str(args["question"])
        session.settled(
            question, args["answer"], trace=getattr(run.trace, "path", None), run_id=args["run_id"],
        )
        session.save()
        return None
    raise ValueError(f"unknown session call {op!r}")


def _native_route(backend, config: Config) -> dict | None:
    """How Node can reach this model itself, when it is an ordinary HTTP client.

    Anything else, an external agent or a client built around a test transport,
    stays on the model callback. The key crosses the engine's private pipe only.
    """
    if type(backend).complete not in (ChatClient.complete, AnthropicClient.complete):
        return None
    client = getattr(backend, "client", None)
    if type(client) is not httpx.Client or type(client._transport) is not httpx.HTTPTransport:
        return None
    timeout = client.timeout.read
    return {
        "name": backend.provider.name,
        "shape": "anthropic" if isinstance(backend, AnthropicClient) else "chat",
        "endpoint": backend.provider.endpoint,
        "headers": backend._headers(),
        "timeout": timeout if timeout is not None else config.api_timeout,
        "deadline": backend.deadline,
        "max_retries": backend.max_retries,
        "backoff": backend.backoff,
        "retry_after_max": backend.retry_after_max,
        "temperature": backend.temperature,
        "reasoning_effort": backend.reasoning_effort,
        "max_tokens": backend.max_tokens,
        "default_max_tokens": getattr(backend, "DEFAULT_MAX_TOKENS", None),
    }


_FAILURES = {
    "AllowanceSpent": AllowanceSpent,
    "TimeUp": TimeUp,
    "Abandoned": Abandoned,
    "CellTimeout": CellTimeout,
    "RuntimeGone": RuntimeGone,
    "RuntimeError": RuntimeError,
    "TypeError": TypeError,
    "ValueError": ValueError,
    "TimeoutError": TimeoutError,
}


def _raise_failure(run: _Run, error: dict) -> None:
    original = run.failures.get(error.get("ref"))
    if original is not None:
        raise original
    name = error.get("name")
    if name == "StepsUsedUp":
        from .engine import StepsUsedUp

        raise StepsUsedUp(error.get("message"))
    if name in _FAILURES:
        raise _FAILURES[name](error.get("message"))
    raise RuntimeError(f"{name}: {error.get('message')}")


def solve_on_node(
    prompt, backend, *, instruction=None, config: Config, runtime_factory=None, trace=None,
    allowance: Allowance | None = None, cancel: threading.Event | None = None, tools=None,
    run_id=None, session=None, output_schema=None, background=None, host_functions=None,
):
    """`solve`, with Node owning the loop."""
    from .engine import Answer

    if (runtime_factory is not None and runtime_factory is not WasmRuntime
            and not callable(getattr(runtime_factory, "NODE_PREPARE", None))):
        raise ValueError("the node engine runs its own interpreters; a runtime_factory cannot be used")
    if config.runtime != "wasm":
        raise ValueError("the node engine needs runtime='wasm'")
    if not wasm_available() or shutil.which("node") is None:
        raise RuntimeError("the node engine needs Node and pyodide: run `rlmness --setup`")

    shape = (
        Shape(output_schema)
        if output_schema is not None and config.enable_structured_output else None
    )
    prepared = describe(tools, need_source=True)
    if allowance is None:
        allowance = Allowance.from_config(
            dataclasses.replace(config, max_live=min(config.max_live, WasmRuntime.MAX_LIVE))
        )
    agent_id = run_id or uuid.uuid4().hex
    wire_id = uuid.uuid4().hex

    host = _shared_host()
    sessions: dict[int, Session] = {}
    session_id = None
    if session is not None:
        previous = session._runtime
        if (not isinstance(previous, _NodeRuntime) or previous._closed
                or previous.host is not host):
            session.close_runtime()
            previous = _NodeRuntime(host, next(_session_ids))
            session._runtime = previous
        session_id = previous.id
        sessions = previous.sessions
        sessions[session_id] = session

    run = _Run(wire_id, backend, trace, sessions, background, host_functions, config.timeout)
    run.runtime_factory = runtime_factory
    run.agent_functions = {}
    run.host = host
    with host._lock:
        host.runs[wire_id] = run
        host.allowances[str(id(allowance))] = allowance

    background_id = None
    if background is not None:
        background_id = str(id(background))
        if getattr(background, "_node_host", None) is not host:
            background._node_host = host

            def closed(key=background_id, engine=host):
                try:
                    engine.send(type="background_closed", backgroundId=key)
                except RuntimeGone:
                    pass

            background.on_close.append(closed)

    host.send(
        type="start", runId=wire_id, agentId=agent_id,
        prompt=prompt, instruction=instruction,
        config=dataclasses.asdict(config),
        allowanceId=str(id(allowance)),
        budget={
            "max_depth": allowance.max_depth, "max_calls": allowance.max_calls,
            "max_cost": allowance.max_cost, "max_live": allowance.max_live,
            "max_seconds": allowance.max_seconds,
            "max_completion_tokens": allowance.max_completion_tokens,
            "max_prompt_tokens": allowance.max_prompt_tokens,
            "calls": allowance.calls, "cost": allowance.cost,
            "completion_tokens": allowance.completion_tokens,
            "prompt_tokens": allowance.prompt_tokens,
            "unpriced_calls": allowance.unpriced_calls,
            "elapsed_seconds": max(0, time.monotonic() - allowance._started),
        },
        route=_native_route(backend, config),
        sources=_SOURCES,
        tools=[
            {"name": tool.name, "description": tool.description, "signature": tool.signature,
             "callable": tool.callable, "source": tool.source}
            for tool in prepared
        ],
        traced=trace is not None,
        watchNamespace=hasattr(trace, "namespace_changed"),
        nativeCallMark=NATIVE_CALL,
        hostFunctions=sorted(host_functions or {}),
        runtimePrepare=callable(getattr(runtime_factory, "NODE_PREPARE", None)),
        backgroundId=background_id,
        backgroundRecords=getattr(background, "node_records", dict)(),
        maxAgents=getattr(background, "max_agents", 50),
        schema=shape.schema if shape is not None else None,
        schemaText=shape.text if shape is not None else None,
        sessionId=session_id,
    )

    told = False
    try:
        while True:
            try:
                outcome = run.outcome.get(timeout=0.05)
                break
            except queue.Empty:
                if told:
                    continue
                if cancel is not None and cancel.is_set():
                    told = True
                    host.cancel(wire_id, "a sibling failed; this branch was abandoned")
                elif allowance.cancelled:
                    told = True
                    host.cancel(wire_id, "work was abandoned", reason="allowance")
    finally:
        if background is None:
            with host._lock:
                host.runs.pop(wire_id, None)

    if isinstance(outcome, Exception):
        raise outcome
    spent = outcome.get("budget") or {}
    run.counts.update(outcome.get("counts") or {})
    last_run_facts.clear()
    last_run_facts.update({**spent, **run.counts})
    if run.sink_failure is not None:
        raise run.sink_failure
    if outcome["status"] != "completed":
        _raise_failure(run, outcome["error"])
    return Answer(output=outcome["output"], steps=outcome["steps"], usage=Spend(**outcome["usage"]))


#: What the last finished run reported: its bridge traffic and sandboxes. For measuring.
last_run_facts: dict = {}
