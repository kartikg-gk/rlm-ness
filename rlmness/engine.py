"""The execution loop."""

from __future__ import annotations

import ast
import dataclasses
import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Mapping

from .providers import ModelClient, Spend, combine
from .limits import Allowance, Abandoned
from .config import Config, load_config
from .briefing import (
    PREVIEW,
    opening_code,
    opening_message,
    shows_everything,
    system_prompt,
)
from .wasm_runtime import WasmRuntime
from .tools import describe
from .in_process import InProcessRuntime
from .runtime import SubprocessRuntime
from .events import emit

RUNTIMES = {
    "subprocess": SubprocessRuntime,
    "wasm": WasmRuntime,
    "in-process": InProcessRuntime,
}

_FENCE = re.compile(r"```([A-Za-z0-9_.+-]*)[ \t]*\r?\n(.*?)```", re.S)
PYTHON_TAGS = {"", "python", "py", "python3"}

NO_CODE = (
    "No fenced Python block was found in your reply. Nothing ran. "
    "Reply with exactly one ```python block."
)

TOO_DEEP = (
    "maximum recursion depth reached: rlm() is not available here. "
    "Solve this task yourself, slicing PROMPT in the namespace."
)

UNSEEN = (
    "Nothing ran. That block answers without reading PROMPT, and you have only "
    "been shown its first and last {preview} characters — the answer would be "
    "based on a sample rather than on the data.\n"
    "Look at PROMPT in the namespace first. If the answer genuinely does not "
    "depend on what is in it, print your reasoning this turn and call FINAL on "
    "the next one."
)


def answers_without_looking(code: str) -> bool:
    """Whether a cell ends the run without ever reading PROMPT.

    A model handed a self-contained-sounding question will sometimes answer it
    from the opening message and use FINAL to deliver the text, never touching
    the data it was given. That is the one mistake this loop cannot recover
    from: every other wrong turn leaves another step to correct it, and this
    one ends the run.

    Reading the name at all counts as looking. `FINAL(PROMPT.count("r"))` is a
    complete and correct answer in one cell, and refusing it would cost a turn
    to learn nothing.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        # Unparseable code is the extractor's problem; the model needs to see
        # the SyntaxError rather than a lecture about PROMPT.
        return False
    finals = any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "FINAL"
        for node in ast.walk(tree)
    )
    if not finals:
        return False
    return not any(
        isinstance(node, ast.Name) and node.id == "PROMPT" for node in ast.walk(tree)
    )


class StepsUsedUp(Exception):
    pass


@dataclass
class Answer:
    output: Any
    steps: int
    usage: Spend


def runnable_code(text: str) -> str | None:
    """Everything the reply meant to run, as one cell.

    A model that splits a plan across two blocks — set the chunks up here,
    hand them out there — meant both to run. Taking only the first drops the
    second silently, and the half that gets dropped is the half that does
    something, because setup comes first.

    Blocks that do not parse are left out rather than joined in. That is what
    separates a sketch from an instruction: models illustrate before they
    commit, and an illustration is rarely valid Python. Joining the survivors
    keeps both behaviours — the sketch is still discarded, the real work is
    still whole.
    """
    candidates = [
        body
        for tag, body in _FENCE.findall(text or "")
        if tag.lower() in PYTHON_TAGS
    ]
    runnable = []
    for body in candidates:
        try:
            ast.parse(body)
        except SyntaxError:
            continue
        runnable.append(body)

    if not runnable:
        # Hand back the first anyway: the resulting SyntaxError is the
        # feedback that lets the model fix itself.
        return candidates[0] if candidates else None
    if len(runnable) == 1:
        return runnable[0]

    joined = "\n".join(block.strip("\n") for block in runnable)
    try:
        ast.parse(joined)
    except SyntaxError:
        # Two halves that each parse but do not compose. The first is the one
        # the model wrote first, so it is the one that was meant to run.
        return runnable[0]
    return joined


#: The old name, from when only one block was ever run.
first_runnable_block = runnable_code


def budget_banner(used: int, max_steps: int) -> str:
    """Tell the model how many turns are left, once it is over halfway.

    A model cannot see the step allowance and will happily spend its last turns
    re-checking something it already printed. Saying nothing early avoids
    spending tokens reminding a fresh agent it has plenty of room.
    """
    if used * 2 < max_steps:
        return ""
    remaining = max_steps - used
    return (
        f"[Steps remaining after this one: {remaining} / {max_steps}]\n"
        "[If you are not close, hand the remaining pieces to sub-agents rather "
        "than looking again yourself.]\n"
    )


def label_output(text: str, limit: int) -> str:
    if not text:
        return "[EMPTY OUTPUT]"
    if len(text) > limit:
        return f"[TRUNCATED: Last {limit} chars shown].. {text[-limit:]}"
    return f"[FULL OUTPUT SHOWN]... {text}"


class _Clock:
    """Wall-clock readings that can still tell short things apart.

    The system clock ticks about once every sixteen milliseconds on some
    platforms, so two readings taken either side of a fast cell come back
    identical and its duration reads as exactly zero — which is the one thing
    these timestamps exist to measure. The performance counter resolves
    fractions of a microsecond but says nothing about the date.

    So the date is read once and the counter supplies every offset from it.
    The result is a real timestamp that is also honest about small gaps, and
    it cannot be dragged backwards by a clock correction mid-run.
    """

    def __init__(self):
        self._wall = datetime.now(timezone.utc)
        self._mark = time.perf_counter()

    def now(self) -> str:
        elapsed = time.perf_counter() - self._mark
        return (self._wall + timedelta(seconds=elapsed)).isoformat()


_CLOCK = _Clock()


def _now() -> str:
    return _CLOCK.now()


_add = combine


def solve(
    prompt,
    backend: ModelClient,
    *,
    instruction: str | None = None,
    config: Config | None = None,
    runtime_factory: Callable | None = None,
    trace=None,
    allowance: Allowance | None = None,
    depth: int = 0,
    cancel: threading.Event | None = None,
    tools: Mapping[str, Callable] | None = None,
    run_id: str | None = None,
    parent_run_id: str | None = None,
) -> Answer:
    config = config or load_config()
    run_id = run_id or uuid.uuid4().hex
    runtime_factory = runtime_factory or RUNTIMES[config.runtime]
    # Validated here rather than inside a cell: a bad tool is a caller's
    # mistake and should surface before a single call is paid for. What counts
    # as bad depends on the runtime, which says whether it rebuilds tools from
    # text or takes the objects.
    prepared = describe(
        tools, need_source=getattr(runtime_factory, "NEEDS_SOURCE", True)
    )
    if allowance is None:
        # What a live agent costs is the runtime's business, not the
        # config's: the same number is a harmless throttle over one
        # runtime and gigabytes of resident memory over another.
        ceiling = getattr(runtime_factory, "MAX_LIVE", config.max_live)
        allowance = Allowance.from_config(
            dataclasses.replace(config, max_live=min(config.max_live, ceiling))
        )
    model = config.model_for(depth)
    can_recurse = config.enable_delegation and allowance.can_recurse(depth)

    def _abort_if_cancelled():
        if cancel is not None and cancel.is_set():
            raise Abandoned("a sibling failed; this branch was abandoned")

    def _llm(text) -> str:
        allowance.reserve()
        answer, usage = backend.complete(
            [{"role": "user", "content": str(text)}], model=model
        )
        allowance.settle(usage)
        return answer

    def _for_child(granted):
        """What a child receives: what the call named, else the configured default.

        A name rather than the function itself, because the model asks from
        inside the runtime where only the tool's own name exists.
        """
        if granted is None:
            return dict(tools or {}) if config.inherit_tools else None
        unknown = [name for name in granted if name not in (tools or {})]
        if unknown:
            raise RuntimeError(
                f"cannot grant {unknown!r}: this agent has no such tool. "
                f"Available: {sorted(tools or {})}"
            )
        return {name: tools[name] for name in granted}

    def _child(subprompt, instruction=None, token=None, granted=None):
        return solve(
            str(subprompt),
            backend,
            instruction=instruction,
            config=config,
            runtime_factory=runtime_factory,
            trace=trace,
            allowance=allowance,
            depth=depth + 1,
            cancel=token,
            tools=_for_child(granted),
            parent_run_id=run_id,
        ).output

    def _rlm(subprompt, instruction=None, tools=None):
        if not can_recurse:
            raise RuntimeError(TOO_DEEP)
        return _child(subprompt, instruction, cancel, tools)

    def _spread(work, items):
        """Run `work` over `items` concurrently, abandoning the rest on the
        first failure rather than paying for results nobody will read.

        A slot is claimed per child, for as long as that child lives, rather
        than for the batch up front. Claiming the batch's worth in advance
        made a nested fan-out share the ceiling badly: a parent's batch held
        its slots for the whole of its children's work, so the last branch to
        ask could find none left and run its own children strictly one at a
        time — a single starved branch, fully serial, while the rest were
        parallel. Measured at eight times slower than the same work spread
        flat.

        Per-child claiming spends the same ceiling evenly. A child that cannot
        get a slot still runs, because refusing to start would deadlock a tree
        against its own descendants; the ceiling stays a brake rather than a
        gate.
        """
        items = list(items)
        if not items:
            return []
        width = min(len(items), max(1, config.max_concurrent))
        token = threading.Event()
        pool = ThreadPoolExecutor(max_workers=width)

        def slotted(item, cancel_token):
            held = allowance.claim_slots(1)
            try:
                return work(item, cancel_token)
            finally:
                allowance.release_slots(held)

        futures = []
        try:
            futures = [pool.submit(slotted, item, token) for item in items]
            return [future.result() for future in futures]
        except BaseException:
            token.set()
            for future in futures:
                future.cancel()
            raise
        finally:
            pool.shutdown(wait=False)

    def _gather_rlm(subprompts, instruction=None, tools=None):
        if not can_recurse:
            raise RuntimeError(TOO_DEEP)
        return _spread(
            lambda item, token: _child(item, instruction, token, tools), subprompts
        )

    def _gather_llm(texts):
        return _spread(lambda item, token: _llm(item), texts)

    # Only what this agent is told about is bound. A name that exists but is
    # never described is a trap, and a cheaper name described beside a better
    # one is worse than a trap, because it gets used.
    # Bound at every depth, including the last one, where calling it raises
    # TOO_DEEP. A name that vanishes at the bottom of the tree gives a leaf a
    # NameError instead of an explanation, and a NameError is a puzzle rather
    # than an instruction. The prompt still only describes it where it works.
    bridges: dict[str, Callable] = {}
    if config.enable_delegation:
        bridges.update({"rlm": _rlm, "gather_rlm": _gather_rlm})
    messages = [
        {
            "role": "system",
            "content": system_prompt(
                can_recurse,
                prepared,
                sealed=getattr(runtime_factory, "SEALED", False),
            ),
        },
    ]
    total = Spend()
    text_prompt = str(prompt)
    emit(
        trace,
        "run_started",
        run_id=run_id,
        parent_run_id=parent_run_id,
        depth=depth,
        model=model,
        instruction=instruction,
        prompt_type=type(prompt).__name__,
        prompt_size=len(text_prompt),
    )

    # Asked before a sandbox is paid for. A child can be abandoned between
    # being handed to the pool and starting, and starting one costs a process —
    # under the sealed runtime a whole interpreter, seconds of it — and then an
    # opening cell, all of it for an answer nobody is waiting for any more.
    try:
        _abort_if_cancelled()
    except Abandoned as failure:
        emit(
            trace, "run_failed",
            run_id=run_id, error=f"{type(failure).__name__}: {failure}",
        )
        raise

    runtime = runtime_factory(prompt, bridges, config.timeout, prepared)

    def _snapshot(step: int) -> None:
        """Ask the runtime what is bound, and only if someone is watching.

        A snapshot costs a round trip to the sandbox. Nothing about the run
        depends on it, so it is skipped entirely when no sink wants it.
        """
        if not hasattr(trace, "namespace_changed"):
            return
        reader = getattr(runtime, "snapshot", None)
        if reader is None:
            return
        try:
            variables = reader()
        except Exception:
            return
        emit(trace, "namespace_changed", run_id=run_id, step=step, variables=variables)

    def _open() -> None:
        """Look at PROMPT once, before the model is asked for anything.

        The result becomes the first turn of the conversation, as the cell and
        the output it produced. No model call is made, so this step costs
        nothing and is recorded with no usage.
        """
        opening = opening_code()
        started = _now()
        emit(trace, "step_started", run_id=run_id, step=0, started=started)
        emit(trace, "code_generated", run_id=run_id, step=0, code=opening)
        cell = runtime.execute(opening)
        shown = cell.stdout + (f"\n{cell.error}" if cell.error else "")
        stamps = {"execution_start": started, "execution_end": _now()}
        emit(
            trace, "output_received",
            run_id=run_id, step=0, output=shown, error=bool(cell.error),
        )
        _log(trace, depth, run_id, parent_run_id, 0, opening, shown, bool(cell.error), Spend(), stamps)
        emit(
            trace, "step_completed",
            run_id=run_id, step=0, usage=Spend(), error=bool(cell.error),
            ended=stamps["execution_end"],
        )
        _snapshot(0)
        messages.append(
            {
                "role": "user",
                "content": opening_message(
                    opening, shown, instruction, config.truncate_len,
                    is_child=depth > 0,
                ),
            }
        )

    try:
        _open()
        for step in range(1, config.max_steps + 1):
            _abort_if_cancelled()
            allowance.reserve()
            llm_call_start = _now()
            emit(trace, "step_started", run_id=run_id, step=step, started=llm_call_start)
            text, usage = backend.complete(messages, model=model)
            llm_call_end = _now()
            allowance.settle(usage)
            total = _add(total, usage)
            messages.append({"role": "assistant", "content": text})

            banner = (
                budget_banner(step, config.max_steps) if config.enable_step_banner else ""
            )

            timestamps = {"llm_call_start": llm_call_start, "llm_call_end": llm_call_end}

            code = runnable_code(text)
            if code is None:
                emit(trace, "code_generated", run_id=run_id, step=step, code=None)
                emit(
                    trace, "output_received",
                    run_id=run_id, step=step, output=NO_CODE, error=True,
                )
                _log(trace, depth, run_id, parent_run_id, step, None, NO_CODE, True, usage, timestamps)
                emit(
                    trace, "step_completed",
                    run_id=run_id, step=step, usage=usage, error=True, ended=_now(),
                )
                messages.append({"role": "user", "content": banner + NO_CODE})
                continue

            emit(trace, "code_generated", run_id=run_id, step=step, code=code)

            # Only the first step, and only when the opening message showed a
            # sample rather than the whole value. After one step the model has
            # read something back, and a short prompt it was handed in full is
            # data it has genuinely seen.
            if (
                config.enable_first_look_guard
                and step == 1
                and not shows_everything(prompt)
                and answers_without_looking(code)
            ):
                notice = UNSEEN.format(preview=PREVIEW)
                emit(
                    trace, "output_received",
                    run_id=run_id, step=step, output=notice, error=True,
                )
                _log(trace, depth, run_id, parent_run_id, step, code, notice, True, usage, timestamps)
                emit(
                    trace, "step_completed",
                    run_id=run_id, step=step, usage=usage, error=True, ended=_now(),
                )
                messages.append({"role": "user", "content": banner + notice})
                continue

            timestamps["execution_start"] = _now()
            cell = runtime.execute(code)
            timestamps["execution_end"] = _now()
            output = cell.stdout + (f"\n{cell.error}" if cell.error else "")

            if cell.has_final:
                emit(
                    trace, "output_received",
                    run_id=run_id, step=step, output=output, error=False,
                )
                _log(trace, depth, run_id, parent_run_id, step, code, output, False, usage, timestamps)
                emit(
                    trace, "step_completed",
                    run_id=run_id, step=step, usage=usage, error=False, ended=_now(),
                )
                _snapshot(step)
                emit(
                    trace, "final",
                    result=cell.final, depth=depth,
                    run_id=run_id, parent_run_id=parent_run_id,
                )
                emit(trace, "run_completed", run_id=run_id, result=cell.final)
                return Answer(output=cell.final, steps=step, usage=total)

            labelled = label_output(output, config.truncate_len)
            emit(
                trace, "output_received",
                run_id=run_id, step=step, output=labelled, error=bool(cell.error),
            )
            _log(trace, depth, run_id, parent_run_id, step, code, labelled, bool(cell.error), usage, timestamps)
            emit(
                trace, "step_completed",
                run_id=run_id, step=step, usage=usage, error=bool(cell.error), ended=_now(),
            )
            _snapshot(step)
            messages.append({"role": "user", "content": f"{banner}Output:\n{labelled}"})

        raise StepsUsedUp(f"no FINAL() after {config.max_steps} steps")
    except BaseException as failure:
        emit(
            trace, "run_failed",
            run_id=run_id, error=f"{type(failure).__name__}: {failure}",
        )
        raise
    finally:
        runtime.close()


def _log(trace, depth, run_id, parent_run_id, step, code, output, error, usage, timestamps) -> None:
    """Write the after-the-fact record, for a sink that keeps one.

    Routed through `emit` like every other event, so a sink that only wants
    the live picture is not obliged to pretend it is a journal.
    """
    emit(
        trace,
        "step",
        step=step,
        code=code,
        output=output,
        error=error,
        usage=usage,
        depth=depth,
        run_id=run_id,
        parent_run_id=parent_run_id,
        timestamps=timestamps,
    )
