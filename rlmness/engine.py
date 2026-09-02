"""The execution loop."""

from __future__ import annotations

import ast
import re
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Mapping

from .providers import ModelClient, Spend
from .limits import Allowance, Abandoned
from .config import Config, load_config
from .briefing import opening_message, system_prompt
from .wasm_runtime import WasmRuntime
from .tools import describe
from .in_process import InProcessRuntime
from .runtime import SubprocessRuntime

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


class StepsUsedUp(Exception):
    pass


@dataclass
class Answer:
    output: Any
    steps: int
    usage: Spend


def first_runnable_block(text: str) -> str | None:
    """Pick the block the model meant to run.

    Models sketch before they commit, and a fenced sketch looks exactly like a
    fenced answer. Whether it parses is the only test that tells them apart, so
    the first candidate that compiles wins. If none do, hand back the first
    anyway: the resulting SyntaxError is the feedback that lets the model fix
    itself.
    """
    candidates = [
        body
        for tag, body in _FENCE.findall(text or "")
        if tag.lower() in PYTHON_TAGS
    ]
    for body in candidates:
        try:
            ast.parse(body)
        except SyntaxError:
            continue
        return body
    return candidates[0] if candidates else None


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


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _add(left: Spend, right: Spend) -> Spend:
    # An unknown cost on either side leaves the sum unknown rather than
    # silently reading as free.
    if left.cost is None and right.cost is None:
        cost = None
    else:
        cost = (left.cost or 0.0) + (right.cost or 0.0)
    return Spend(
        prompt_tokens=left.prompt_tokens + right.prompt_tokens,
        completion_tokens=left.completion_tokens + right.completion_tokens,
        total_tokens=left.total_tokens + right.total_tokens,
        cost=cost,
    )


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
    allowance = allowance if allowance is not None else Allowance.from_config(config)
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
        first failure rather than paying for results nobody will read."""
        items = list(items)
        if not items:
            return []
        wanted = min(len(items), max(1, config.max_concurrent))
        taken = allowance.claim_slots(wanted)
        token = threading.Event()
        pool = ThreadPoolExecutor(max_workers=max(1, taken))
        futures = []
        try:
            futures = [pool.submit(work, item, token) for item in items]
            return [future.result() for future in futures]
        except BaseException:
            token.set()
            for future in futures:
                future.cancel()
            raise
        finally:
            pool.shutdown(wait=False)
            allowance.release_slots(taken)

    def _gather_rlm(subprompts, instruction=None, tools=None):
        if not can_recurse:
            raise RuntimeError(TOO_DEEP)
        return _spread(
            lambda item, token: _child(item, instruction, token, tools), subprompts
        )

    def _gather_llm(texts):
        return _spread(lambda item, token: _llm(item), texts)

    bridges = {"llm": _llm, "gather_llm": _gather_llm}
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
        {"role": "user", "content": opening_message(prompt, instruction)},
    ]
    total = Spend()
    runtime = runtime_factory(prompt, bridges, config.timeout, prepared)

    try:
        for step in range(1, config.max_steps + 1):
            _abort_if_cancelled()
            allowance.reserve()
            llm_call_start = _now()
            text, usage = backend.complete(messages, model=model)
            llm_call_end = _now()
            allowance.settle(usage)
            total = _add(total, usage)
            messages.append({"role": "assistant", "content": text})

            banner = (
                budget_banner(step, config.max_steps) if config.enable_step_banner else ""
            )

            timestamps = {"llm_call_start": llm_call_start, "llm_call_end": llm_call_end}

            code = first_runnable_block(text)
            if code is None:
                _log(trace, depth, run_id, parent_run_id, step, None, NO_CODE, True, usage, timestamps)
                messages.append({"role": "user", "content": banner + NO_CODE})
                continue

            timestamps["execution_start"] = _now()
            cell = runtime.execute(code)
            timestamps["execution_end"] = _now()
            output = cell.stdout + (f"\n{cell.error}" if cell.error else "")

            if cell.has_final:
                _log(trace, depth, run_id, parent_run_id, step, code, output, False, usage, timestamps)
                if trace:
                    trace.final(cell.final, depth=depth, run_id=run_id, parent_run_id=parent_run_id)
                return Answer(output=cell.final, steps=step, usage=total)

            labelled = label_output(output, config.truncate_len)
            _log(trace, depth, run_id, parent_run_id, step, code, labelled, bool(cell.error), usage, timestamps)
            messages.append({"role": "user", "content": f"{banner}Output:\n{labelled}"})

        raise StepsUsedUp(f"no FINAL() after {config.max_steps} steps")
    finally:
        runtime.close()


def _log(trace, depth, run_id, parent_run_id, step, code, output, error, usage, timestamps) -> None:
    if trace:
        trace.step(
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
