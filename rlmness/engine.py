"""The execution loop."""

from __future__ import annotations

import ast
import dataclasses
import inspect
import json
import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Mapping

from .providers import NATIVE_CALL, ModelClient, Spend, combine
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
from .schema import Shape

SCHEMA_SHOWN = "FINAL must be given a value matching this JSON Schema:\n{schema}\n---\n"

SCHEMA_REFUSED = (
    "FINAL was not accepted: the value does not match the output schema. "
    "Nothing else was lost. Every name you bound is still here, so put the "
    "value right and hand it to FINAL once more rather than redoing the "
    "work.\n\n"
    "Schema:\n{schema}\n\nWhat does not match:\n{problems}"
)

RUNTIMES = {
    "subprocess": SubprocessRuntime,
    "wasm": WasmRuntime,
    "in-process": InProcessRuntime,
}

# The newline after the language tag is optional: models sometimes start the
# code on the fence's own line, as in ```python# first comment.
_FENCE = re.compile(r"```([A-Za-z0-9_.+-]*)[ \t]*(?:\r?\n)?(.*?)```", re.S)

# A call in a model's own markup, left as text by a provider that did not
# parse it. When it names code, the code inside is what the model meant to run.
_INVOKE = re.compile(r'<invoke name="([^"]+)">(.*?)</invoke>', re.S)
_PARAMETER = re.compile(r'<parameter name="([^"]+)">(.*?)</parameter>', re.S)
CODE_CALLS = {"code", "python", "run", "execute", "exec", "run_code", "execute_code"}
PYTHON_TAGS = {"", "python", "py", "python3"}

NO_CODE = (
    "No fenced Python block was found in your reply. Nothing ran. "
    "Send a single ```python block."
)

NO_TOOLS = (
    "That was a tool call, and nothing here runs tool calls, so nothing ran. "
    "rlm, gather_rlm and FINAL are Python functions: write the call inside a "
    "```python block, with await. Pass data by name, such as PROMPT['memo'] or "
    "a variable you bound, rather than pasting text you were shown — what is "
    "printed is cut short, so a pasted copy is not the whole of it."
)

TOO_DEEP = (
    "maximum recursion depth reached: rlm() is not available here. "
    "Solve this task yourself, slicing PROMPT in the namespace."
)

ASK_ONE = (
    "STOP. Confirm before this sub-agent call runs.\n"
    "Your own PROMPT is {parent:,} characters. The one you are handing over is "
    "{child:,} — {share}% of it, so the child is being asked to read what you "
    "have not reduced. It begins: {preview}\n"
    "Sub-agents pay off on pieces you have already narrowed: slice, filter or "
    "summarise in your own namespace first, then hand over the smaller result.\n"
    "Should this call go ahead? Put ALLOW or STOP on its own first line, then "
    "one line saying why. This answer does not run: write no code here, only "
    "the verdict."
)

ASK_BATCH = (
    "STOP. Confirm before these {count} sub-agent calls run.\n"
    "Your own PROMPT is {parent:,} characters. {oversized} of the pieces are a "
    "large, barely reduced share of it:\n{lines}\n"
    "Sub-agents pay off on pieces you have already narrowed: slice, filter or "
    "summarise in your own namespace first, then hand over the smaller "
    "results.\n"
    "Should the whole batch go ahead? Put ALLOW or STOP on its own first "
    "line, then one line saying why. This answer does not run: write no code "
    "here, only the verdict."
)

REFUSED_HANDOFF = (
    "The sub-agent call was not made. You were asked to confirm it and "
    "answered: {reason}\n"
    "Nothing was lost — every name you bound is still here. Reduce what you "
    "were about to hand over, then call again."
)

UNSEEN = (
    "Nothing ran. That block answers without reading PROMPT, and you have only "
    "been shown its first and last {preview} characters — the answer would be "
    "based on a sample rather than on the data.\n"
    "Look at PROMPT in the namespace first. If the answer genuinely does not "
    "depend on what is in it, print your reasoning this turn and call FINAL on "
    "the next one."
)


def _gist(text: str, limit: int = 300) -> str:
    """The verdict and its reason, without whatever the model wrote after them.

    The answer is quoted back into the refusal the agent reads, and a model
    asked a question at the end of its own conversation often keeps going:
    code, or a copy of what it was reading. Quoted whole, that buried the
    refusal under a page of the agent's own data and the agent drew the wrong
    conclusion from it.
    """
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    kept = " ".join(line for line in lines[:2] if not line.startswith("```"))
    kept = kept or "(no reason given)"
    return kept if len(kept) <= limit else kept[:limit].rstrip() + "…"


def approves(text: str) -> bool:
    """Whether a confirmation said anything other than stop.

    Open on anything unclear, on purpose. A check that halts whenever it
    cannot parse an answer spends a turn on every reply that opens with a
    word it did not expect, and the cost of letting a doubtful handoff run is
    one call. Only the word STOP holds it back.
    """
    first = re.search(r"[A-Za-z]+", text or "")
    return (first.group(0).upper() if first else "") != "STOP"


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


BLIND = (
    "That FINAL was not accepted. Its value is fixed text written in the same "
    "block that reads PROMPT, so it was settled before any of the output above "
    "existed. Check it against that output, then call FINAL on its own."
)


def writes_answer_blind(code: str) -> bool:
    """Whether a cell reads PROMPT and hands FINAL a value it cannot have read.

    A FINAL of fixed text in the same block as the code meant to produce the
    answer was written before that code ran. Nothing the block prints can
    have informed it, so it is a guess or a memory, never a result. A FINAL
    whose value is computed -- a name, an f-string over names, a call -- is
    the block's own answer and is left alone, and so is fixed text in a block
    that does not read PROMPT, which is the ordinary way to answer from
    output already seen.

    Holding the answer costs one step when it happens to be right. Every
    held answer found in recorded runs was wrong, recalled rather than read,
    or saved only by the block crashing before FINAL was reached.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    if not any(isinstance(node, ast.Name) and node.id == "PROMPT" for node in ast.walk(tree)):
        return False
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "FINAL"
        ):
            values = list(node.args) + [keyword.value for keyword in node.keywords]
            if not any(
                isinstance(inner, ast.Name) for value in values for inner in ast.walk(value)
            ):
                return True
    return False


class StepsUsedUp(Exception):
    pass


@dataclass
class Answer:
    output: Any
    steps: int
    usage: Spend


def _marked_up_code(text: str) -> list[str]:
    """Code inside a call written in a model's own markup, when nothing is fenced.

    Only calls that name code are read, and only their code: a call whose
    parameters are prose would run as a SyntaxError and teach nothing.
    """
    found = []
    for name, body in _INVOKE.findall(text):
        if name.strip().lower() not in CODE_CALLS:
            continue
        parameters = dict(_PARAMETER.findall(body))
        code = next(
            (parameters[key] for key in ("code", "source", "python") if key in parameters),
            None,
        )
        if code is None and not parameters:
            code = body
        if code and code.strip():
            found.append(code.strip("\n") + "\n")
    return found


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
    ] or _marked_up_code(text or "")
    # A model that shows the same block twice — restating it after a sentence
    # of explanation, or fencing one plan in two pieces — meant it to run
    # once. Joining both copies runs every statement twice, which redefines
    # names, doubles whatever was appended and fails outright on anything that
    # cannot be repeated.
    runnable, seen = [], set()
    for body in candidates:
        try:
            ast.parse(body)
        except SyntaxError:
            continue
        settled = body.strip()
        if settled in seen:
            continue
        seen.add(settled)
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
        f"[Turns left once this one ends: {remaining} of {max_steps}]\n"
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


def _accepts(factory, keyword: str) -> bool:
    """Whether a runtime factory can be handed this keyword.

    A keyword added after a runtime or a test double was written would break
    it if passed unconditionally. Asking first keeps those working.
    """
    try:
        parameters = inspect.signature(factory).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(
        parameter.name == keyword or parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters
    )


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
    session=None,
    output_schema=None,
) -> Answer:
    """Answer a question about `prompt`, which the model never sees whole.

    `instruction` is the question, kept apart from the data so the agent is
    told what it is looking for rather than having to find that out first.

    A caller holding one string that already contains both can pass it as
    `prompt` and leave `instruction` unset. If so, put the question at the
    very start or the very end of it. The opening step shows the model the
    head and the tail of PROMPT and nothing in between, so a question buried
    in the middle is one the agent has to go hunting for before it can begin —
    and hunting for it looks exactly like hunting for the answer, which is how
    a run spends half its steps before it starts.
    """
    config = config or load_config()
    # Compiled before anything is started, so a malformed schema is reported
    # as the caller's mistake without a sandbox or a model call being paid for.
    shape = (
        Shape(output_schema)
        if output_schema is not None and config.enable_structured_output
        else None
    )
    # With structured input off, a dict or a list arrives as the JSON text it
    # would print as, so the run it measures sees text and nothing else.
    if not config.enable_structured_output and not isinstance(prompt, str):
        prompt = json.dumps(prompt, default=str)
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
        answer, usage, _ = _reply(
            backend, [{"role": "user", "content": str(text)}], model
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

    def _handed(piece):
        """What a sub-agent is given as its PROMPT.

        Text stays text. A dict or a list is passed as itself, so a piece with
        structure reaches the sub-agent with that structure rather than as
        text it has to parse back apart. Anything else is refused: a number or
        an object is not a piece of the data, and turning it into text would
        hide the mistake that sent it.
        """
        if isinstance(piece, str):
            return piece
        if isinstance(piece, (dict, list)):
            return piece if config.enable_structured_output else json.dumps(piece, default=str)
        raise TypeError(
            f"a sub-agent is given a string, a dict or a list, not a "
            f"{type(piece).__name__}"
        )

    def _size(value) -> int:
        return len(value if isinstance(value, str) else json.dumps(value, default=str))

    def _too_big(pieces) -> list:
        """Which pieces are a large, barely reduced share of this agent's own PROMPT."""
        parent = _size(prompt)
        if parent < config.handoff_min_chars:
            return []
        return [
            piece for piece in pieces
            if _size(piece) >= config.handoff_share * parent
        ]

    def _confirm(question: str) -> tuple[bool, str]:
        """Put a handoff to this agent's own model, and charge it like any call."""
        allowance.reserve()
        asked = messages + [{"role": "user", "content": question}]
        text, usage, _ = _reply(backend, asked, model)
        allowance.settle(usage)
        _count(usage)
        return approves(text), _gist(text)

    def _count(usage) -> None:
        """Add spending to this agent's total, from whichever thread it happened on.

        A fan-out's children finish on threads of their own, and so does a
        handoff check, so the total is guarded rather than read and rewritten
        by several at once.
        """
        nonlocal total
        with counting:
            total = _add(total, usage)

    def _permitted(pieces) -> None:
        """Ask before handing over what the parent has not reduced.

        A whole batch is one question rather than one per child: a fan-out over
        a dozen pieces would otherwise cost a dozen extra calls to say the same
        thing about the same slice.
        """
        if not config.enable_handoff_guard:
            return
        pieces = list(pieces)
        oversized = _too_big(pieces)
        if not oversized:
            return
        parent = _size(prompt)
        if len(pieces) == 1:
            shown = str(pieces[0])[:140]
            question = ASK_ONE.format(
                parent=parent, child=_size(pieces[0]),
                share=round(_size(pieces[0]) / max(1, parent) * 100), preview=shown,
            )
        else:
            lines = "\n".join(
                f"  [{n + 1}] {_size(piece):,} characters "
                f"({round(_size(piece) / max(1, parent) * 100)}% of yours): "
                f"{str(piece)[:140]}"
                for n, piece in enumerate(pieces)
            )
            question = ASK_BATCH.format(
                count=len(pieces), parent=parent, oversized=len(oversized), lines=lines,
            )
        allowed, reason = _confirm(question)
        if not allowed:
            raise RuntimeError(REFUSED_HANDOFF.format(reason=reason))

    def _child(piece, instruction=None, token=None, granted=None, schema=None):
        # What a sub-agent spent is part of what this agent spent. Returning
        # only its answer left the reported cost at the root's own calls, so a
        # run that fanned out showed a fraction of its bill.
        answer = solve(
            _handed(piece),
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
            output_schema=schema,
        )
        _count(answer.usage)
        return answer.output

    def _rlm(piece, instruction=None, tools=None, schema=None):
        if not can_recurse:
            raise RuntimeError(TOO_DEEP)
        _permitted([piece])
        return _child(piece, instruction, cancel, tools, schema)

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

    def _gather_rlm(pieces, instruction=None, tools=None, schema=None):
        if not can_recurse:
            raise RuntimeError(TOO_DEEP)
        pieces = list(pieces)
        _permitted(pieces)
        return _spread(
            lambda item, token: _child(item, instruction, token, tools, schema), pieces
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
                structured=config.enable_structured_output,
                schema=shape.text if shape is not None else None,
            ),
        },
    ]
    total = Spend()
    counting = threading.Lock()
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

    # A session belongs to the run that was asked the question. Children get
    # their own empty namespace: a sub-agent is given a piece and a question,
    # and inheriting the parent's working variables would hand it the parent's
    # half-finished thinking as though it were data.
    carried = session.state_for_guest() if (session is not None and depth == 0) else None
    # Passed as a keyword and only when there is one, so a runtime or a test
    # double that predates sessions keeps its old signature.
    extra = {"session": carried} if carried is not None else {}
    # A sub-agent call handed to asyncio.gather skips everything the gather
    # helper does for a fan-out: the limit on how many run at once, the slot
    # each child holds, and stopping the rest when one piece fails. Refused
    # where the runtime can refuse it, and only where delegation is bound.
    if (
        config.enable_batching_guard
        and "rlm" in bridges
        and _accepts(runtime_factory, "batch_only")
    ):
        extra["batch_only"] = {"rlm": "gather_rlm"}
    runtime = runtime_factory(prompt, bridges, config.timeout, prepared, **extra)

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

    _session_preamble = (
        session.preamble() if (session is not None and depth == 0) else ""
    )
    if session is not None and depth == 0:
        # Written down before a single step runs, so a run that dies leaves a
        # record that it was asked at all. The next run's preamble says so,
        # and says the variables of the dead one may still be here.
        session.asking = instruction or str(prompt)
        session.save()

    def _keep(step_number: int, code_text: str | None, ok: bool) -> None:
        """Take everything the namespace holds into the session.

        After every step rather than at the end, and written down there and
        then: a run that dies at step nine should not cost the eight steps of
        work that came before it, and a run that is killed outright never
        reaches an end to be saved at.
        """
        if session is None or depth != 0:
            return
        session.absorb(
            runtime.sweep(code_text),
            {"question": len(session.answered), "step": step_number,
             "ok": ok, "code": code_text or ""},
            getattr(runtime, "restore_failed", set()),
        )
        session.save()

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
        if shape is not None:
            shown = SCHEMA_SHOWN.format(schema=shape.text) + shown
        # The inventory of what a session put back goes here, with the output
        # of the cell that looked at the namespace, rather than in the
        # preamble. It is a report of what is bound right now, and the model
        # should read it in the same place it reads everything else it knows
        # about the namespace.
        if session is not None and depth == 0:
            shown += session.probe(
                getattr(runtime, "restore_failed", ()),
                [tool.name for tool in prepared],
            )
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
                "content": (_session_preamble or "") + opening_message(
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
            text, usage, reasoning = _reply(backend, messages, model)
            llm_call_end = _now()
            allowance.settle(usage)
            _count(usage)
            messages.append({"role": "assistant", "content": text})

            banner = (
                budget_banner(step, config.max_steps) if config.enable_step_banner else ""
            )

            timestamps = {"llm_call_start": llm_call_start, "llm_call_end": llm_call_end}

            code = runnable_code(text)
            if code is None:
                # A model that reached for its own tool-call format was trying
                # to act, and is told how to say the same thing here. Told only
                # that it sent no code, it repeats the call.
                notice = NO_TOOLS if NATIVE_CALL in text else NO_CODE
                emit(trace, "code_generated", run_id=run_id, step=step, code=None)
                emit(
                    trace, "output_received",
                    run_id=run_id, step=step, output=notice, error=True,
                )
                _log(trace, depth, run_id, parent_run_id, step, None, notice, True, usage, timestamps, reasoning)
                emit(
                    trace, "step_completed",
                    run_id=run_id, step=step, usage=usage, error=True, ended=_now(),
                )
                messages.append({"role": "user", "content": banner + notice})
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
                _log(trace, depth, run_id, parent_run_id, step, code, notice, True, usage, timestamps, reasoning)
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

            # The block still ran, so nothing it did is lost. Only its answer
            # is set aside, and the model is shown the output it wrote that
            # answer without seeing.
            held = (
                cell.final_given
                and config.enable_blind_final_guard
                and writes_answer_blind(code)
            )
            # Checked only once the answer would otherwise stand. A refusal
            # keeps the block's work, like a held answer, and says what to fix.
            refused = (
                shape.problems(cell.final)
                if shape is not None and cell.final_given and not held
                else []
            )
            if cell.final_given and not held and not refused:
                emit(
                    trace, "output_received",
                    run_id=run_id, step=step, output=output, error=False,
                )
                _log(trace, depth, run_id, parent_run_id, step, code, output, False, usage, timestamps, reasoning)
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
                _keep(step, code, not cell.error)
                if session is not None and depth == 0:
                    session.settled(
                        instruction or str(prompt), cell.final,
                        trace=getattr(trace, "path", None), run_id=run_id,
                    )
                    session.save()
                return Answer(output=cell.final, steps=step, usage=total)

            _keep(step, code, not cell.error)
            labelled = label_output(output, config.truncate_len)
            if held:
                labelled += "\n\n" + BLIND
            if refused:
                labelled += "\n\n" + SCHEMA_REFUSED.format(
                    schema=shape.text,
                    problems="\n".join(f"  - {line}" for line in refused),
                )
            failed = bool(cell.error) or bool(refused)
            emit(
                trace, "output_received",
                run_id=run_id, step=step, output=labelled, error=failed,
            )
            _log(trace, depth, run_id, parent_run_id, step, code, labelled, failed, usage, timestamps, reasoning)
            emit(
                trace, "step_completed",
                run_id=run_id, step=step, usage=usage, error=failed, ended=_now(),
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


def _reply(backend, messages, model):
    """One model call, whichever shape the backend answers in.

    A backend may report the model's reasoning alongside the reply, and most
    do not. Both are accepted rather than requiring every caller and every
    test double to grow a third item they have nothing to put in.
    """
    answer = backend.complete(messages, model=model)
    if len(answer) == 3:
        return answer
    text, usage = answer
    return text, usage, None


def _log(trace, depth, run_id, parent_run_id, step, code, output, error, usage,
         timestamps, reasoning=None) -> None:
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
        reasoning=reasoning,
    )
