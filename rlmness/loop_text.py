"""The words of the loop and the checks made on a reply.

Plain functions over plain data, with nothing imported from the rest of the
package, so that an engine running this code in another interpreter is working
from the same text and the same rules rather than a copy of them.
"""

from __future__ import annotations

import ast
import json
import re

SCHEMA_SHOWN = "FINAL must be given a value matching this JSON Schema:\n{schema}\n---\n"

SCHEMA_REFUSED = (
    "FINAL was not accepted: the value does not match the output schema. "
    "Nothing else was lost. Every name you bound is still here, so put the "
    "value right and hand it to FINAL once more rather than redoing the "
    "work.\n\n"
    "Schema:\n{schema}\n\nWhat does not match:\n{problems}"
)

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


def _size(value) -> int:
    return len(value if isinstance(value, str) else json.dumps(value, default=str))


def handoff_question(prompt, pieces, min_chars: int, share: float) -> str | None:
    """What to ask before handing over pieces barely smaller than `prompt`, if anything.

    A whole batch is one question rather than one per child: a fan-out over a
    dozen pieces would otherwise cost a dozen extra calls to say the same thing
    about the same slice.
    """
    pieces = list(pieces)
    parent = _size(prompt)
    if parent < min_chars:
        return None
    oversized = [piece for piece in pieces if _size(piece) >= share * parent]
    if not oversized:
        return None
    if len(pieces) == 1:
        return ASK_ONE.format(
            parent=parent, child=_size(pieces[0]),
            share=round(_size(pieces[0]) / max(1, parent) * 100), preview=str(pieces[0])[:140],
        )
    lines = "\n".join(
        f"  [{n + 1}] {_size(piece):,} characters "
        f"({round(_size(piece) / max(1, parent) * 100)}% of yours): "
        f"{str(piece)[:140]}"
        for n, piece in enumerate(pieces)
    )
    return ASK_BATCH.format(
        count=len(pieces), parent=parent, oversized=len(oversized), lines=lines,
    )
