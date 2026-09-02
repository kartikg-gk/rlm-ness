"""System and opening messages."""

from __future__ import annotations

PREVIEW = 500

_BASE = """\
You answer questions about data you cannot see all at once.

The data is bound as PROMPT inside a persistent Python namespace. You never
receive it as text. To learn anything about it, you write code that inspects it
and read the output.

Each turn:
  - Reply with exactly one fenced Python block and nothing else that matters.
  - The block runs in the namespace. Names you bind persist into later turns.
  - Whatever the block prints is returned to you, truncated to its last
    characters if it is long. Print deliberately; do not dump the whole of
    PROMPT.

When you have the answer, call FINAL(answer) in a block. That ends the run and
returns the value. Call it with the answer itself, not a description of it.

Work in small steps. Look before you conclude."""

_FLAT = """\

Two helpers reach a language model from inside the namespace. Both are awaited.

  answer = await llm(text)
      One model call on the text you pass. No loop, no namespace of its own.
      Use it to read a slice you have already cut down to a readable size.

  answers = await gather_llm([text, text, ...])
      The same call over a list, run at the same time, results returned in the
      order you gave them.
"""

_RECURSIVE = """\
Two more spawn a whole sub-agent, also awaited.

  answer = await rlm(subprompt, instruction=None)
      It gets its own namespace with your subprompt bound as its PROMPT, works
      the same way you do, and returns what it passes to FINAL. It cannot see
      your variables and you cannot see its steps.

  answers = await gather_rlm([subprompt, ...], instruction=None)
      The same over a list, run at the same time, results in the order given.

A sub-agent handed nearly all of PROMPT has saved nothing and costs a full run.
Cut first, then delegate the cuts.
"""

_BATCHING = """
Calling a helper in a loop is the slowest thing you can do here. These two:

    answers = [await {one}(c) for c in chunks]     # one at a time
    answers = await {many}(chunks)                 # all at once

do identical work, and the second finishes in about the time of the slowest
chunk instead of the sum of all of them. Whenever you are asking the same
question of many pieces, put the pieces in a list and make one gather call.

`asyncio.gather` will NOT do this — these helpers reach the host one call at a
time, so gathering them yourself still runs them in series.
"""

_GUIDANCE = """
Split the work by what each side is good at. Code locates, counts, slices and
filters on exact text. The helpers judge meaning: what a passage is saying,
what stance it takes, whether it satisfies a description you can only phrase in
words.

When the question is about meaning, no amount of string matching will answer
it — the words you would search for are usually the ones a writer avoids. If
two searches have not found it, stop searching and hand the pieces to a helper.
That is what they are for, and it is almost always the shorter path from there.\
"""


_SEALED = """
Your code runs inside WebAssembly. There is no network and no host filesystem.
A module that reaches for either still imports — it fails when you call it —
and a tool you have been given is bound by this too, so work from PROMPT and
what is already here rather than fetching or reading.

Third-party packages are not installed: no pandas, no numpy, no requests. The
Python standard library is, so parse with it: `csv` for CSV or TSV,
`xml.etree.ElementTree` for XML, `tomllib` for TOML, `json` for JSON.
"""


def _tool_section(tools) -> str:
    """List the tools by name, signature and description.

    They are ordinary functions in the namespace, so they are described as
    such — an agent that is told about a helper it cannot reach wastes a turn
    discovering that, so only what this agent has is listed.
    """
    if not tools:
        return ""
    lines = ["", "These names are already bound in the namespace:", ""]
    for tool in tools:
        shape = f"{tool.name}{tool.signature}" if tool.callable else tool.name
        lines.append(f"  {shape}")
        if tool.description:
            for line in tool.description.splitlines():
                lines.append(f"      {line}")
    lines.append("")
    if any(tool.callable for tool in tools):
        lines.append(
            "Call the functions as you would any function — no await needed."
        )
    if any(not tool.callable for tool in tools):
        lines.append("The rest are values; read them directly.")
    return "\n".join(lines) + "\n"


def system_prompt(can_recurse: bool = False, tools=(), sealed: bool = False) -> str:
    """Describe only what the caller can actually reach.

    `llm` and `gather_llm` are flat calls and work at any depth, so a leaf
    agent gets them too. Advertising `rlm` to an agent that can only get an
    error from it wastes a turn.

    `sealed` says the runtime has no syscalls. Without it the model finds out
    by writing a fetch and reading the failure, which costs a turn to learn
    something the runtime knew all along.
    """
    parts = [_BASE, _FLAT]
    if can_recurse:
        parts.append(_RECURSIVE)
        parts.append(_BATCHING.format(one="rlm", many="gather_rlm"))
    else:
        parts.append(_BATCHING.format(one="llm", many="gather_llm"))
    parts.append(_GUIDANCE)
    if sealed:
        parts.append(_SEALED)
    parts.append(_tool_section(tools))
    return "".join(parts)


def shows_everything(prompt) -> bool:
    """Whether the opening carried the whole value, not a sample.

    Anything that reasons about what the model has actually been shown has to
    ask here rather than re-deriving the rule, or the two answers drift and
    the model gets told it has not seen something it was handed in full.
    """
    return len(str(prompt)) <= 2 * PREVIEW


OPENING_CODE = '''print("PROMPT type:", type(PROMPT).__name__)
print("PROMPT length:", len(PROMPT) if hasattr(PROMPT, "__len__") else "N/A")

if len(str(PROMPT)) > {preview}:
    print("first {preview} characters of str(PROMPT):", str(PROMPT)[:{preview}])
    print("---")
    print("last {preview} characters of str(PROMPT):", str(PROMPT)[-{preview}:])
else:
    print("PROMPT:", PROMPT)
'''


def opening_code() -> str:
    """The cell that looks at PROMPT before the model is asked anything."""
    return OPENING_CODE.format(preview=PREVIEW)


def opening_message(code: str, output: str, instruction: str | None = None,
                    truncate_len: int = 10000) -> str:
    """The first turn: a cell that ran, and what it printed.

    Written as an exchange rather than as a description, because the shape of
    the opening is the shape the model continues. A turn that reads "here is
    code, here is its output" is followed by more code; a paragraph describing
    the data is followed by more prose — and prose is how a model ends up
    answering from the summary it was shown instead of from the data itself.

    It is also true rather than illustrative: this code really ran in the
    namespace, and the output really is what it printed.
    """
    task = instruction or "answer the question in PROMPT."
    return (
        f"Outputs are truncated to their last {truncate_len} characters.\n\n"
        f"code:\n```python\n{code}```\n\n"
        f"Output:\n{output}\n\n"
        f"Task: {task}"
    )
