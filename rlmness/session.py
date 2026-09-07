"""What a session remembers between runs.

A run answers one question and throws its namespace away. A session keeps
that namespace, and the questions already answered, so the next run starts
where the last one stopped instead of from nothing.

The state is a plain JSON file. It has a version because it will be read by a
build that is not the one that wrote it, and an older reader that guesses is
worse than one that knows it is looking at something newer.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: Raised whenever a field is added that an older reader would need. Readers
#: accept anything at or below their own version and say so plainly otherwise.
VERSION = 1

PREVIEW = 1500


@dataclass
class Answered:
    """A question this session has already settled."""

    question: str
    answer: Any
    trace: str | None = None
    run_id: str | None = None


@dataclass
class Session:
    version: int = VERSION
    answered: list[Answered] = field(default_factory=list)
    #: Set while a question is being worked on and cleared when it is
    #: answered. A question still sitting here on load is one whose run died,
    #: which is worth saying rather than quietly dropping.
    asking: str | None = None
    cells: list[dict] = field(default_factory=list)
    variables: dict[str, dict] = field(default_factory=dict)
    functions: dict[str, str] = field(default_factory=dict)
    modules: dict[str, str] = field(default_factory=dict)
    dropped: dict[str, str] = field(default_factory=dict)

    # ---- persistence -----------------------------------------------------

    @classmethod
    def load(cls, path: Path | str) -> "Session":
        """Read a session, or start one if the file is not there yet.

        A missing file is the ordinary first run, not an error. A corrupt one
        is an error worth raising: silently starting fresh would throw away
        work the file might still have held.
        """
        path = Path(path)
        if not path.exists():
            return cls()
        raw = json.loads(path.read_text(encoding="utf-8"))
        found = int(raw.get("version", 1))
        if found > VERSION:
            raise ValueError(
                f"{path} was written by a newer build (version {found}, "
                f"this one reads {VERSION})"
            )
        return cls(
            version=found,
            answered=[Answered(**a) for a in raw.get("answered", [])],
            asking=raw.get("asking"),
            cells=raw.get("cells", []),
            variables=raw.get("variables", {}),
            functions=raw.get("functions", {}),
            modules=raw.get("modules", {}),
            dropped=raw.get("dropped", {}),
        )

    def save(self, path: Path | str) -> None:
        """Write the whole state, atomically.

        Through a temporary file and a rename: a run interrupted mid-write
        would otherwise leave a half-written file where the session used to
        be, and lose every question it had already answered.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(path.suffix + ".part")
        partial.write_text(
            json.dumps(
                {
                    "version": VERSION,
                    "answered": [vars(a) for a in self.answered],
                    "asking": self.asking,
                    "cells": self.cells,
                    "variables": self.variables,
                    "functions": self.functions,
                    "modules": self.modules,
                    "dropped": self.dropped,
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        partial.replace(path)

    # ---- what the guest sends back ---------------------------------------

    def absorb(self, swept: dict, cell: dict, kept: set[str]) -> None:
        """Take one step's sweep as the new picture of the namespace.

        The sweep is everything, not a change list, so it replaces rather than
        merges — a name the model deleted has to be able to disappear. The
        exception is a name that failed to restore into this run: the sweep
        cannot see it, and dropping it here would lose a value over a fault
        that had nothing to do with it.
        """
        self.cells.append(cell)
        variables = dict(swept.get("variables") or {})
        functions = dict(swept.get("functions") or {})
        for name in kept:
            if name not in variables and name in self.variables:
                variables[name] = self.variables[name]
            if name not in functions and name in self.functions:
                functions[name] = self.functions[name]
        self.variables = variables
        self.functions = functions
        self.modules = dict(swept.get("modules") or {})
        self.dropped = {**self.dropped, **(swept.get("dropped") or {})}
        for name in variables:
            self.dropped.pop(name, None)

    def settled(self, question: str, answer: Any, trace=None, run_id=None) -> None:
        self.answered.append(Answered(question, answer, str(trace) if trace else None, run_id))
        self.asking = None

    # ---- what the next run is told ---------------------------------------

    def preamble(self, show_code: bool = True) -> str:
        """The account of this session given to the run that resumes it.

        The questions and answers are here; the variables are not, because the
        opening step prints what is actually in the namespace and a list
        written here could disagree with it. One of the two has to be the
        authority and it should be the one reading the real thing.

        The code is included because how something was built is most of what
        makes it reusable — a resumed run that can see the earlier cells
        rewrites far less of them.
        """
        if not self.answered and not self.cells and self.asking is None:
            return ""
        lines = [
            "This run continues a session. Earlier runs are not replayed, but "
            "what they settled and what they built is below, and the variables "
            "they saved are already in your namespace — the opening step lists "
            "them. PROMPT holds the new question only.",
            "",
            "Already answered here:",
        ]
        for index, item in enumerate(self.answered, 1):
            lines.append(f"  [{index}] {_short(item.question)}")
            lines.append(f"      -> {_short(item.answer)}")
        if self.asking is not None:
            lines.append(
                f"  [{len(self.answered) + 1}] {_short(self.asking)}"
                "\n      -> unfinished: that run stopped before it answered. Its "
                "variables may still be here."
            )
        if self.dropped:
            lines.append("")
            lines.append("Not carried over:")
            for name, reason in self.dropped.items():
                lines.append(f"  {name}: {reason}")
        if self.cells and show_code:
            lines.append("")
            lines.append("Code from earlier runs, in order:")
            lines.append("```python")
            for cell in self.cells:
                mark = "" if cell.get("ok", True) else "  (failed)"
                lines.append(f"# question {cell.get('question', 0) + 1}, step {cell.get('step')}{mark}")
                body = (cell.get("code") or "").strip()
                lines.append(body if cell.get("ok", True) else f"# {_short(body, 120)}")
            lines.append("```")
        lines.append("")
        return "\n".join(lines)

    def state_for_guest(self) -> dict:
        return {
            "variables": self.variables,
            "functions": self.functions,
            "modules": self.modules,
        }


def _short(value, limit: int = PREVIEW) -> str:
    text = value if isinstance(value, str) else repr(value)
    if len(text) <= limit:
        return text
    return f"{text[:limit]}... [{len(text)} characters in all]"
