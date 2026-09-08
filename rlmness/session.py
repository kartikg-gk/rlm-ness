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

from .session_guest import RESERVED

#: A session addressed by a directory keeps its state under this name, so a
#: directory can hold the state beside whatever else belongs to the session.
STATE_FILE = "state.json"

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
    #: Where this session came from and where it goes back to. Bound by
    #: `load`, so every later save knows its own destination and a step can
    #: write itself down without being handed the path again.
    path: Path | None = field(default=None, compare=False, repr=False)
    #: Whether the code of earlier runs is shown to the next one. On by
    #: default: how something was built is most of what makes it reusable. Off
    #: for a long-lived session, where the dump grows without bound and
    #: eventually costs more than it saves.
    show_code: bool = field(default=True, compare=False, repr=False)
    #: What was last written there. A step that changed nothing should not
    #: rewrite the file: the write is the expensive part of saving often, and
    #: skipping it also narrows the window in which a crash lands mid-rename.
    _written: str = field(default="", compare=False, repr=False)

    # ---- persistence -----------------------------------------------------

    @staticmethod
    def resolve(target: Path | str, session_id: str | None = None) -> Path:
        """The state file named by a target that may be a file or a directory.

        A bare file path is the state itself. A directory holds it under a
        fixed name, and an id names a directory inside that one — which is how
        several sessions share a parent without sharing a state.
        """
        target = Path(target)
        if session_id:
            return target / session_id / STATE_FILE
        if target.is_dir() or target.suffix == "":
            return target / STATE_FILE
        return target

    @classmethod
    def load(cls, path: Path | str, session_id: str | None = None) -> "Session":
        """Read a session, or start one if the file is not there yet.

        A missing file is the ordinary first run, not an error. A corrupt one
        is an error worth raising: silently starting fresh would throw away
        work the file might still have held.
        """
        path = cls.resolve(path, session_id)
        if not path.exists():
            return cls(path=path)
        raw = json.loads(path.read_text(encoding="utf-8"))
        found = int(raw.get("version", 1))
        if found > VERSION:
            raise ValueError(
                f"{path} was written by a newer build (version {found}, "
                f"this one reads {VERSION})"
            )
        # Read at whatever version it was written, written back at this one.
        # Anything an older build did not record simply comes back missing and
        # takes its default, so a file only has to be migrated forwards once.
        return cls(
            path=path,
            version=VERSION,
            answered=[Answered(**a) for a in raw.get("answered", [])],
            asking=raw.get("asking"),
            cells=raw.get("cells", []),
            variables=raw.get("variables", {}),
            functions=raw.get("functions", {}),
            modules=raw.get("modules", {}),
            dropped=raw.get("dropped", {}),
        )

    def save(self, path: Path | str | None = None) -> None:
        """Write the whole state, atomically, and only when it changed.

        Through a temporary file and a rename: a run interrupted mid-write
        would otherwise leave a half-written file where the session used to
        be, and lose every question it had already answered.

        This is called after every step rather than once at the end, because
        the end is exactly what a killed process never reaches. A run stopped
        at step forty then resumes from step thirty-nine instead of from
        nothing. The unchanged check is what makes that affordable.
        """
        if path is not None:
            self.path = self.resolve(path)
        if self.path is None:
            return
        body = json.dumps(
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
        )
        if body == self._written:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        partial = self.path.with_suffix(self.path.suffix + ".part")
        partial.write_text(body, encoding="utf-8")
        partial.replace(self.path)
        self._written = body

    def clear(self) -> None:
        """Forget everything but where the state lives.

        The file is left alone until the next save, so clearing and then
        crashing leaves the old state readable rather than deleted.
        """
        self.answered = []
        self.asking = None
        self.cells = []
        self.variables = {}
        self.functions = {}
        self.modules = {}
        self.dropped = {}

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

    def probe(self, restore_failed=()) -> str:
        """The inventory of what is actually bound, for the opening step.

        This belongs with the opening cell's output rather than in the
        preamble, because it is a report of the namespace as it stands and the
        preamble is a report of what earlier runs did. A name that failed to
        come back is listed as missing here even though the preamble still
        counts it as saved: the model is told what it has, not what it should
        have had.
        """
        failed = set(restore_failed)
        lines = [
            "",
            "This namespace is kept between runs. Everything bound here when a "
            "step finishes is written down and comes back next time; anything "
            "that will not pickle is dropped and named. commit(name, note) "
            "keeps a value past the size limit and records what it is for.",
        ]
        living = [
            (name, meta) for name, meta in self.variables.items() if name not in failed
        ]
        if living or self.functions:
            lines.append("")
            lines.append("Restored into this namespace:")
        for name, meta in living:
            shown = f"{name}_saved" if name in RESERVED else name
            described = _describe(meta)
            lines.append(
                f"  {shown}: {meta.get('type', '?')} = {_short(meta.get('preview', ''), 160)}"
                + (f"  -- {described}" if described else "")
            )
        for name, source in self.functions.items():
            if name not in failed:
                lines.append(f"  {name}: defined in an earlier run")
        gone = {name: "did not come back" for name in failed}
        gone.update(self.dropped)
        if gone:
            lines.append("")
            lines.append("Not here:")
            for name, reason in gone.items():
                lines.append(f"  {name}: {reason}")
        lines.append("")
        return "\n".join(lines)

    def preamble(self, show_code: bool | None = None) -> str:
        """The account of this session given to the run that resumes it.

        The questions and answers are here; the variables are not, because
        `probe` lists what is actually bound and a second list written here
        could disagree with it. One of the two has to be the authority and it
        should be the one reporting the namespace itself.

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
        if self.cells and (self.show_code if show_code is None else show_code):
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


def _describe(meta: dict) -> str:
    """What was said about a value, in the order of how deliberate it was.

    A note was written by calling `commit`, a comment was written beside the
    assignment. Both are the model's own words, but only one of them was
    written to be read later.
    """
    parts = [meta.get("note"), meta.get("comment")]
    return "; ".join(part for part in parts if part)


def _short(value, limit: int = PREVIEW) -> str:
    text = value if isinstance(value, str) else repr(value)
    if len(text) <= limit:
        return text
    return f"{text[:limit]}... [{len(text)} characters in all]"
