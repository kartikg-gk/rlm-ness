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
import os
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

#: How much earlier code a resumed run is shown, in characters. A session is
#: meant to be used for a long time, and without a bound the whole history of
#: it is prepended to every step of every later question.
CODE_BUDGET = 24_000

#: How many earlier questions are spelled out. Past this the count is given
#: instead: what those runs settled is what matters, and the namespace holds
#: the result of it.
ANSWERS_SHOWN = 12


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
    #: What the file looked like when this run last wrote it. A file that no
    #: longer matches was written by somebody else.
    _stamp: tuple | None = field(default=None, compare=False, repr=False)
    #: Set once this run has stopped writing to the path it was given, because
    #: another run had taken it over. Read by the caller so it can say where
    #: the work actually went.
    diverted: Path | None = field(default=None, compare=False, repr=False)

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
        if self._foreign():
            # Another run owns this file now. Overwriting it would throw away
            # whatever it has answered since, and refusing to write would
            # throw away this run instead, so this one steps aside and keeps
            # its own work under a name of its own.
            self.diverted = _beside(self.path)
            self.path = self.diverted
            self._stamp = None
        partial = self.path.with_suffix(self.path.suffix + ".part")
        partial.write_text(body, encoding="utf-8")
        partial.replace(self.path)
        self._written = body
        self._stamp = _stamp_of(self.path)

    def _foreign(self) -> bool:
        """Whether the file changed underneath this run since it last wrote.

        Only meaningful once this run has written at all: before that, a file
        that is already there is the session being resumed, not a rival.
        """
        if self._stamp is None:
            return False
        return _stamp_of(self.path) != self._stamp

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

    def probe(self, restore_failed=(), taken=()) -> str:
        """The inventory of what is actually bound, for the opening step.

        This belongs with the opening cell's output rather than in the
        preamble, because it is a report of the namespace as it stands and the
        preamble is a report of what earlier runs did. A name that failed to
        come back is listed as missing here even though the preamble still
        counts it as saved: the model is told what it has, not what it should
        have had.
        """
        failed = set(restore_failed)
        # Whatever the caller bound under its own name. A saved value that
        # collided with one was parked beside it, so it is listed the way it
        # is actually bound rather than the way it was saved.
        reserved = RESERVED | set(taken)
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
            shown = f"{name}_saved" if name in reserved else name
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
        hidden = len(self.answered) - ANSWERS_SHOWN
        if hidden > 0:
            lines.append(f"  [{hidden} earlier questions, not listed]")
        for index, item in enumerate(self.answered[-ANSWERS_SHOWN:], max(hidden, 0) + 1):
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
            shown, elided = self._recent_code()
            lines.append("")
            if elided:
                lines.append(
                    f"Code from earlier runs, most recent last. {elided} earlier "
                    "cells are not shown; what they built is in the namespace "
                    "either way."
                )
            else:
                lines.append("Code from earlier runs, in order:")
            lines.append("```python")
            for cell in shown:
                mark = "" if cell.get("ok", True) else "  (failed)"
                lines.append(f"# question {cell.get('question', 0) + 1}, step {cell.get('step')}{mark}")
                body = (cell.get("code") or "").strip()
                lines.append(body if cell.get("ok", True) else f"# {_short(body, 120)}")
            lines.append("```")
        lines.append("")
        return "\n".join(lines)

    def _recent_code(self) -> tuple[list[dict], int]:
        """As much of the earlier code as is worth carrying, newest first.

        Every step of every question would otherwise be prepended to every
        step of the next one, and the message list is already resent whole on
        each turn — so an old session pays for its whole history twice over,
        once per step, forever. The recent cells are the ones a resumed run
        actually builds on; the old ones describe a namespace it can simply
        look at.
        """
        kept, spent = [], 0
        for cell in reversed(self.cells):
            body = (cell.get("code") or "").strip()
            if kept and spent + len(body) > CODE_BUDGET:
                break
            if len(body) > CODE_BUDGET:
                # The newest cell is always shown, so a single cell larger
                # than the whole budget would otherwise carry straight past
                # it. A cell that size is almost always data pasted into
                # code, and its head says what it was.
                cell = {**cell, "code": body[:CODE_BUDGET] + (
                    f"\n# ... {len(body) - CODE_BUDGET} more characters of "
                    "this cell not shown"
                )}
                body = cell["code"]
            spent += len(body)
            kept.append(cell)
        kept.reverse()
        return kept, len(self.cells) - len(kept)

    def state_for_guest(self) -> dict:
        return {
            "variables": self.variables,
            "functions": self.functions,
            "modules": self.modules,
        }


def _stamp_of(path: Path) -> tuple | None:
    try:
        found = path.stat()
    except OSError:
        return None
    return (found.st_size, found.st_mtime_ns)


def _beside(path: Path) -> Path:
    """A free name next to one another run has taken."""
    for index in range(1, 1000):
        candidate = path.with_name(f"{path.stem}.{index}{path.suffix}")
        if not candidate.exists():
            return candidate
    return path.with_name(f"{path.stem}.{os.getpid()}{path.suffix}")


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
