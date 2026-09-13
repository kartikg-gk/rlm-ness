"""Read a run back.

A journal is written to be complete rather than readable: one JSON object per
step, every agent in the tree interleaved into one file in the order things
happened. That is the right thing to write and the wrong thing to look at, so
this turns it into something a person can follow — indented by depth, so a
sub-agent's work sits under the step that asked for it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

TRACES = Path("traces")
WIDTH = 400


def _records(path: Path) -> list[dict]:
    found = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                found.append(json.loads(line))
            except json.JSONDecodeError:
                # A run killed mid-write leaves a partial line. The steps
                # before it are still worth reading.
                continue
    return found


def _shorten(text: str, width: int | None) -> str:
    if width is None or len(text) <= width:
        return text
    return f"{text[:width]}\n    ... {len(text) - width} more characters"


def _indent(depth: int) -> str:
    return "  " * depth


def render(path: Path | str, width: int | None = WIDTH, reasoning: bool = False) -> str:
    path = Path(path)
    records = _records(path)
    if not records:
        return f"{path}: empty\n"

    lines = [str(path), ""]
    steps = 0
    agents = set()
    tokens = 0
    cost = 0.0
    priced = True

    for record in records:
        depth = record.get("depth", 0)
        pad = _indent(depth)
        kind = record.get("kind")
        # Counted by identity, not by depth. A fan-out puts many agents at one
        # depth, and counting depths reports a tree of nineteen as a tree of
        # four. A record that carries no identity at all falls back to its
        # depth, which is the best available answer and what this did for
        # every record before identity was written down.
        agents.add(record.get("run_id") or ("depth", depth))

        if kind == "final":
            lines.append(f"{pad}final: {record.get('result')!r}")
            lines.append("")
            continue
        # The record opens and closes each agent. Those are not steps, and
        # counting them as steps adds two to every agent in the tree.
        if kind in ("run_started", "run_completed", "run_failed"):
            if kind == "run_failed" and record.get("error"):
                lines.append(f"{pad}run failed: {record['error']}")
                lines.append("")
            continue

        steps += 1
        usage = record.get("usage") or {}
        tokens += usage.get("total_tokens") or 0
        if usage.get("cost") is None:
            priced = False
        else:
            cost += usage["cost"]

        mark = " [error]" if record.get("error") else ""
        lines.append(f"{pad}step {record.get('step')}{mark}")
        code = record.get("code")
        if code:
            for line in _shorten(code.strip(), width).splitlines():
                lines.append(f"{pad}  | {line}")
        output = (record.get("output") or "").strip()
        if output:
            for line in _shorten(output, width).splitlines():
                lines.append(f"{pad}  > {line}")
        thinking = (record.get("reasoning") or "").strip()
        if thinking:
            # Off unless asked for. It is usually the longest thing in a
            # record and is wanted only when the question is why a step went
            # the way it did, so the marker says it is there and the flag
            # prints it.
            if reasoning:
                for line in _shorten(thinking, width).splitlines():
                    lines.append(f"{pad}  ? {line}")
            else:
                lines.append(f"{pad}  ? [reasoning recorded; --reasoning to show]")
        lines.append("")

    spend = f"${cost:.4f}" if priced else "unknown (a provider reported no price)"
    lines.append(
        f"{steps} steps across {len(agents)} agents  "
        f"{tokens} tokens  cost {spend}"
    )
    return "\n".join(lines) + "\n"


def _line(value, width: int = 120) -> str:
    text = " ".join(str(value).split())
    return text if len(text) <= width else text[: width - 1] + "…"


def _tree_of(records: list[dict], run_id: str | None) -> list[dict]:
    """The records of one run and every agent it started.

    Several runs can share a file. A question's cost is its own tree's, so
    agents are followed down from the root by who started them rather than
    taken from the whole file.
    """
    if run_id is None:
        return records
    members = {run_id}
    grew = True
    while grew:
        grew = False
        for record in records:
            child, parent = record.get("run_id"), record.get("parent_run_id")
            if child and parent in members and child not in members:
                members.add(child)
                grew = True
    return [record for record in records if record.get("run_id") in members]


def _totals(records: list[dict]) -> dict:
    steps = tokens = 0
    cost = 0.0
    priced = True
    for record in records:
        if record.get("kind") in ("final", "run_started", "run_completed", "run_failed"):
            continue
        steps += 1
        usage = record.get("usage") or {}
        tokens += usage.get("total_tokens") or 0
        if usage.get("cost") is None:
            priced = False
        else:
            cost += usage["cost"]
    return {"steps": steps, "tokens": tokens, "cost": cost, "priced": priced}


def _locate(trace: str, state: Path) -> Path | None:
    """Where a trace named in a session is now.

    A trace is recorded as the run saw it, which is usually relative to the
    directory the run was started from, so the session's own directory is
    tried as well as the current one.
    """
    written = Path(trace)
    for candidate in (written, state.parent / written, state.parent.parent / written):
        if candidate.is_file():
            return candidate
    return None


def render_session(state: Path | str) -> str:
    """A session's questions in order, each with its answer and what its run cost."""
    from .session import Session

    state = Path(state)
    book = Session.load(state)
    lines = [
        f"session   {state.parent}",
        f"state     {state}  (version {book.version})",
        f"answered  {len(book.answered)}",
    ]
    if book.asking is not None:
        lines.append(f"unfinished {_line(book.asking)}")
    lines.append(
        f"kept      {len(book.variables)} variables, {len(book.functions)} functions, "
        f"{len(book.dropped)} dropped"
    )
    lines.append("")

    linked = 0
    tokens = 0
    cost = 0.0
    priced = True
    for number, answered in enumerate(book.answered, 1):
        lines.append(f"[{number}] {_line(answered.question)}")
        lines.append(f"    FINAL  {_line(repr(answered.answer))}")
        if not answered.trace:
            lines.append("    trace  (not linked)")
        else:
            found = _locate(answered.trace, state)
            if found is None:
                lines.append(f"    trace  {answered.trace}  (missing: moved or deleted)")
            else:
                run = _totals(_tree_of(_records(found), answered.run_id))
                linked += 1
                tokens += run["tokens"]
                if run["priced"]:
                    cost += run["cost"]
                else:
                    priced = False
                spend = f"${run['cost']:.4f}" if run["priced"] else "unknown"
                lines.append(
                    f"    run    {run['steps']} steps, {run['tokens']} tokens, cost {spend}"
                )
                lines.append(f"    trace  {found}  (rlmness-viewlog {found})")
        lines.append("")

    spend = f"${cost:.4f}" if priced else "unknown (a provider reported no price)"
    lines.append(
        f"session total  {tokens} tokens  cost {spend}  "
        f"across {linked} of {len(book.answered)} runs"
    )
    return "\n".join(lines) + "\n"


def _recent(limit: int = 10) -> list[Path]:
    if not TRACES.is_dir():
        return []
    return sorted(TRACES.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)[
        :limit
    ]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="rlmness-viewlog")
    parser.add_argument("trace", nargs="?")
    parser.add_argument(
        "--full",
        action="store_true",
        help="show code and output in full rather than shortened",
    )
    parser.add_argument(
        "--reasoning",
        action="store_true",
        help="show what the model said it was thinking, where a provider sent it",
    )
    parser.add_argument(
        "--session-id",
        help="read the session of this name inside a session directory",
    )
    arguments = parser.parse_args(argv if argv is not None else sys.argv[1:])

    if arguments.trace is not None:
        target = Path(arguments.trace)
        if arguments.session_id or target.is_dir() or target.suffix == ".json":
            from .session import Session

            state = Session.resolve(target, arguments.session_id)
            if not state.is_file():
                print(f"no session at {state}", file=sys.stderr)
                return 1
            try:
                print(render_session(state), end="")
            except ValueError as failure:
                print(str(failure), file=sys.stderr)
                return 1
            return 0

    if arguments.trace is None:
        recent = _recent()
        if not recent:
            print(f"no traces in {TRACES}/", file=sys.stderr)
            return 1
        print(f"recent traces in {TRACES}/:")
        for path in recent:
            print(f"  {path.name}")
        return 1

    path = Path(arguments.trace)
    if not path.exists():
        candidate = TRACES / arguments.trace
        if candidate.exists():
            path = candidate
        else:
            print(f"not found: {arguments.trace}", file=sys.stderr)
            return 1

    print(
        render(
            path,
            width=None if arguments.full else WIDTH,
            reasoning=arguments.reasoning,
        ),
        end="",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
