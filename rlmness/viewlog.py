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
    arguments = parser.parse_args(argv if argv is not None else sys.argv[1:])

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
