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


def render(path: Path | str, width: int | None = WIDTH) -> str:
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
        agents.add(depth)

        if record.get("kind") == "final":
            lines.append(f"{pad}final: {record.get('result')!r}")
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

    print(render(path, width=None if arguments.full else WIDTH), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
