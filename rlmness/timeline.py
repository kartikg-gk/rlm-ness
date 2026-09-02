"""When each agent was alive, read back off a journal.

`rlmness-viewlog` answers what happened, in the order it happened. This
answers when, and by whom. They are different questions: a fan-out interleaves
its children into one file, so the reading order says almost nothing about
which agent was busy while which other one waited.

Each agent is one row, placed against the wall clock of the whole run, so
overlapping bars are exactly the work that ran at the same time.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

TRACES = Path("traces")
BAR = 48


def _moment(text):
    try:
        return datetime.fromisoformat(str(text).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None


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
                continue
    return found


def gather(records) -> list[dict]:
    """One entry per agent: when it started, when it stopped, how much it did.

    Grouped on `run_id` alone. Depth cannot do this job — every child of one
    batch shares a depth, and merging them would report a fan-out as a single
    long-running agent.
    """
    agents: dict[str, dict] = {}
    for record in records:
        run_id = record.get("run_id")
        if not run_id:
            continue
        agent = agents.setdefault(
            run_id,
            {
                "run_id": run_id,
                "parent": record.get("parent_run_id"),
                "depth": record.get("depth", 0),
                "start": None,
                "end": None,
                "steps": set(),
                "errors": 0,
            },
        )
        if record.get("kind") == "step":
            agent["steps"].add(record.get("step"))
            if record.get("error"):
                agent["errors"] += 1
        stamps = record.get("timestamps") or {}
        for key in ("llm_call_start", "llm_call_end", "execution_start", "execution_end"):
            moment = _moment(stamps.get(key))
            if moment is None:
                continue
            if agent["start"] is None or moment < agent["start"]:
                agent["start"] = moment
            if agent["end"] is None or moment > agent["end"]:
                agent["end"] = moment
    return sorted(
        agents.values(),
        key=lambda a: (a["start"] or datetime.max.replace(tzinfo=None), a["depth"]),
    )


def render(path: Path | str, width: int = BAR) -> str:
    path = Path(path)
    agents = gather(_records(path))
    if not agents:
        return f"{path}: no agent timestamps recorded\n"

    timed = [a for a in agents if a["start"] and a["end"]]
    if not timed:
        return f"{path}: no agent timestamps recorded\n"
    first = min(a["start"] for a in timed)
    last = max(a["end"] for a in timed)
    span = (last - first).total_seconds() or 1.0

    lines = [str(path), f"{len(agents)} agents over {span:.2f}s", ""]
    for agent in agents:
        name = "root" if agent["parent"] is None else agent["run_id"][:6]
        if not (agent["start"] and agent["end"]):
            lines.append(f"  d{agent['depth']} {name:>6}  (no timestamps)")
            continue
        begin = (agent["start"] - first).total_seconds()
        finish = (agent["end"] - first).total_seconds()
        left = int(begin / span * width)
        length = max(1, int((finish - begin) / span * width))
        bar = " " * left + "█" * min(length, width - left)
        note = f" {agent['errors']} err" if agent["errors"] else ""
        lines.append(
            f"  d{agent['depth']} {name:>6} |{bar:<{width}}| "
            f"{begin:6.2f}-{finish:6.2f}s  {len(agent['steps'])} steps{note}"
        )
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="rlmness-timeline")
    parser.add_argument("trace", nargs="?")
    parser.add_argument("--width", type=int, default=BAR)
    arguments = parser.parse_args(argv if argv is not None else sys.argv[1:])

    if arguments.trace is None:
        recent = (
            sorted(TRACES.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
            if TRACES.is_dir()
            else []
        )
        if not recent:
            print(f"no traces in {TRACES}/", file=sys.stderr)
            return 1
        print(f"recent traces in {TRACES}/:")
        for candidate in recent[:10]:
            print(f"  {candidate.name}")
        return 1

    path = Path(arguments.trace)
    if not path.exists():
        candidate = TRACES / arguments.trace
        if not candidate.exists():
            print(f"not found: {arguments.trace}", file=sys.stderr)
            return 1
        path = candidate

    print(render(path, width=arguments.width), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
