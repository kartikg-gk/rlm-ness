"""Per-step JSONL trace."""

from __future__ import annotations

import json
import secrets
import threading
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from .providers import Spend

TRACE_DIR = Path("traces")


def default_path(name: str = "run") -> Path:
    """A path no other run will pick, however close together they start.

    Records are appended, so two runs that agree on a filename interleave
    into one file and neither can be read afterwards. A second-resolution
    stamp agrees far too easily: the clock behind it moves in 16ms jumps on
    some platforms, so back-to-back runs land on the same value. The suffix
    makes the name unique without giving up a stamp that sorts.
    """
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return TRACE_DIR / f"{name}_{stamp}_{secrets.token_hex(3)}.jsonl"


class Journal:
    def __init__(self, path: Path | str | None = None):
        self.path = Path(path) if path is not None else default_path()
        self._depths: dict = {}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # A batch traces from several threads into the one file.
        self._lock = threading.Lock()

    def _write(self, record: dict) -> None:
        line = json.dumps(record, default=str) + "\n"
        with self._lock, self.path.open("a", encoding="utf-8") as handle:
            handle.write(line)

    def step(
        self,
        *,
        step: int,
        code: str | None,
        output: str,
        error: bool,
        usage: Spend,
        depth: int = 0,
        run_id: str | None = None,
        parent_run_id: str | None = None,
        timestamps: dict | None = None,
        reasoning: str | None = None,
    ) -> None:
        record = {
            "kind": "step",
            "depth": depth,
            "run_id": run_id,
            "parent_run_id": parent_run_id,
            "step": step,
            "code": code,
            "output": output,
            "error": error,
            "usage": asdict(usage),
            "timestamps": timestamps or {},
        }
        # Written only when there is some: most providers send none, and a
        # null on every step of every run is a lot of file for no fact.
        if reasoning:
            record["reasoning"] = reasoning
        self._write(record)

    def run_started(self, *, run_id, parent_run_id, depth, model=None,
                    instruction=None, **rest) -> None:
        """Open an agent's span.

        The steps already carry `run_id`, so a reader could infer an agent
        began by seeing its first step. It could not infer an agent that began
        and produced no step at all -- one that failed reserving its budget,
        or was abandoned before its first call -- and that is exactly the
        agent worth finding.
        """
        with self._lock:
            self._depths[run_id] = depth
        self._write(
            {
                "kind": "run_started",
                "depth": depth,
                "run_id": run_id,
                "parent_run_id": parent_run_id,
                "model": model,
                "instruction": instruction,
            }
        )

    def _depth_of(self, run_id) -> int:
        """The depth this run opened at.

        The closing events do not carry it -- their sinks take a fixed set of
        keywords and widening those to suit this one would break them -- so it
        is remembered from the opening instead. Every record in the file then
        carries a depth, and a reader can group by it without special-casing
        the two kinds that would otherwise lack one.
        """
        with self._lock:
            return self._depths.get(run_id, 0)

    def run_completed(self, *, run_id, result=None, **rest) -> None:
        self._write(
            {
                "kind": "run_completed",
                "run_id": run_id,
                "depth": self._depth_of(run_id),
            }
        )

    def run_failed(self, *, run_id, error=None, **rest) -> None:
        """Close the span of an agent that ended without an answer.

        Paired with `run_started`, this is what separates a tree that finished
        from one that stopped: without it a killed branch and a branch still
        thinking look identical in the file afterwards.
        """
        self._write(
            {
                "kind": "run_failed",
                "run_id": run_id,
                "depth": self._depth_of(run_id),
                "error": str(error) if error else None,
            }
        )

    def final(
        self,
        result,
        *,
        depth: int = 0,
        run_id: str | None = None,
        parent_run_id: str | None = None,
    ) -> None:
        self._write(
            {
                "kind": "final",
                "depth": depth,
                "run_id": run_id,
                "parent_run_id": parent_run_id,
                "result": result,
            }
        )
