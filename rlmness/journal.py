"""Per-step JSONL trace."""

from __future__ import annotations

import json
import threading
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from .providers import Spend

TRACE_DIR = Path("traces")


def default_path(name: str = "run") -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return TRACE_DIR / f"{name}_{stamp}.jsonl"


class Journal:
    def __init__(self, path: Path | str | None = None):
        self.path = Path(path) if path is not None else default_path()
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
    ) -> None:
        self._write(
            {
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
