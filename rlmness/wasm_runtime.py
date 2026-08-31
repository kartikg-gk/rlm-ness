"""CPython compiled to WebAssembly, hosted in a Node subprocess."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Callable, Mapping

from .runtime import ProtocolRuntime

_WORKER = str(Path(__file__).with_name("wasm_guest.mjs"))
_ROOT = str(Path(__file__).resolve().parents[1])


class WasmRuntime(ProtocolRuntime):
    #: Its own process, so a tool has to be rebuilt from text.
    NEEDS_SOURCE = True

    def __init__(
        self,
        prompt,
        bridges: Mapping[str, Callable] = (),
        timeout: float = 180.0,
        tools=(),
        node: str = "node",
    ):
        if shutil.which(node) is None:
            raise RuntimeError(f"{node!r} is not on PATH")
        process = subprocess.Popen(
            [node, _WORKER],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            bufsize=1,
            cwd=_ROOT,
        )
        super().__init__(process, prompt, bridges, timeout, tools)


def wasm_available(node: str = "node") -> bool:
    if shutil.which(node) is None:
        return False
    try:
        probe = subprocess.run(
            [node, "-e", "require.resolve('pyodide')"],
            cwd=_ROOT,
            capture_output=True,
            timeout=30,
        )
        return probe.returncode == 0
    except Exception:
        return False
