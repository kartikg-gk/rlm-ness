"""Where an installed copy keeps what does not ship inside the package."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

# A source checkout, when the package is run from one. Its own node_modules
# and rlmness.yaml keep working there without a setup step.
CHECKOUT = Path(__file__).resolve().parents[1]

PYODIDE = "^314.0.6"

STARTER = """\
# Models for the root agent and the sub-agents it starts.
primary_agent: z-ai/glm-5
sub_agent: minimax/minimax-m2.5

# wasm, subprocess or in-process
runtime: wasm
provider: openrouter

max_cost: 1.0
max_seconds: 1800
"""


def home() -> Path:
    return Path(os.environ.get("RLMNESS_HOME") or Path.home() / ".rlmness")


def _has_pyodide(root: Path) -> bool:
    return (root / "node_modules" / "pyodide").is_dir()


def node_root() -> Path:
    """The folder Node resolves pyodide from."""
    if _has_pyodide(home()):
        return home()
    if _has_pyodide(CHECKOUT):
        return CHECKOUT
    return home()


def pyodide_entry() -> Path:
    """The module the sandbox loads pyodide from."""
    return node_root() / "node_modules" / "pyodide" / "pyodide.mjs"


def _npm(args: list[str], cwd: Path) -> int:
    # npm is a .cmd script on Windows, which only a shell will run.
    return subprocess.run(["npm", *args], cwd=cwd, shell=os.name == "nt").returncode


def setup(print=print) -> int:
    """Install the sandbox into the home folder and leave a starter config."""
    if shutil.which("node") is None or shutil.which("npm") is None:
        print("Node.js 18 or newer is needed for the wasm runtime: https://nodejs.org")
        print("Without it, rlmness runs on the subprocess runtime.")
        return 1
    root = home()
    root.mkdir(parents=True, exist_ok=True)
    manifest = {"name": "rlmness-sandbox", "private": True, "type": "module",
                "dependencies": {"pyodide": PYODIDE}}
    (root / "package.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"installing the sandbox into {root} ...")
    if _npm(["install"], root) != 0:
        print("npm install failed; see the output above.")
        return 1
    config = root / "rlmness.yaml"
    if not config.exists():
        config.write_text(STARTER, encoding="utf-8")
        print(f"wrote a starter config to {config}")
    print("done. Try: rlmness \"How many r's are in strawberry?\"")
    return 0
