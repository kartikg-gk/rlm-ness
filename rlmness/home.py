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
# Models. The root agent answers the question; sub-agents read the pieces it
# hands out, so a cheaper model there is what makes delegating worth doing.
primary_agent: z-ai/glm-5
sub_agent: minimax/minimax-m2.5

# openrouter, anthropic or deepseek. Each reads its own API key from your
# shell — OPENROUTER_API_KEY, ANTHROPIC_API_KEY or DEEPSEEK_API_KEY.
provider: openrouter

# Where the model's code runs: wasm (sealed, needs `rlmness --setup`),
# subprocess (faster, can reach your machine) or in-process (no isolation).
runtime: wasm

# What one run may spend, sub-agents included.
max_cost: 1.0
max_seconds: 1800
max_steps: 20
max_depth: 3

# Uncomment anything below to change it.

# max_concurrent: 16          # sub-agents running at once in one batch
# max_tokens:                 # cap on one reply; set it on a low-credit key
# truncate_len: 10000         # characters of a cell's output the model sees
# timeout: 120                # seconds one cell may run
# temperature: 0.1
# reasoning_effort: low

# Ask before an agent hands a sub-agent most of its own input.
# enable_handoff_guard: true
# handoff_min_chars: 5000
# handoff_share: 0.6

# enable_delegation: true     # let agents start sub-agents at all
# enable_step_banner: true    # tell an agent how many turns it has left
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
