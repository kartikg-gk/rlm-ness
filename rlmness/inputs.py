"""Reading a prompt from a file."""

from __future__ import annotations

import json
from pathlib import Path

import yaml


class UnreadableInput(Exception):
    pass


def load_input(path: Path | str):
    """A file as the model should receive it.

    A structured file is parsed here and handed over as the object it
    describes, so the model indexes it instead of parsing text back apart.
    Anything else stays text: the model slices it in the REPL, which is the
    point of holding it there rather than in a context window. YAML is parsed
    here because the sandbox has no yaml module to parse it with.
    """
    path = Path(path)
    suffix = path.suffix.lower().lstrip(".")
    try:
        text = path.read_text(encoding="utf-8")
        if suffix == "json":
            return json.loads(text)
        if suffix in ("jsonl", "ndjson"):
            return [json.loads(line) for line in text.splitlines() if line.strip()]
        if suffix in ("yaml", "yml"):
            return yaml.safe_load(text)
        return text
    except OSError as failure:
        raise UnreadableInput(f"{path} could not be read: {failure}") from None
    except (json.JSONDecodeError, yaml.YAMLError) as failure:
        raise UnreadableInput(f"{path} is not valid {suffix}: {failure}") from None
