"""Run configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

DEFAULT_PATH = Path(__file__).resolve().parents[1] / "rlmness.yaml"


class MissingModel(Exception):
    pass


@dataclass(frozen=True)
class Config:
    primary_agent: str
    sub_agent: str | None = None
    # A step is one model call. Delegating costs one turn to set up and one
    # to read the answers back, which only buys anything when there are
    # turns to spare.
    max_steps: int = 20
    # How much of a cell's output comes back. Generous truncation lets the
    # model read a chunk itself rather than hand it to a helper, which is
    # the choice this whole design exists to make attractive.
    truncate_len: int = 2000
    timeout: float = 120.0
    max_depth: int = 3
    # Tree-wide, not per agent. It has to clear the worst honest tree —
    # a root spending every step plus a gather of children doing the same —
    # or it stops being a backstop and becomes a tax on delegating.
    max_calls: int = 400
    max_cost: float = 1.0
    # A gather that cannot get slots runs its children one after another,
    # so a tight ceiling here does not fail a run — it quietly makes
    # delegating the slower choice, which is worse.
    max_concurrent: int = 16
    # Not unbounded, because every live agent here is a real process. A
    # design that keeps its sandboxes inside one process can afford to
    # have no ceiling at all; this one cannot.
    max_live: int = 32
    max_seconds: float | None = None
    max_completion_tokens: int | None = None
    max_prompt_tokens: int | None = None
    runtime: str = "subprocess"
    provider: str = "openrouter"
    api_max_retries: int = 3
    api_backoff: float = 0.5
    # How long one model call may hang before it is abandoned. Separate
    # from `timeout`, which bounds a cell rather than a call. A provider
    # that stops answering used to cost four full attempts at two minutes
    # each, and a gather waits for every child, so one stuck call set the
    # clock for a whole tree.
    api_timeout: float = 60.0
    # Code generation wants a near-deterministic sample, and thinking that
    # happens inside the model is thinking the REPL never sees — a run that
    # reasons its way to an answer has skipped the mechanism entirely.
    temperature: float | None = 0.1
    reasoning_effort: str | None = "low"
    # Ablations. Turn one off to measure what it was worth.
    # Off by default. Counting down the remaining steps is scarcity
    # pressure, and a model with two turns left will not begin work that
    # costs two turns — which is exactly what delegating costs.
    enable_step_banner: bool = False
    enable_delegation: bool = True
    # Refuse a first-step answer that never read PROMPT. Off by default: the
    # run now opens by reading PROMPT, which makes answering blind much less
    # likely, and a refusal costs a turn whenever it is wrong.
    enable_first_look_guard: bool = False
    # A child starts with nothing its parent did not hand it. Turning this on
    # makes a child receive its parent's tools when the call does not say.
    inherit_tools: bool = False

    def model_for(self, depth: int) -> str:
        if depth == 0:
            return self.primary_agent
        return self.sub_agent or self.primary_agent


def load_config(
    path: Path | str | None = None,
    *,
    primary_agent: str | None = None,
    runtime: str | None = None,
    provider: str | None = None,
) -> Config:
    path = Path(path) if path is not None else DEFAULT_PATH
    try:
        raw = yaml.safe_load(path.read_text()) or {}
    except FileNotFoundError:
        raw = {}

    model = primary_agent or raw.get("primary_agent")
    if not model:
        raise MissingModel(
            "primary_agent is required: set it in the config file or pass it explicitly"
        )

    defaults = Config(primary_agent=model)
    return Config(
        primary_agent=model,
        sub_agent=raw.get("sub_agent") or None,
        max_steps=int(raw.get("max_steps", defaults.max_steps)),
        truncate_len=int(raw.get("truncate_len", defaults.truncate_len)),
        timeout=float(raw.get("timeout", defaults.timeout)),
        max_depth=int(raw.get("max_depth", defaults.max_depth)),
        max_calls=int(raw.get("max_calls", defaults.max_calls)),
        max_cost=float(raw.get("max_cost", defaults.max_cost)),
        max_concurrent=int(raw.get("max_concurrent", defaults.max_concurrent)),
        max_live=int(raw.get("max_live", defaults.max_live)),
        max_seconds=(
            float(raw["max_seconds"]) if raw.get("max_seconds") is not None else None
        ),
        max_completion_tokens=(
            int(raw["max_completion_tokens"])
            if raw.get("max_completion_tokens") is not None
            else None
        ),
        max_prompt_tokens=(
            int(raw["max_prompt_tokens"])
            if raw.get("max_prompt_tokens") is not None
            else None
        ),
        # Caller beats file: the caller has already folded in the flag and the
        # environment, both of which are more specific than a checked-in file.
        runtime=runtime or raw.get("runtime") or defaults.runtime,
        provider=provider or raw.get("provider") or defaults.provider,
        api_max_retries=int(raw.get("api_max_retries", defaults.api_max_retries)),
        api_backoff=float(raw.get("api_backoff", defaults.api_backoff)),
        api_timeout=float(raw.get("api_timeout", defaults.api_timeout)),
        temperature=(
            float(raw["temperature"]) if raw.get("temperature") is not None
            else defaults.temperature
        ),
        reasoning_effort=raw.get("reasoning_effort", defaults.reasoning_effort) or None,
        enable_step_banner=bool(
            raw.get("enable_step_banner", defaults.enable_step_banner)
        ),
        enable_delegation=bool(raw.get("enable_delegation", defaults.enable_delegation)),
        enable_first_look_guard=bool(
            raw.get("enable_first_look_guard", defaults.enable_first_look_guard)
        ),
        inherit_tools=bool(raw.get("inherit_tools", defaults.inherit_tools)),
    )
