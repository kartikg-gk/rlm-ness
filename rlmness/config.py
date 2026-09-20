"""Run configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from . import home as _home

CONFIG_NAME = "rlmness.yaml"
CHECKOUT_PATH = _home.CHECKOUT / CONFIG_NAME


def default_path() -> Path:
    """The working folder's config, else the home folder's, else a checkout's."""
    for candidate in (Path.cwd() / CONFIG_NAME, _home.home() / CONFIG_NAME, CHECKOUT_PATH):
        if candidate.is_file():
            return candidate
    return Path.cwd() / CONFIG_NAME


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
    # How much of a cell's output comes back, kept from the end. Too little
    # and a child printing its whole piece never sees the middle of it, and
    # reports that what it was asked for is not there.
    truncate_len: int = 10000
    timeout: float = 120.0
    max_depth: int = 3
    # Tree-wide, not per agent, and a backstop rather than the guard that is
    # meant to bite. `max_cost` is the one that should stop an ordinary run,
    # because money is what is actually being spent; this exists for the
    # providers that report no price at all, where cost can never fire.
    #
    # So it has to sit above the cost ceiling, not below it. At the prices
    # measured on real runs — roughly $0.0015 to $0.002 a call — a $1.00
    # ceiling is somewhere near 500 to 650 calls, and the old 400 fired first.
    # That put the backstop in front of the guard: runs stopped on a call
    # count while the budget they were given was still unspent.
    #
    # It also has to clear an honest tree deeper than one gather. A root
    # spending every step plus one level of children is 20 + 16 x 20 = 340;
    # a second level is far more, and delegating is the thing this project
    # exists to do.
    max_calls: int = 2000
    max_cost: float = 1.0
    # A gather that cannot get slots runs its children one after another,
    # so a tight ceiling here does not fail a run — it quietly makes
    # delegating the slower choice, which is worse.
    max_concurrent: int = 16
    # Not unbounded, because every live agent is a real process and a wide
    # tree of them is real memory. The runtime says what one of its own
    # costs, and this is clamped to that.
    max_live: int = 32
    max_seconds: float | None = None
    max_completion_tokens: int | None = None
    max_prompt_tokens: int | None = None
    # Sealed by default: the model's code has no network and no host files.
    # Needs Node and `npm install`; the command falls back to subprocess, and
    # says so, when those are missing and nobody asked for wasm by name.
    runtime: str = "wasm"
    provider: str = "openrouter"
    api_max_retries: int = 3
    api_backoff: float = 0.5
    # How long one model call may hang before it is abandoned. Separate
    # from `timeout`, which bounds a cell rather than a call. A provider
    # that stops answering used to cost four full attempts at two minutes
    # each, and a gather waits for every child, so one stuck call set the
    # clock for a whole tree.
    api_timeout: float = 60.0
    # A ceiling on one reply, sent with the request when set. Off by default:
    # a reply cut off mid-cell costs a step. Worth setting where a provider
    # holds credit for the largest reply a request could produce, since an
    # unstated ceiling is then the model's own maximum.
    max_tokens: int | None = None
    # A provider that refuses and says when to come back is obeyed, up to
    # this much of a wait.
    api_retry_after_max: float = 60.0
    # A bound on a whole reply, where `api_timeout` bounds one read of it.
    # Without it a reply that trickles never times out at all.
    api_deadline: float | None = 600.0
    # Code generation wants a near-deterministic sample, and thinking that
    # happens inside the model is thinking the REPL never sees — a run that
    # reasons its way to an answer has skipped the mechanism entirely.
    temperature: float | None = 0.1
    reasoning_effort: str | None = "low"
    # Ablations. Turn one off to measure what it was worth.
    # On past halfway, silent before it: a fresh agent does not need reminding
    # that it has room, and an agent near the end does. The banner names the
    # count and points at delegation, because the answer to a short budget is
    # to hand pieces out rather than look again.
    enable_step_banner: bool = True
    enable_delegation: bool = True
    # Refuse a first-step answer that never read PROMPT. Off by default: the
    # run now opens by reading PROMPT, which makes answering blind much less
    # likely, and a refusal costs a turn whenever it is wrong.
    enable_first_look_guard: bool = False
    # Set aside a FINAL of fixed text written in the same block that reads
    # PROMPT: it was decided before that block's output existed. Off by
    # default, so a FINAL is accepted in any block, as the loop has always
    # done. Every recorded case of it firing was an answer that was wrong,
    # recalled, or about to be; turn it on to trade one step for that.
    enable_blind_final_guard: bool = False
    # A child starts with nothing its parent did not hand it. Turning this on
    # makes a child receive its parent's tools when the call does not say.
    inherit_tools: bool = False
    # Refuse a sub-agent call handed to asyncio.gather and name the gather
    # helper instead. The helper keeps a fan-out inside the concurrency limit
    # and stops the rest when one piece fails; a hand-built gather does
    # neither. Enforced only where the runtime owns its interpreter.
    enable_batching_guard: bool = True
    # Ask an agent to confirm before it hands a sub-agent a large, barely
    # reduced share of its own PROMPT. Delegation pays off on pieces already
    # narrowed; passing the whole thing down buys nothing and is billed twice.
    enable_compression_guard: bool = True
    # Below this, an agent's PROMPT is small enough that handing it over whole
    # costs little, and a question about it would cost more than it saves.
    compression_min_chars: int = 5000
    # The share of the parent's own PROMPT that counts as barely reduced.
    compression_ratio: float = 0.6

    # Check FINAL against an output schema when the caller gives one, for the
    # run and for any sub-agent asked for a shape. Off ignores every schema.
    enable_structured_output: bool = True

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
    path = Path(path) if path is not None else default_path()
    try:
        raw = yaml.safe_load(path.read_text()) or {}
    except FileNotFoundError:
        raw = {}

    model = primary_agent or raw.get("primary_agent")
    if not model:
        raise MissingModel(
            "no model set: pass --model, or run `rlmness --setup` for a starter "
            "config, or add primary_agent to rlmness.yaml"
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
        max_tokens=(
            int(raw["max_tokens"]) if raw.get("max_tokens") is not None else None
        ),
        api_retry_after_max=float(
            raw.get("api_retry_after_max", defaults.api_retry_after_max)
        ),
        api_deadline=(
            float(raw["api_deadline"]) if raw.get("api_deadline") is not None
            else defaults.api_deadline
        ),
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
        enable_blind_final_guard=bool(
            raw.get("enable_blind_final_guard", defaults.enable_blind_final_guard)
        ),
        inherit_tools=bool(raw.get("inherit_tools", defaults.inherit_tools)),
        enable_batching_guard=bool(
            raw.get("enable_batching_guard", defaults.enable_batching_guard)
        ),
        enable_structured_output=bool(
            raw.get("enable_structured_output", defaults.enable_structured_output)
        ),
        enable_compression_guard=bool(
            raw.get("enable_compression_guard", defaults.enable_compression_guard)
        ),
        compression_min_chars=int(
            raw.get("compression_min_chars", defaults.compression_min_chars)
        ),
        compression_ratio=float(
            raw.get("compression_ratio", defaults.compression_ratio)
        ),
    )
