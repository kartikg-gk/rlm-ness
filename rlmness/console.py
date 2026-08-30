"""Command line entry point."""

from __future__ import annotations

import argparse
import dataclasses
import os
import sys

from .providers import PROVIDERS, make_client
from .config import load_config
from .engine import RUNTIMES, Answer, solve
from .journal import Journal


def _parse(argv):
    parser = argparse.ArgumentParser(prog="rlmness")
    parser.add_argument("query", nargs="?")
    parser.add_argument("--model")
    parser.add_argument("--instruction")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--max-depth", type=int)
    parser.add_argument("--config")
    parser.add_argument(
        "--runtime",
        choices=sorted(RUNTIMES),
        # Flag beats environment beats config file. Unset here means the file
        # decides, so the default has to stay None rather than "local".
        default=os.environ.get("RLMNESS_RUNTIME"),
        help="subprocess is quick to start; wasm removes syscalls and needs npm install",
    )
    parser.add_argument(
        "--provider",
        choices=sorted(PROVIDERS),
        default=os.environ.get("RLMNESS_PROVIDER"),
        help="which API to call; each reads its own key from the environment",
    )
    return parser.parse_args(argv)


def main(argv=None, *, backend=None) -> int:
    args = _parse(argv if argv is not None else sys.argv[1:])
    query = args.query if args.query is not None else sys.stdin.read()

    try:
        config = load_config(
            args.config,
            primary_agent=args.model,
            runtime=args.runtime,
            provider=args.provider,
        )
        overrides = {}
        if args.max_steps:
            overrides["max_steps"] = args.max_steps
        if args.max_depth is not None:
            overrides["max_depth"] = args.max_depth
        if overrides:
            config = dataclasses.replace(config, **overrides)
        backend = backend or make_client(
            config.provider,
            max_retries=config.api_max_retries,
            backoff=config.api_backoff,
        )
    except Exception as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1

    trace = Journal()
    try:
        result = solve(
            query,
            backend,
            instruction=args.instruction,
            config=config,
            trace=trace,
        )
    except Exception as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        print(f"trace: {trace.path}", file=sys.stderr)
        return 1

    cost = "unknown (this provider reports no price)" if result.usage.cost is None else f"${result.usage.cost:.4f}"
    print(result.output)
    print(f"steps: {result.steps}  tokens: {result.usage.total_tokens}  cost: {cost}")
    print(f"trace: {trace.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
