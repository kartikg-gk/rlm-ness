"""Command line entry point."""

from __future__ import annotations

import argparse
import dataclasses
import os
import sys

from .providers import PROVIDERS, MissingApiKey, make_client
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
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="watch the run in a terminal dashboard (needs the tui extra)",
    )
    return parser.parse_args(argv)


def main(argv=None, *, backend=None) -> int:
    args = _parse(argv if argv is not None else sys.argv[1:])

    # No query and a terminal to draw on: open the dashboard and let the
    # question be typed there. Piped input still means "read it and answer",
    # so `cat file | rlmness` keeps working headless.
    interactive = args.query is None and sys.stdin.isatty()
    query = args.query if args.query is not None else ("" if interactive else sys.stdin.read())
    wants_dashboard = args.dashboard or interactive

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
    except MissingApiKey as error:
        # Nothing can run without a key, so this is fatal either way. Say which
        # provider is being asked for and how to point at a different one,
        # since the usual cause is the configured default rather than a
        # forgotten export.
        print(f"{error}, or choose another provider.", file=sys.stderr)
        print(
            f"  configured provider: {config.provider}\n"
            f"  others: {', '.join(sorted(set(PROVIDERS) - {config.provider}))}\n"
            f"  e.g. rlmness --provider deepseek --model deepseek-v4-flash",
            file=sys.stderr,
        )
        return 1
    except Exception as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1

    trace = Journal()
    sink = trace
    live = None
    if wants_dashboard:
        try:
            from .dashboard import show
            from .events import Broadcast, RunTree
        except ImportError:
            print(
                "the dashboard needs textual: pip install 'rlm-ness[tui]'",
                file=sys.stderr,
            )
            return 1
        live = RunTree()
        live.query = query
        # The journal is still written. The dashboard is an extra reader of
        # the same events, never a replacement for the record.
        sink = Broadcast(trace, live)

    def run(text: str) -> Answer:
        return solve(
            text,
            backend,
            instruction=args.instruction,
            config=config,
            trace=sink,
        )

    try:
        if interactive:
            # A fresh journal per question: one file holding two unrelated
            # runs would make the trace unreadable and the timeline wrong.
            def ask(text: str) -> Answer:
                nonlocal trace, sink
                trace = Journal()
                sink = Broadcast(trace, live)
                return run(text)

            show(live, ask=ask, title=config.primary_agent)
            return 0
        if wants_dashboard:
            app = show(live, runner=lambda: run(query), title=config.primary_agent)
            if app.failure is not None:
                raise app.failure
            if app.result is None:
                print(f"trace: {trace.path}")
                return 0
            result = app.result
        else:
            result = run(query)
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
