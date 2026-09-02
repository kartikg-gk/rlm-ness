"""Measure what a setting is worth, by turning it off.

A prompt change cannot be unit tested: the model is stochastic, so the result
is a rate rather than an assertion. This runs the same tasks with a setting on
and off, in the same session, and reports the difference with its spread.

    python -m evals.ablation --model deepseek-v4-flash --setting enable_step_banner
    python -m evals.ablation --model deepseek-v4-flash --tasks longbench:hotpotqa -n 5

Read the tier before believing a result. A difference on sanity tasks means the
thing still runs; only benchmark tasks support a claim that it runs better.

Costs real API calls. Nothing here runs during the test suite.
"""

from __future__ import annotations

import argparse
import dataclasses
import re
import statistics
import time

from rlmness.providers import make_client
from rlmness.limits import Allowance
from rlmness.config import Config
from rlmness.engine import solve
from rlmness.runtime import SubprocessRuntime

from .tasks import BENCHMARK, Task, resolve


@dataclasses.dataclass
class Outcome:
    score: float
    solved: bool
    steps: int
    calls: int
    seconds: float
    delegated: bool
    helped: bool = False
    extra_calls: int = 0
    failure: str | None = None


class StepCounter:
    """Counts the root agent's steps, including on a run that raises.

    A failed run has still done work, and how much is the interesting part.
    Inferring it from the step ceiling reports every failure as having used the
    whole allowance, which is wrong whenever it died early.
    """

    def __init__(self):
        self.steps = 0

    def step(self, *, step, code, output, error, usage, depth=0, **_ignored):
        if depth == 0:
            self.steps = max(self.steps, step)

    def final(self, result, *, depth=0, **_ignored):
        pass


# Each helper matched on its own, with a boundary in front, so `gather_llm(`
# is not also counted as `llm(`. The earlier version looked for the literal
# "await rlm(" and for the two gathers, which missed a plain `llm(` entirely
# and missed any spawn the model spelled with different spacing.
SPAWNS = re.compile(r"(?<![\w.])(?:rlm|gather_rlm)\s*\(")
HELPERS = re.compile(r"(?<![\w.])(?:rlm|llm|gather_rlm|gather_llm)\s*\(")


def _watching(backend, flags):
    """Note what a reply reached for, without changing behaviour."""

    class Watched:
        def complete(self, messages, *, model):
            text, usage = backend.complete(messages, model=model)
            if SPAWNS.search(text):
                flags["spawned"] = True
            if HELPERS.search(text):
                flags["helped"] = True
            return text, usage

    return Watched()


def _once(task: Task, config: Config, provider: str) -> Outcome:
    allowance = Allowance.from_config(config)
    flags = {"spawned": False, "helped": False}
    backend = _watching(make_client(provider), flags)
    counter = StepCounter()

    start = time.perf_counter()
    try:
        result = solve(
            task.prompt,
            backend,
            instruction=task.instruction,
            config=config,
            runtime_factory=SubprocessRuntime,
            allowance=allowance,
            trace=counter,
        )
        score, failure = task.score(result.output), None
    except Exception as error:
        score, failure = 0.0, type(error).__name__

    # A root agent spends one call per step. Anything above that was spent by
    # a helper, whatever the reply happened to look like — arithmetic the
    # pattern above cannot disagree with.
    return Outcome(
        score=score,
        solved=score >= task.threshold,
        steps=counter.steps,
        calls=allowance.calls,
        seconds=time.perf_counter() - start,
        delegated=flags["spawned"],
        helped=flags["helped"] or allowance.calls > counter.steps,
        extra_calls=max(0, allowance.calls - counter.steps),
        failure=failure,
    )


def _row(label: str, outcomes: list[Outcome]) -> str:
    steps = [outcome.steps for outcome in outcomes]
    spread = statistics.stdev(steps) if len(steps) > 1 else 0.0
    return (
        f"  {label:<4} solved {sum(o.solved for o in outcomes)}/{len(outcomes)}"
        f"  score {statistics.mean(o.score for o in outcomes):.2f}"
        f"  delegated {sum(o.delegated for o in outcomes)}/{len(outcomes)}"
        f"  helped {sum(o.helped for o in outcomes)}/{len(outcomes)}"
        f"  steps {statistics.mean(steps):.1f}±{spread:.1f}"
        f"  calls {statistics.mean(o.calls for o in outcomes):.1f}"
        f"  {statistics.mean(o.seconds for o in outcomes):.1f}s"
    )


def main() -> int:
    parser = argparse.ArgumentParser(prog="ablation")
    parser.add_argument("--model", required=True)
    parser.add_argument("--provider", default="deepseek")
    parser.add_argument("--setting", default="enable_step_banner")
    parser.add_argument("--tasks", default="sanity", help="sanity | longbench[:config]")
    parser.add_argument("-n", "--num-samples", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=6)
    arguments = parser.parse_args()

    base = Config(
        primary_agent=arguments.model,
        max_steps=arguments.max_steps,
        truncate_len=6000,
        timeout=300.0,
        max_depth=2,
        max_calls=80,
        max_cost=1e9,
        max_concurrent=8,
        max_live=16,
        max_seconds=1800,
    )
    if not hasattr(base, arguments.setting):
        raise SystemExit(f"no such setting: {arguments.setting}")

    tasks = resolve(arguments.tasks, arguments.num_samples)
    print(f"ablating {arguments.setting} on {arguments.provider}/{arguments.model}")
    print(f"{len(tasks)} task(s) x {arguments.repeats} repeats x 2 arms\n")

    for task in tasks:
        print(f"{task.name}  [{task.tier}]")
        for label, value in (("on", True), ("off", False)):
            config = dataclasses.replace(base, **{arguments.setting: value})
            outcomes = [
                _once(task, config, arguments.provider) for _ in range(arguments.repeats)
            ]
            print(_row(label, outcomes))
            failures = sorted({o.failure for o in outcomes if o.failure})
            if failures:
                print(f"       failures: {', '.join(failures)}")
        print()

    if not any(task.tier == BENCHMARK for task in tasks):
        print("Sanity tier only: this shows nothing broke, not that anything improved.")
    print("A difference smaller than the step spread is not a difference.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
