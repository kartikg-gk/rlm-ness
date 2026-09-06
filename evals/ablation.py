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
import sys
import time

from rlmness.providers import make_client
from rlmness.limits import Allowance
from rlmness.config import Config
from rlmness.engine import solve
from rlmness.runtime import SubprocessRuntime

from .tasks import BENCHMARK, Task, resolve, with_question_inside


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
            # Passed through whole rather than unpacked: a backend may report
            # the model's reasoning as a third item, and a wrapper that only
            # knows about two would quietly drop it on its way to the trace.
            answer = backend.complete(messages, model=model)
            text = answer[0]
            if SPAWNS.search(text):
                flags["spawned"] = True
            if HELPERS.search(text):
                flags["helped"] = True
            return answer

    return Watched()


def _once(task: Task, config: Config, provider: str) -> Outcome:
    # A task that says how much output it needs gets it. The default suits
    # most, and a task whose useful output runs longer would otherwise be
    # shown only its tail -- which reads to the agent as though the work did
    # not happen, and it runs it again.
    if task.truncate_len is not None:
        config = dataclasses.replace(config, truncate_len=task.truncate_len)
    allowance = Allowance.from_config(config)
    flags = {"spawned": False, "helped": False}
    backend = _watching(
        make_client(
            provider,
            temperature=config.temperature,
            reasoning_effort=config.reasoning_effort,
            timeout=config.api_timeout,
        ),
        flags,
    )
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
        score, failure = 0.0, _failure_label(error)

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


def _failure_label(error: Exception) -> str:
    """Say enough about a failure to tell it apart from a result.

    A bare class name reads the same whether the run answered badly or never
    reached the model at all. A spent key comes back as a refusal with a
    status on it, and reported as just its type it sits in the table next to
    real runs and gets counted as one.
    """
    message = str(error).strip().splitlines()
    detail = message[0][:120] if message else ""
    return f"{type(error).__name__}: {detail}" if detail else type(error).__name__


def _row(label: str, outcomes: list[Outcome]) -> str:
    """Average over the runs that happened, and count the ones that did not.

    A run that failed has no steps and no delegation to report. Averaged in,
    its zeros are indistinguishable from a run that went all the way and
    chose not to delegate — which is the difference the whole table exists
    to show.
    """
    done = [outcome for outcome in outcomes if not outcome.failure]
    if not done:
        return f"  {label:<4} nothing completed ({len(outcomes)} failed)"
    steps = [outcome.steps for outcome in done]
    spread = statistics.stdev(steps) if len(steps) > 1 else 0.0
    row = (
        f"  {label:<4} solved {sum(o.solved for o in done)}/{len(done)}"
        f"  score {statistics.mean(o.score for o in done):.2f}"
        f"  delegated {sum(o.delegated for o in done)}/{len(done)}"
        f"  helped {sum(o.helped for o in done)}/{len(done)}"
        f"  steps {statistics.mean(steps):.1f}±{spread:.1f}"
        f"  calls {statistics.mean(o.calls for o in done):.1f}"
        f"  {statistics.mean(o.seconds for o in done):.1f}s"
    )
    if len(done) < len(outcomes):
        row += f"  ({len(outcomes) - len(done)} failed)"
    return row


def main() -> int:
    parser = argparse.ArgumentParser(prog="ablation")
    parser.add_argument("--model", required=True)
    parser.add_argument("--provider", default="deepseek")
    parser.add_argument("--setting", default="enable_step_banner")
    parser.add_argument("--tasks", default="sanity", help="sanity | longbench[:config]")
    parser.add_argument("-n", "--num-samples", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--truncate-len", type=int, default=2000)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument(
        "--question-inside",
        choices=("top", "bottom"),
        help=(
            "fold each task's question into PROMPT at one end, the way a "
            "caller with a single string does, instead of passing it "
            "alongside"
        ),
    )
    arguments = parser.parse_args()

    _defaults = Config(primary_agent=arguments.model)
    base = Config(
        primary_agent=arguments.model,
        max_steps=arguments.max_steps,
        truncate_len=arguments.truncate_len,
        timeout=300.0,
        max_depth=arguments.max_depth,
        # Taken from the shared defaults rather than restated here: a
        # ceiling that binds before a delegating tree finishes turns this
        # harness into a measurement of its own budget.
        max_calls=_defaults.max_calls,
        max_cost=1e9,
        max_concurrent=_defaults.max_concurrent,
        max_live=_defaults.max_live,
        max_seconds=_defaults.max_seconds or 1800,
    )
    if not hasattr(base, arguments.setting):
        raise SystemExit(f"no such setting: {arguments.setting}")

    # A run takes minutes per task and prints as it goes. Redirected to a
    # file, the block buffer holds all of that until the process ends, so a
    # run that is working looks exactly like one that has hung.
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is not None:
        reconfigure(line_buffering=True)

    tasks = resolve(arguments.tasks, arguments.num_samples)
    if arguments.question_inside:
        tasks = [with_question_inside(t, arguments.question_inside) for t in tasks]
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
