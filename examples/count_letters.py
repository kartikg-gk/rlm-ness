"""The smallest thing that exercises the whole loop.

No API keys are spent on thinking: the answer is a count, so a wrong one is
obvious. Run this first after changing anything in the loop.

    python examples/count_letters.py
"""

from rlmness.config import load_config
from rlmness.engine import solve
from rlmness.journal import Journal
from rlmness.providers import make_client

TEXT = ("strawberry " * 300).strip()


def main() -> None:
    config = load_config()
    backend = make_client(config.provider)
    trace = Journal()

    answer = solve(
        TEXT,
        backend,
        instruction="How many times does the letter r appear in PROMPT? Return an integer.",
        config=config,
        trace=trace,
    )

    print(f"answer:   {answer.output}")
    print(f"expected: {TEXT.count('r')}")
    print(f"steps:    {answer.steps}")
    print(f"trace:    {trace.path}")


if __name__ == "__main__":
    main()
