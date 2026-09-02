"""Six sub-agents at once, then one answer assembled from all six.

The point is the shape of the work rather than the answer: each list is
independent, so a loop would pay for six round trips in sequence where one
gather call pays for the slowest. Watch it with --dashboard to see the six
children appear side by side.

    python examples/fan_out.py
"""

from rlmness.config import load_config
from rlmness.engine import solve
from rlmness.journal import Journal
from rlmness.providers import make_client

TASK = """
Build six lists of 25 names each: fruits, animals, US states, Indian states,
European countries, and common first names.

Ask for all six at once with a single gather_rlm call — one sub-agent per list,
each told to return a Python list of strings. Do not call them one at a time.

Then combine the six lists and return a dict mapping every name to the number
of times the letter 'r' appears in it, case-insensitive.
"""


def main() -> None:
    config = load_config()
    backend = make_client(config.provider)
    trace = Journal()

    answer = solve(TASK, backend, config=config, trace=trace)

    result = answer.output
    print(f"names:  {len(result) if hasattr(result, '__len__') else '?'}")
    print(f"steps:  {answer.steps}")
    print(f"tokens: {answer.usage.total_tokens}")
    print(f"trace:  {trace.path}")
    print("view the timeline with: rlmness-timeline " + str(trace.path))


if __name__ == "__main__":
    main()
