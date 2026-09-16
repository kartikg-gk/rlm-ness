# rlm-ness

A Recursive Language Model runtime. The input never goes into the model's
context. It lives in a Python REPL as `PROMPT`; the model writes code to look
at it, and hands pieces to sub-agents of itself when a piece needs judgement
rather than string work.

| In the REPL | What it does |
|---|---|
| `PROMPT` | the input — text, or a dict/list when structured input is on |
| `await rlm(x, instruction=None, tools=None, schema=None)` | ask one sub-agent |
| `await gather_rlm([x, ...], instruction=None, tools=None, schema=None)` | ask many at once, in parallel |
| `FINAL(value)` | the answer |
| `commit(name, note=None)` | keep a variable in the session whatever its size |

## Install

```bash
pip install -e .
```

Requirements:

- Python 3.10+
- Node 18+ and `npm install` — only for the `wasm` runtime
- `pip install -e ".[tui]"` — only for the live dashboard and the session TUI

## Environment variables

| Variable | Provider |
|---|---|
| `OPENROUTER_API_KEY` | `openrouter` (default) |
| `DEEPSEEK_API_KEY` | `deepseek` |
| `RLMNESS_PROVIDER` | pick the provider without a flag |
| `RLMNESS_RUNTIME` | pick the runtime without a flag |

## Quick start

```python
from rlmness import Config, make_client, solve

config = Config(primary_agent="z-ai/glm-5")
answer = solve("Generate 50 fruits and count the letter r in each",
               make_client("openrouter"), config=config)

print(answer.output)
print(answer.steps, answer.usage.total_tokens, answer.usage.cost)
```

`primary_agent` has no default. `sub_agent` falls back to it.

### From the command line

```bash
rlmness "Generate 50 fruits and count the letter r in each" --model z-ai/glm-5
```

A file goes in through stdin, as text:

```bash
cat reviews.json | rlmness --instruction "Aggregate the reviews into a verdict"
```

Flags: `--model`, `--instruction`, `--max-steps`, `--max-depth`, `--config`,
`--runtime`, `--provider`, `--dashboard`, and the session flags below. With no
query and a terminal attached, the dashboard opens and takes the question.

## Arbitrarily long context

```python
transcripts = open("all_transcripts.txt").read()

answer = solve(transcripts, client, config=config,
               instruction="Summarise what the first five ML guests said about AGI.")
```

The model searches, filters and chunks `PROMPT` itself. Passing the question
as `instruction` keeps it out of the data; if it has to travel inside, put it
at the very top or bottom.

## Structured input and output

`PROMPT` can be a dict or list, not only a string — the opening step shows the
model its keys, or its type and length, so it can index `PROMPT["reviews"]`
directly rather than parsing text back apart. A sub-agent given a dict or list
receives it as itself, the same way: `await rlm({"items": rows}, instruction="sum them")`
hands the child a real dict, not a string it has to re-parse. Turn it off with
`enable_structured_output: false` and everything — root prompt and every
handoff — becomes JSON text instead.

`FINAL` can be checked against a shape:

```python
answer = solve(prompt, client, config=config,
               output_schema={"type": "object", "properties": {"total": {"type": "integer"}},
                              "required": ["total"]})
```

A JSON Schema dict, a plain type (`str`, `int`, `list`, ...), or a pydantic
model/type (pydantic is only imported if you pass one). A `FINAL` that does
not match is refused with the schema and what did not match; the loop
continues rather than losing the run. The same `schema=` keyword works on
`rlm(...)` and `gather_rlm(...)` for a child's own answer.

## Tools

Pass functions by name; they are bound in the REPL of the root agent.

```python
def filter_short(items: list[str], max_len: int = 20) -> list[str]:
    """Return only items shorter than max_len."""
    return [x for x in items if len(x) < max_len]

solve(prompt, client, config=config, tools={"filter_short": filter_short})
```

- Add a description with `{"filter_short": {"tool": filter_short, "description": "..."}}`.
- A non-callable value is installed as data.
- Sub-agents get nothing unless the parent passes it:
  `await rlm(text, tools=["filter_short"])`. Set `inherit_tools: true` to flip
  that default.
- Under `subprocess` and `wasm` a tool is rebuilt from its source: it must be a
  module-level function with no closures, doing its own imports, returning
  JSON-safe data. Under `in-process` any callable works.
- A saved variable never silently replaces a tool of the same name; it is
  parked beside it as `{name}_saved`.

## Sandboxes

| `runtime` | Isolation | Notes |
|---|---|---|
| `subprocess` (default) | separate Python process | quickest to start |
| `wasm` | Pyodide in one shared Node host | no network: `js`, `pyodide.http` are shut |
| `in-process` | none | tools may be live objects; trusted code only |

The `wasm` runtime closes the documented ways out. Pyodide shares a JavaScript
context with its host and is not a security boundary.

Every fan-out — `asyncio.gather`, `wait`, `as_completed`, `create_task`,
`ensure_future`, a `TaskGroup` — is refused if any piece of it is a sub-agent
call, on `subprocess` and `wasm`: the message names `gather_rlm` and shows an
example. `gather_rlm` itself keeps a fan-out inside `max_concurrent`, claims a
slot per child, and stops the rest as soon as one fails, which a hand-built
gather does none of. Turn it off with `enable_batching_guard: false`.
`in-process` shares its event loop with the caller and cannot enforce this.

## Resumable sessions

A session keeps the root agent's namespace and its earlier answers between
questions. After every step, picklable variables are saved, functions and
classes are saved as source, and imported modules by name. The next run
restores them — including what was committed, noted or commented, so a
resumed session remembers a name as fully as the run that built it did — and
is shown the earlier questions, answers and a bounded view of the code that
built them.

```bash
rlmness --session ./notes "Load the logs and build an index by service"
rlmness --session ./notes "Using the index: which service failed most?"
```

```python
from rlmness.session import Session

book = Session.load("notes", "podcasts")      # notes/podcasts/state.json
solve(transcripts, client, config=config, instruction="Build a guest index", session=book)
solve("", client, config=config, instruction="Who was most optimistic?", session=book)

book.variables   # what is kept
book.answered    # earlier questions and answers, each linked to its trace
```

Where state lives:

| | |
|---|---|
| `--session file.json` | that file |
| `--session dir` | `dir/state.json` |
| `--session dir --session-id x` | `dir/x/state.json` |
| `--session-ephemeral` / `Session()` | memory only, gone when the process exits |

Behaviour:

- Written after every successful step, atomically. A killed run resumes from its last step.
- Two runs on one file do not overwrite each other: the one that finds the
  file changed under it saves to `state.1.json` instead and says so — and the
  run that loaded the original names every such file sitting beside it, so
  that work is findable rather than only announced once.
- Not a process clone. Open handles and generators are dropped and reported; large values are skipped unless `commit`ted.
- The conversation is not carried, only the ledger, the code and the restored names, so a resumed prompt stays bounded.
- `--session-no-code` hides earlier code.
- A saved name never replaces a tool or helper of the same name; it is restored beside it as `name_saved`.
- Sub-agents stay fresh. Only the root agent's state persists.

## Instructions

`instruction` is shown to one agent only. A sub-agent gets one when its parent
passes it:

```python
# inside the REPL
amounts = await gather_rlm(chunks, instruction="Return only dollar amounts, as a JSON list.")
```

A child's own sub-agents start without it.

## Configuration

`rlmness.yaml` in the working directory, `--config path`, or `Config(...)`.

| Field | Default | Meaning |
|---|---|---|
| `primary_agent` | required | root model |
| `sub_agent` | `primary_agent` | model for children |
| `max_depth` | 3 | how deep sub-agents nest |
| `max_steps` | 20 | turns per agent |
| `truncate_len` | 10000 | characters of cell output shown per step, kept from the end |
| `max_cost` | 1.0 | dollars for the whole run |
| `max_calls` | 2000 | calls across the run; backstop where no price is reported |
| `max_completion_tokens` | 500000 | completion tokens across the run |
| `max_prompt_tokens` | 200000 | one call's input plus output |
| `max_seconds` | 1800 | wall clock for the run |
| `max_concurrent` | 16 | pieces of one `gather_rlm` in flight |
| `max_live` | 32 | agents alive at once across the tree |
| `timeout` | 120 | seconds for one cell |
| `max_tokens` | unset | a ceiling on one reply, sent with the request when set — some providers hold credit for the largest reply a request could produce, so this matters on a small key |
| `api_timeout` | 60 | seconds for one read of a reply |
| `api_deadline` | 600 | seconds for a whole reply, so one that trickles cannot outlast `api_timeout` forever |
| `api_retry_after_max` | 60 | how long a provider's own `Retry-After` is obeyed, capped |
| `api_max_retries` / `api_backoff` | 3 / 0.5 | retry policy for a failed call |
| `temperature` / `reasoning_effort` | 0.1 / low | sampling |
| `runtime` / `provider` | subprocess / openrouter | |

Behaviour switches:

| Field | Default | Effect |
|---|---|---|
| `enable_delegation` | true | bind `rlm` and `gather_rlm` |
| `enable_step_banner` | true | show steps remaining once past halfway |
| `enable_batching_guard` | true | refuse a hand-built fan-out over sub-agent calls and point to `gather_rlm` |
| `enable_structured_output` | true | dict/list `PROMPT` and handoffs stay structured; `output_schema` is checked |
| `enable_blind_final_guard` | false | hold a literal answer written in a cell that reads `PROMPT` |
| `enable_first_look_guard` | false | refuse a first-step answer that never read `PROMPT` |
| `inherit_tools` | false | children receive their parent's tools by default |

## Progress

Pass any object as `trace`. It receives `run_started`, `step`, `final`,
`run_completed` and `run_failed` for the root and every sub-agent as they
happen, each carrying `run_id`, `parent_run_id` and `depth`.

```python
from rlmness import Journal
from rlmness.events import Broadcast

class Printer:
    def step(self, *, step, depth=0, usage=None, **_):
        print(f"depth {depth} step {step}")
    def final(self, result, **_):
        pass

solve(prompt, client, config=config, trace=Broadcast(Journal(), Printer()))
```

## Logs

Every CLI run writes a JSONL trace under `traces/` and prints its path.
`Journal(path)` picks the file from Python.

```bash
rlmness-viewlog traces/run_xxx.jsonl        # one run, as a tree
rlmness-viewlog traces/run_xxx.jsonl --tui  # the same, interactively

rlmness-viewlog --session ./notes           # every question a session has answered,
rlmness-viewlog --session ./notes --tui     # what each cost, and its own trace

rlmness-timeline traces/run_xxx.jsonl       # who ran when
rlmness --dashboard "..."                   # live tree while it runs
```

## Measuring a change

```bash
python -m evals.ablation --provider openrouter --model deepseek/deepseek-v4-flash --setting enable_step_banner
python -m evals.ablation --provider openrouter --model deepseek/deepseek-v4-flash --tasks longbench:hotpotqa -n 5
```

Runs each task with the setting on and off and prints the spread. `sanity`
shows nothing broke; only `longbench` supports a claim that something improved.

## License

MIT
