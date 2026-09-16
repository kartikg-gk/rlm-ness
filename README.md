# rlm-ness

A Python runtime for **Recursive Language Models** — the inference technique
from [Recursive Language Models](https://arxiv.org/abs/2512.24601) (Zhang,
Kraska & Khattab, MIT CSAIL; [author's write-up](https://alexzhang13.github.io/blog/2025/rlm/)).
The paper's idea is that a prompt does not have to be something a model reads.
It can be something a model *operates on*.

So here the input never enters anyone's context window. It is a variable
called `PROMPT` in a live Python REPL. The model is shown what type it is and
roughly how big, and then writes code — search it, slice it, count it, parse
it. When a piece needs judgement rather than string work, the model spawns a
sub-agent of itself on just that piece.

What the sub-agent returns is a **value, not a transcript**. It lands in the
parent's REPL as an ordinary Python object bound to a name the parent's own
code chose. Fan ten pieces out and the parent holds one list of ten answers,
having paid for none of the ten conversations that produced them. That is the
whole trick: recursion stays cheap because context does not accumulate.

```python
# what the model writes, not what you write
incidents = re.findall(r"^INCIDENT.*?^---", PROMPT, re.M | re.S)
causes = await gather_rlm(incidents, instruction="Name the root cause in under 5 words.")
FINAL(collections.Counter(causes).most_common(3))
```

Three of those incidents could each be a megabyte. The root agent never sees
one.

## Install

```bash
pip install -e .
```

Python 3.10+. Two optional extras, neither needed for an ordinary run:

| For | Do |
|---|---|
| the `wasm` runtime | Node 18+, then `npm install` |
| the live dashboard and session browser | `pip install -e ".[tui]"` |

Set a key for whichever provider you want:

```bash
export OPENROUTER_API_KEY=...     # default provider
export DEEPSEEK_API_KEY=...       # --provider deepseek
```

`RLMNESS_PROVIDER` and `RLMNESS_RUNTIME` set those without a flag. Precedence
is flag, then environment, then `rlmness.yaml`.

## Quick start

```bash
rlmness "Generate 50 fruits and count the letter r in each" --model z-ai/glm-5
```

Pipe the data in and keep the question separate from it:

```bash
cat incidents.log | rlmness --instruction "Which service failed most, and when?"
```

From Python:

```python
from rlmness import Config, make_client, solve

answer = solve(
    open("incidents.log").read(),
    make_client("openrouter"),
    config=Config(primary_agent="z-ai/glm-5", sub_agent="minimax/minimax-m2.5"),
    instruction="Which service failed most, and when?",
)

print(answer.output)                                   # whatever FINAL was given
print(answer.steps, answer.usage.total_tokens, answer.usage.cost)
```

`primary_agent` has no default and never guesses one. `sub_agent` falls back
to it, but setting it to something cheaper is what makes delegating worth
doing — a child that costs a fraction of its parent changes the arithmetic of
handing work out.

Run `rlmness` with no query in a terminal and the dashboard opens and takes
the question there.

## What the model is given

| Name | What it is |
|---|---|
| `PROMPT` | the input — text, or a dict/list left as itself |
| `await rlm(x, instruction=, tools=, schema=)` | one sub-agent on one piece |
| `await gather_rlm([x, ...], instruction=, tools=, schema=)` | many at once, bounded and cancelled together |
| `FINAL(value)` | the answer, any Python value |
| `commit(name, note=)` | keep this variable in the session however large it is |

Nothing else is injected. The first cell is executed for the model rather than
described to it, so step one is a real look at real data.

### Structure survives the handoff

`PROMPT` can be a dict or a list. The opening step lists its keys with a look
at each value, so the model indexes `PROMPT["reviews"]` instead of regexing a
JSON blob back apart. Handing structure *down* works the same way:

```python
await gather_rlm(
    [{"site": name, "rows": rows} for name, rows in by_site.items()],
    instruction="Total the crates shipped.",
)
```

Each child gets a real dict as its own `PROMPT`. Set
`enable_structured_output: false` and every one of these becomes JSON text
instead, which is the honest comparison when you want to measure whether the
structure was worth anything.

### Answers can be required to fit a shape

```python
answer = solve(prompt, client, config=config, output_schema={
    "type": "object",
    "properties": {"total": {"type": "integer"}, "per_site": {"type": "object"}},
    "required": ["total", "per_site"],
})
```

A JSON Schema dict, a bare type (`int`, `list`, ...), or a pydantic model —
pydantic is only imported if you hand it one. A `FINAL` that does not fit is
refused with the schema and every mismatch, and **the run continues**: the
agent still has every variable it built, so it corrects the value rather than
redoing the work. Children take the same `schema=`, so a fan-out can come back
uniformly typed.

## Tools

Functions you pass by name are bound into the root agent's REPL:

```python
def days_since(date: str) -> int:
    """Days between an ISO date and today."""
    import datetime
    return (datetime.date.today() - datetime.date.fromisoformat(date)).days

solve(prompt, client, config=config, tools={"days_since": days_since})
```

- Describe one with `{"days_since": {"tool": days_since, "description": "..."}}`.
- Children inherit nothing by default. A parent grants explicitly with
  `await rlm(piece, tools=["days_since"])`, or `inherit_tools: true` flips it.
- On `subprocess` and `wasm` a tool is rebuilt from source inside the sandbox:
  module-level, no closures, own imports, JSON-safe return. On `in-process`
  any callable works, including bound methods and live objects.

## Getting values in without spending context

Anything in `tools` that is not callable is bound as data under its own name.
This is how settings, credentials-by-proxy, lookup tables and anything else
the model needs reach the REPL without being pasted into a prompt:

```python
solve(prompt, client, config=config, tools={
    "region": "eu-west-1",
    "sla_hours": {"gold": 4, "silver": 24},
})
```

`region` and `sla_hours` are ordinary variables in the model's first cell. No
tokens spent describing them, no round trip to ask for them, and the same
`tools=[...]` grant decides whether a child sees them. A value handed in this
way also never gets silently overwritten by a restored session variable of the
same name — the saved one is parked beside it as `{name}_saved`.

## Configuration

`rlmness.yaml` beside you, `--config path`, or `Config(...)` in Python.

**Models and plumbing**

| Field | Default | Meaning |
|---|---|---|
| `primary_agent` | required | the root model |
| `sub_agent` | `primary_agent` | the model children run on |
| `runtime` | `subprocess` | `subprocess`, `wasm` or `in-process` |
| `provider` | `openrouter` | `openrouter` or `deepseek` |
| `temperature` / `reasoning_effort` | 0.1 / `low` | near-deterministic; reasoning the REPL never sees is reasoning that skipped the mechanism |

**What a run may spend**

Every limit is held on one budget object shared by the whole tree, so a child
four levels down is spending the same allowance as the root, not a fresh copy
of it.

| Field | Default | Bounds |
|---|---|---|
| `max_cost` | 1.0 | dollars, whole run |
| `max_calls` | 2000 | calls, whole run — the backstop where a provider reports no price |
| `max_seconds` | 1800 | wall clock, whole run |
| `max_completion_tokens` | 500000 | completion tokens, whole run |
| `max_prompt_tokens` | 200000 | one call's input plus output — the ceiling that actually bounds context growth, since the conversation is resent every turn |
| `max_steps` | 20 | turns per agent |
| `max_depth` | 3 | how deep sub-agents nest |
| `max_concurrent` | 16 | pieces of one `gather_rlm` in flight |
| `max_live` | 32 | agents alive at once anywhere in the tree |
| `timeout` | 120 | seconds for one cell |
| `truncate_len` | 10000 | characters of a cell's output shown, kept from the end |

**Talking to a provider**

| Field | Default | Meaning |
|---|---|---|
| `api_timeout` | 60 | one read of a reply |
| `api_deadline` | 600 | a whole reply — without it, a response trickling a few bytes at a time resets the read timeout forever |
| `max_tokens` | unset | ceiling on one reply, sent with the request. Matters where a provider reserves credit for the largest reply a request *could* produce: unset, that reservation is the model's own maximum |
| `api_retry_after_max` | 60 | how far a provider's own `Retry-After` is obeyed. A refusal that names a time is retried; one that names none is taken as final |
| `api_max_retries` / `api_backoff` | 3 / 0.5 | retries and growth |

**Switches, each there to be turned off and measured**

| Field | Default | Effect |
|---|---|---|
| `enable_delegation` | true | bind `rlm` and `gather_rlm` at all |
| `enable_structured_output` | true | dicts stay dicts; `output_schema` is enforced |
| `enable_step_banner` | true | tell an agent its remaining steps once past halfway |
| `enable_batching_guard` | true | refuse hand-rolled fan-outs (below) |
| `enable_blind_final_guard` | false | hold a literal `FINAL` written in the same cell that first reads `PROMPT` |
| `enable_first_look_guard` | false | refuse a first-step answer that never touched `PROMPT` |
| `inherit_tools` | false | children get their parent's tools without being granted them |

## Where the code runs

| `runtime` | Isolation | Cost |
|---|---|---|
| `subprocess` (default) | its own Python process | ~0.2s to start, ~28MB per agent |
| `wasm` | Pyodide, no network — `js` and `pyodide.http` are shut | ~1.5s first boot, ~45MB per extra sandbox in a shared host |
| `in-process` | none | free; trusted code only, and the fan-out guard cannot hold here |

`wasm` closes the documented ways out of the sandbox. Pyodide shares a
JavaScript context with its host and was never built as a security boundary,
so this is a seal against model-written code doing something careless, not
against code trying to escape.

**Fan-outs must go through `gather_rlm`.** `asyncio.gather`, `wait`,
`as_completed`, `create_task`, `ensure_future` and `TaskGroup` are all refused
if handed a sub-agent call, with a message naming the helper. Not pedantry:
`gather_rlm` holds the fan-out inside `max_concurrent`, claims a live slot per
child, and cancels the remainder the moment one fails. A hand-built gather
does none of the three and nothing would have said so. `in-process` shares its
event loop with the caller and cannot enforce it.

## Sessions

A session carries the root agent's namespace and its answers between
questions, so the second question can use what the first one built:

```bash
rlmness --session ./notes "Parse the logs and index them by service"
rlmness --session ./notes "Using that index — which service failed most?"
```

Saved after every successful step, atomically, so a killed run resumes from
its last one. Variables are pickled, functions and classes kept as source,
modules by name, along with whatever the model committed, noted or commented
about them — a resumed session knows a name as well as the run that created it
did.

| Target | State lives in |
|---|---|
| `--session file.json` | that file |
| `--session dir` | `dir/state.json` |
| `--session dir --session-id x` | `dir/x/state.json` |
| `--session-ephemeral` | memory, gone at exit |

Two runs pointed at one file do not clobber each other: whichever finds the
file changed underneath it writes `state.1.json` instead, and any run loading
the original names the strays sitting beside it, so nobody's work goes
missing quietly. Sub-agents are always fresh — only the root agent persists.

## Watching and reading a run

Every run writes a JSONL trace and prints its path.

```bash
rlmness --dashboard "..."                     # live tree, code and namespace as it goes
rlmness-viewlog traces/run_x.jsonl            # one run as a tree
rlmness-viewlog traces/run_x.jsonl --tui      # the same, navigable
rlmness-viewlog --session ./notes             # every question a session answered, and what each cost
rlmness-viewlog --session ./notes --tui       # browse them, drill into any run, inspect memory
rlmness-timeline traces/run_x.jsonl           # who was running when
```

In Python, pass anything as `trace` — it receives `run_started`, `step`,
`final`, `run_completed` and `run_failed` for every agent, each carrying
`run_id`, `parent_run_id` and `depth`. `Broadcast` fans one run out to several
readers:

```python
from rlmness import Journal, solve
from rlmness.events import Broadcast

class Printer:
    def step(self, *, step, depth=0, **_):
        print(f"depth {depth} step {step}")

solve(prompt, client, config=config, trace=Broadcast(Journal(), Printer()))
```

## Measuring whether any of it helps

Every switch above exists so it can be removed. The ablation runner runs a
task set with a setting on and off and prints the spread:

```bash
python -m evals.ablation --model deepseek/deepseek-v4-flash --setting enable_step_banner
python -m evals.ablation --model deepseek/deepseek-v4-flash --tasks longbench:hotpotqa -n 5
```

The `sanity` set is generated, costs nothing, and shows only that nothing
broke — though its `ledger` tasks are the interesting ones, since the number
asked for is a running total that appears nowhere in the text and so cannot be
found by searching. `longbench` is what supports a claim that something got
better.

## License

MIT
