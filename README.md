# rlm-ness

A **Recursive Language Model** runtime, from the paper
[Recursive Language Models](https://arxiv.org/abs/2512.24601) (Zhang, Kraska &
Khattab).

Your input is never pasted into the model's context. It sits in a Python REPL
as `PROMPT`, the model writes code to work through it, and hands pieces to
sub-agents of itself when a piece needs reading rather than searching. What a
sub-agent returns lands back in the REPL as a plain value.

## Run it

```bash
pip install -e .
npm install                     # sandbox runtime, needs Node 18+
export OPENROUTER_API_KEY=sk-or-...

rlmness "How many r's are in strawberry?"
cat server.log | rlmness --instruction "Which errors repeat most, and when?"
rlmness --provider deepseek --model deepseek-chat "..."
```

Models are set in `rlmness.yaml` (`primary_agent` for the root, `sub_agent` for
the agents it spawns) or with `--model`. Keys come from the provider's own
variable: `OPENROUTER_API_KEY` or `DEEPSEEK_API_KEY`.

Run `rlmness` with no question to open the live screen and ask from there.

## Choose where the code runs

Pick with `--runtime`, no code changes:

```bash
rlmness --runtime wasm "..."          # default
rlmness --runtime subprocess "..."
rlmness --runtime in-process "..."
```

- **`wasm`** (default) — Python compiled to WebAssembly, hosted in Node. No
  network and no access to your files, so it is safe on input you didn't write:
  fetched pages, uploaded documents, other people's data. Needs `npm install`
  once. Without it, `rlmness` runs on `subprocess` and tells you so.
- **`subprocess`** — the model's code runs in its own Python process. Starts
  faster, and can use any package you have installed. It can touch your
  machine, so keep it for input you trust.
- **`in-process`** — runs inside your own Python process. No isolation. For
  trusted code where your tools need to be live objects.

To change the default, set it once:

```bash
export RLMNESS_RUNTIME=subprocess     # this shell
```

```yaml
runtime: subprocess                   # rlmness.yaml, every run
```

On the live screen, a picker next to the question box does the same thing per
question, and only lists runtimes that can start on your machine.

## Keep working across questions

```bash
rlmness --session ./notes "Parse the logs and index them by service"
rlmness --session ./notes "Using that index, which service failed most?"
```

The second question starts with everything the first one built. State is saved
after every step, so an interrupted run picks up where it stopped.
`--session-id` keeps several sessions in one folder, `--session-ephemeral` keeps
one only for the life of the process.

## Watch a run

```bash
pip install -e '.[tui]'

rlmness --dashboard "..."                   # live: agents, code, output, variables
rlmness-viewlog traces/run_x.jsonl --tui    # replay a finished run
rlmness-viewlog ./notes --tui               # every question a session answered
rlmness-timeline traces/run_x.jsonl         # which agent ran when
```

Every run writes its trace under `traces/` and prints the path at the end.

## Use it from Python

```python
from rlmness import Config, make_client, solve

def lookup_owner(service: str) -> str:
    """Team that owns a service."""
    return {"billing": "payments", "auth": "identity"}.get(service, "unknown")

answer = solve(
    open("server.log").read(),
    make_client("openrouter"),
    config=Config(primary_agent="z-ai/glm-5", sub_agent="minimax/minimax-m2.5"),
    instruction="Which service failed most, and who owns it?",
    tools={"lookup_owner": lookup_owner, "region": "eu-west-1"},
    output_schema={"type": "object", "required": ["service", "owner"]},
)
print(answer.output, answer.usage.cost)
```

- **`tools`** — functions become callable in the REPL; anything else (strings,
  dicts, tables) becomes a plain variable the model can read.
- **`output_schema`** — the answer must match this JSON Schema. A mismatch is
  sent back to the model to fix, without losing its work.
- **`PROMPT` can be a dict or list**, not just text, and sub-agents can be handed
  dicts too.
- **`runtime_factory`** — pick the runtime in code, e.g.
  `from rlmness.wasm_runtime import WasmRuntime`.

## Configuration

Everything lives in `rlmness.yaml`; `--config` points at another file. The
settings most people touch:

| Setting | Default | |
|---|---|---|
| `primary_agent` / `sub_agent` | — | models for the root and its sub-agents |
| `runtime` | `wasm` | where the code runs |
| `provider` | `openrouter` | `openrouter` or `deepseek` |
| `max_cost` | `1.0` | dollar limit for a whole run |
| `max_seconds` | `1800` | time limit for a whole run |
| `max_steps` | `20` | turns per agent |
| `max_depth` | `3` | how deeply sub-agents can nest |
| `max_concurrent` | `16` | sub-agents running at once in one batch |
| `max_tokens` | unset | cap on one reply; set it if your key has little credit |

Limits apply to the whole run, sub-agents included. The file lists every other
setting with its default.

## Layout

```
rlmness/
  console.py         the rlmness command
  engine.py          the loop: ask the model, run its code, feed back, repeat
  briefing.py        what the model is told
  providers.py       OpenRouter and DeepSeek clients, retries, deadlines
  config.py          rlmness.yaml
  limits.py          cost, time, call and depth limits shared across a run
  runtime.py         subprocess runtime
  cell_runner.py       its side of the process
  wasm_runtime.py    WebAssembly runtime
  wasm_guest.mjs       its side, in Node
  in_process.py      in-process runtime
  tools.py           turning your functions and values into REPL names
  schema.py          output schema checks
  session.py         sessions: save, restore, conflicts
  session_guest.py     the REPL side of saving
  events.py          run events, shared by every viewer
  journal.py         the JSONL trace
  dashboard.py       live screen
  session_view.py    session browser
  viewlog.py         rlmness-viewlog
  timeline.py        rlmness-timeline
  namespace.py       summarising REPL variables for display
evals/               comparing settings on generated and LongBench tasks
```

## License

MIT
