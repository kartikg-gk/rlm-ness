# rlm-ness

A **Recursive Language Model** runtime, from the paper
[Recursive Language Models](https://arxiv.org/abs/2512.24601) (Zhang, Kraska &
Khattab).

Your input is never pasted into the model's context. It sits in a Python REPL
as `PROMPT`, the model writes code to work through it, and hands pieces to
sub-agents of itself when a piece needs reading rather than searching. What a
sub-agent returns lands back in the REPL as a plain value.

## Install

```bash
pip install rlm-ness
rlmness --setup
export OPENROUTER_API_KEY=sk-or-...        # or ANTHROPIC_API_KEY, DEEPSEEK_API_KEY
```

`rlmness --setup` installs the sandbox the model's code runs in (needs
[Node.js](https://nodejs.org) 18+) and writes a starter config to
`~/.rlmness/rlmness.yaml`. Skip it and everything still works, on a less
isolated runtime.

## Run it

```bash
rlmness "How many r's are in strawberry?"
rlmness --input-file server.log "Which errors repeat most, and when?"
cat server.log | rlmness --instruction "Which errors repeat most, and when?"
rlmness --model deepseek/deepseek-v4-flash "..."
rlmness --provider anthropic "..."           # or deepseek; each reads its own key
```

`--input-file` takes the data and the question stays separate. A `.json`,
`.jsonl` or `.yaml` file arrives as the data it describes; anything else
arrives as text for the model to slice.

Run `rlmness` with no question to open the live screen and ask from there.

## Choose where the code runs

```bash
rlmness --runtime wasm "..."          # default
rlmness --runtime subprocess "..."
rlmness --runtime in-process "..."
```

- **`wasm`** (default) — a WebAssembly sandbox with no network and no access to
  your files. Safe on input you didn't write: fetched pages, uploaded
  documents, other people's data. Set up by `rlmness --setup`; until then
  `rlmness` uses `subprocess` and tells you so.
- **`subprocess`** — a separate Python process. Starts faster and can use any
  package you have installed, but can touch your machine. For input you trust.
- **`in-process`** — inside your own Python process, no isolation. For trusted
  code where your tools need to be live objects.

To change the default, set it once instead of passing the flag:

```bash
export RLMNESS_RUNTIME=subprocess
```

or put `runtime: subprocess` in your config. On the live screen, a picker next
to the question box switches it per question.

## Keep working across questions

```bash
rlmness --session ./notes "Parse the logs and index them by service"
rlmness --session ./notes "Using that index, which service failed most?"
```

The second question starts with everything the first one built. Progress is
saved after every step, so an interrupted run picks up where it stopped.
`--session-id` keeps several sessions in one folder; `--session-ephemeral`
keeps one only until the program exits.

## Watch a run

```bash
pip install "rlm-ness[tui]"

rlmness --dashboard "..."                   # live: agents, code, output, variables
rlmness-viewlog traces/run_x.jsonl --tui    # replay a finished run
rlmness-viewlog ./notes --tui               # every question a session answered
rlmness-timeline traces/run_x.jsonl         # which agent ran when
```

Each run saves a trace under `traces/` in the current folder and prints its
path at the end.

## Configuration

`rlmness --setup` writes `~/.rlmness/rlmness.yaml`, with every setting in it
and a comment on what each one does. Change it there, or keep a
`rlmness.yaml` beside your work for one project.

Which file is used, first one found:

1. the file given with `--config`
2. `rlmness.yaml` in the current folder
3. `~/.rlmness/rlmness.yaml` (move it with `RLMNESS_HOME`)

`--model`, `--runtime` and `--provider` override the file for a single run.

The settings worth knowing about:

| Setting | Default | |
|---|---|---|
| `primary_agent` | — | model for the root agent (required) |
| `sub_agent` | same as `primary_agent` | model for the agents it starts |
| `provider` | `openrouter` | `openrouter`, `anthropic` or `deepseek` |
| `runtime` | `wasm` | where the model's code runs |
| `max_cost` | `1.0` | dollar limit for a whole run |
| `max_seconds` | `1800` | time limit for a whole run |
| `max_steps` | `20` | turns each agent gets |
| `max_depth` | `3` | how deeply sub-agents can nest |

## Use it from Python

```python
from rlmness import Config, make_client, solve

answer = solve(
    open("server.log").read(),
    make_client("openrouter"),
    config=Config(primary_agent="z-ai/glm-5", sub_agent="minimax/minimax-m2.5"),
    instruction="Which service failed most, and when?",
)
print(answer.output, answer.usage.cost)
```

`solve()` also takes `tools=` (your functions, callable in the REPL, and any
other value as a plain variable) and `output_schema=` (a JSON Schema the
answer must match). `load_config()` reads the same file the command does.

## License

MIT
