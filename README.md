# rlm-ness

A Python implementation of Recursive Language Models, from the paper
[Recursive Language Models](https://arxiv.org/abs/2512.24601) (Zhang, Kraska &
Khattab), running the model's code in Node and Pyodide.

Your full input sits in a Python REPL as `PROMPT`. The model sees an opening
preview, then writes code to work through the input and hands pieces to
sub-agents of itself, whose answers come back as plain values in the REPL.

## Install

```bash
pip install rlmness
export OPENROUTER_API_KEY=sk-or-...        # or ANTHROPIC_API_KEY, DEEPSEEK_API_KEY
rlmness "How many r's are in strawberry?" --model z-ai/glm-5
```

That is the whole install. Nothing else is required, and no config file is
needed.

**Optional:** `rlmness --setup` installs the WebAssembly sandbox the model's
code can run in — a one-time download that needs [Node.js](https://nodejs.org)
18+. Until you run it, `rlmness` runs the code in a subprocess instead and
says so. It also leaves a fully commented `rlmness.yaml` in `~/.rlmness` if
you would rather edit settings than pass flags.

## Run it

```bash
rlmness "How many r's are in strawberry?" --model z-ai/glm-5

# a file as the data, the question stays separate
rlmness "Which errors repeat most, and when?" --input-file server.log --model z-ai/glm-5

# or pipe it in
cat server.log | rlmness --instruction "Which errors repeat most?" --model z-ai/glm-5

# a cheaper model for the sub-agents the root hands pieces to
rlmness "..." --model z-ai/glm-5 --sub-model minimax/minimax-m2.5
```

A model is never guessed for you, since it decides what a run costs: pass
`--model`, or set `primary_agent` in a config file and drop the flag.

`--input-file` parses by extension — `.json`, `.jsonl`, `.yaml` arrive as the
data they describe, anything else as text the model slices itself.

`--provider anthropic` or `--provider deepseek` switches API; each reads its
own key. Run `rlmness --help` for the rest.

Run `rlmness` with no question to open the live screen and ask from there.

## Choose where the code runs

```bash
rlmness --runtime wasm "..."          # default
rlmness --runtime subprocess "..."
rlmness --runtime in-process "..."
```

- **`wasm`** (default) — a WebAssembly sandbox that blocks ordinary network
  and host-file access. It limits what model-written code can reach, but is
  not a hardened security boundary. Needs `rlmness --setup`; until then runs
  fall back to `subprocess` and say so.
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
pip install "rlmness[tui]"

rlmness --dashboard "..."                   # live: agents, code, output, variables
rlmness-viewlog traces/run_x.jsonl --tui    # replay a finished run
rlmness-viewlog ./notes --tui               # every question a session answered
rlmness-timeline traces/run_x.jsonl         # which agent ran when
```

Each run saves a trace under `traces/` in the current folder and prints its
path at the end.

## Configuration

Optional — every setting can be passed as a flag. A config file just saves
repeating them:

```yaml
# rlmness.yaml
primary_agent: z-ai/glm-5
sub_agent: minimax/minimax-m2.5
max_cost: 1.0
```

The first of these that exists is used, and flags override it for one run:

1. `--config <path>`
2. `./rlmness.yaml`
3. `~/.rlmness/rlmness.yaml`

Write either file yourself, or let `rlmness --setup` drop a commented one in
`~/.rlmness` for you.

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
