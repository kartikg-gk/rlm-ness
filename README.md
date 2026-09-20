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

- **`tools`** — functions become callable in the REPL; other values (strings,
  dicts, tables) become plain variables the model can read.
- **`output_schema`** — the answer must match this JSON Schema. A mismatch goes
  back to the model to fix, without losing its work.
- **`prompt`** can be a dict or list as well as text.
- **`Config(runtime="subprocess")`** picks the runtime; `load_config()` reads
  the same config file the command uses.

## Configuration

`rlmness` reads the first config it finds:

1. the file given with `--config`
2. `rlmness.yaml` in the current folder
3. `~/.rlmness/rlmness.yaml` (move it with `RLMNESS_HOME`)

`--model`, `--runtime` and `--provider` override the file for one run.

| Setting | Default | |
|---|---|---|
| `primary_agent` | — | model for the root agent (required) |
| `sub_agent` | same as `primary_agent` | model for the agents it starts |
| `runtime` | `wasm` | `wasm`, `subprocess` or `in-process` |
| `provider` | `openrouter` | `openrouter`, `anthropic` or `deepseek` |
| `max_cost` | `1.0` | dollar limit for a whole run |
| `max_seconds` | `1800` | time limit for a whole run |
| `max_steps` | `20` | turns per agent |
| `max_depth` | `3` | how deeply sub-agents can nest |
| `max_concurrent` | `16` | sub-agents running at once in one batch |
| `max_tokens` | unset | cap on one reply; set it if your key has little credit |

<details>
<summary>All settings</summary>

| Setting | Default | |
|---|---|---|
| `max_calls` | `2000` | model calls in a whole run |
| `max_completion_tokens` | `500000` | output tokens in a whole run |
| `max_prompt_tokens` | `200000` | input plus output of a single call |
| `max_live` | `32` | agents alive at once across the whole run |
| `timeout` | `120` | seconds one code cell may run |
| `truncate_len` | `10000` | characters of a cell's output the model sees |
| `api_timeout` | `60` | seconds to wait on each read from the provider |
| `api_deadline` | `600` | seconds for a whole reply |
| `api_max_retries` | `3` | retries on a failed call |
| `api_backoff` | `0.5` | seconds before the first retry, doubling after |
| `api_retry_after_max` | `60` | longest wait the provider may ask for between retries |
| `temperature` | `0.1` | sampling temperature |
| `reasoning_effort` | `low` | reasoning level, where the model supports it |
| `inherit_tools` | `false` | sub-agents get their parent's tools without being given them |
| `enable_delegation` | `true` | let agents start sub-agents |
| `enable_structured_output` | `true` | keep dicts and lists as they are; check `output_schema` |
| `enable_compression_guard` | `true` | ask an agent to confirm before it hands a sub-agent most of its own input |
| `compression_min_chars` | `5000` | inputs smaller than this are never questioned |
| `compression_ratio` | `0.6` | share of the parent's input that counts as barely reduced |
| `enable_step_banner` | `true` | tell an agent how many turns it has left |
| `enable_batching_guard` | `true` | require sub-agents to be started in batches through `gather_rlm` |
| `enable_blind_final_guard` | `false` | hold back an answer written before the data was read |
| `enable_first_look_guard` | `false` | refuse a first-turn answer that never looked at the input |

</details>

## License

MIT
