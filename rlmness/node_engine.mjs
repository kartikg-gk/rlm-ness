// The Node engine: the agent loop, recursion, batches, background
// agents and the budget, all inside this one Node process.
//
// A child agent is a direct call to runAgent. Python is reached only for what
// lives there: frontends, sessions on disk, schema checks and any model route
// without a native client here. The words of the loop and the checks made on a
// reply are not written twice: one private interpreter runs the same Python
// source the other engine imports.
//
// Cells run in separate workers. Coordinator heartbeats continue while a worker
// is busy, so its deadline can terminate it without resetting another agent.

import process from "node:process";
import readline from "node:readline";
import { randomUUID } from "node:crypto";

import { loadPyodide, openSandbox } from "./node_sandbox.mjs";

const PROTOCOL = 1;
const HEARTBEAT_MS = 1000;
const POLL_MS = 100;
const resident = new Map();
const activeSessions = new Map();

function closeSession(id) {
  resident.get(id)?.close();
  resident.delete(id);
  const active = activeSessions.get(id);
  if (active) {
    active.token.cancel("session closed");
    active.box.sandbox?.close();
  }
}

function send(record) {
  process.stdout.write(JSON.stringify({ v: PROTOCOL, ...record }) + "\n");
}

class EngineError extends Error {
  constructor(name, message, ref = null) {
    super(message);
    this.name = name;
    this.ref = ref;
  }
}

const describe = (error) => `${error.name}: ${error.message}`;

// Cancellation that reaches everything under it: a token's children are told
// when it is, and anything waiting can ask to be woken.
class Token {
  constructor(parent = null) {
    this.cancelled = false;
    this.reason = null;
    this.waiters = new Set();
    this.parent = parent;
    if (parent) {
      if (parent.cancelled) this.cancel(parent.reason);
      else parent.waiters.add(this);
    }
  }

  cancel(reason) {
    if (this.cancelled) return;
    this.cancelled = true;
    this.reason = reason;
    if (this.parent) this.parent.waiters.delete(this);
    for (const waiter of [...this.waiters]) {
      if (waiter instanceof Token) waiter.cancel(reason);
      else waiter(reason);
    }
    this.waiters.clear();
  }

  detach() {
    if (this.parent) this.parent.waiters.delete(this);
  }

  // Settles with `work`, or rejects as soon as this token is cancelled.
  race(work) {
    if (this.cancelled) {
      work.catch?.(() => {});
      return Promise.reject(new EngineError("Abandoned", this.reason));
    }
    return new Promise((resolve, reject) => {
      const wake = (reason) => reject(new EngineError("Abandoned", reason));
      this.waiters.add(wake);
      work.then(
        (value) => { this.waiters.delete(wake); resolve(value); },
        (error) => { this.waiters.delete(wake); reject(error); },
      );
    });
  }
}

// The one budget of a run. Claiming a call and checking the limits happen
// together, in one synchronous stretch, so a concurrent batch cannot pass the
// check before any of it has spent.
class Budget {
  constructor(limits, owner) {
    Object.assign(this, limits);
    this.owner = owner;
    this.started = performance.now() - 1000 * limits.elapsed_seconds;
    this.slots = Math.max(1, this.max_live);
    this.liveNow = 0;
    this.liveMost = 0;
    this.sandboxes = 0;
    this.cancelled = false;
  }

  sync(limits) {
    for (const name of ["max_depth", "max_calls", "max_cost", "max_seconds",
      "max_completion_tokens", "max_prompt_tokens"]) this[name] = limits[name];
    this.slots = Math.max(0, this.slots + limits.max_live - this.max_live);
    this.max_live = limits.max_live;
    for (const name of ["calls", "cost", "completion_tokens", "prompt_tokens", "unpriced_calls"]) {
      this[name] = Math.max(this[name], limits[name]);
    }
  }

  report(delta) {
    send({ type: "budget", allowanceId: this.owner, delta });
  }

  check() {
    if (this.cancelled) throw new EngineError("Abandoned", "work was abandoned");
    if (this.max_seconds !== null) {
      const elapsed = (performance.now() - this.started) / 1000;
      if (elapsed >= this.max_seconds) {
        throw new EngineError("TimeUp", `${elapsed.toFixed(1)}s reaches the limit of ${this.max_seconds}s`);
      }
    }
    if (this.calls >= this.max_calls) {
      throw new EngineError("AllowanceSpent", `${this.calls} calls reaches the limit of ${this.max_calls}`);
    }
    if (this.cost >= this.max_cost) {
      throw new EngineError("AllowanceSpent", `cost ${this.cost.toFixed(4)} reaches the limit of ${this.max_cost}`);
    }
    if (this.max_completion_tokens !== null && this.completion_tokens >= this.max_completion_tokens) {
      throw new EngineError(
        "AllowanceSpent",
        `${this.completion_tokens} completion tokens reaches the limit of ${this.max_completion_tokens}`,
      );
    }
  }

  reserve() {
    this.check();
    this.calls += 1;
    this.report({ calls: 1 });
  }

  settle(usage) {
    this.completion_tokens += usage.completion_tokens;
    this.prompt_tokens += usage.prompt_tokens;
    if (usage.cost === null || usage.cost === undefined) this.unpriced_calls += 1;
    else this.cost += usage.cost;
    this.report({ completion_tokens: usage.completion_tokens, prompt_tokens: usage.prompt_tokens,
      cost: usage.cost ?? 0, unpriced_calls: usage.cost == null ? 1 : 0 });
    const carried = usage.prompt_tokens + usage.completion_tokens;
    if (this.max_prompt_tokens !== null && carried > this.max_prompt_tokens) {
      throw new EngineError(
        "AllowanceSpent",
        `${carried} tokens in one call (input ${usage.prompt_tokens} + output ${usage.completion_tokens}) passes the per-call limit of ${this.max_prompt_tokens}`,
      );
    }
  }

  // A brake rather than a gate, as in the other engine: a child that gets no
  // slot still runs, because refusing would deadlock a tree against itself.
  claimSlot() {
    if (this.slots > 0) { this.slots -= 1; return 1; }
    return 0;
  }

  releaseSlot(held) {
    this.slots += held;
  }

  sandboxOpened() {
    this.sandboxes += 1;
    this.liveNow += 1;
    this.liveMost = Math.max(this.liveMost, this.liveNow);
  }

  sandboxClosed() {
    this.liveNow -= 1;
  }

  snapshot() {
    return {
      calls: this.calls, cost: this.cost,
      completion_tokens: this.completion_tokens, prompt_tokens: this.prompt_tokens,
      unpriced_calls: this.unpriced_calls,
      sandboxes: this.sandboxes, live_most: this.liveMost,
    };
  }
}

// A batch admits at most width active branches. Parked branches keep their
// interpreter but return the permit and reacquire it before resuming inference.
class FanoutSlots {
  constructor(width, token) {
    this.width = width;
    this.token = token;
    this.active = 0;
    this.starting = new Map();
    this.resuming = new Map();
  }

  acquire(resuming) {
    if (this.token.cancelled) {
      return Promise.reject(new EngineError("Abandoned", this.token.reason));
    }
    if (this.active < this.width) {
      this.active += 1;
      return Promise.resolve();
    }
    const queue = resuming ? this.resuming : this.starting;
    return new Promise((resolve, reject) => {
      const cancel = (reason) => {
        queue.delete(cancel);
        reject(new EngineError("Abandoned", reason));
      };
      queue.set(cancel, resolve);
      this.token.waiters.add(cancel);
    });
  }

  release() {
    for (const queue of [this.resuming, this.starting]) {
      const waiting = queue.entries().next();
      if (waiting.done) continue;
      const [cancel, admit] = waiting.value;
      queue.delete(cancel);
      this.token.waiters.delete(cancel);
      admit();
      return;
    }
    this.active -= 1;
  }

  lease(budget) {
    let held = null;
    const acquire = async (resuming = false) => {
      await this.acquire(resuming);
      held = budget.claimSlot();
    };
    const release = () => {
      if (held === null) return;
      budget.releaseSlot(held);
      held = null;
      this.release();
    };
    return { acquire, release, park: async (wait) => {
      release();
      const result = await wait();
      await acquire(true);
      return result;
    } };
  }
}

function emptySpend() {
  return {
    prompt_tokens: 0, completion_tokens: 0, total_tokens: 0,
    cost: null, cached_tokens: null, reasoning_tokens: null,
  };
}

const sumOrNull = (a, b) => (a == null && b == null ? null : (a ?? 0) + (b ?? 0));

function addSpend(left, right) {
  return {
    prompt_tokens: left.prompt_tokens + right.prompt_tokens,
    completion_tokens: left.completion_tokens + right.completion_tokens,
    total_tokens: left.total_tokens + right.total_tokens,
    cost: sumOrNull(left.cost, right.cost),
    cached_tokens: sumOrNull(left.cached_tokens, right.cached_tokens),
    reasoning_tokens: sumOrNull(left.reasoning_tokens, right.reasoning_tokens),
  };
}

// --- Calling back into Python ------------------------------------------------

const waiting = new Map();
let nextRequest = 1;
const stats = { callbacks: 0 };

function callback(run, agentId, kind, args, token = null) {
  const requestId = nextRequest++;
  stats.callbacks += 1;
  const reply = new Promise((resolve, reject) => {
    waiting.set(requestId, { resolve, reject });
  });
  send({ type: "callback", runId: run.id, agentId, requestId, kind, args });
  if (!token) return reply;
  return token.race(reply).catch((error) => {
    // The answer may still arrive. It has nowhere to go, and Python is told it
    // need not hurry; a call already running there cannot be pulled back.
    if (waiting.delete(requestId)) send({ type: "callback_cancelled", runId: run.id, requestId });
    throw error;
  });
}

function answered(record) {
  const entry = waiting.get(record.requestId);
  if (!entry) return;
  waiting.delete(record.requestId);
  if (record.ok) entry.resolve(record.value);
  else entry.reject(new EngineError(record.error.name, record.error.message, record.error.ref ?? null));
}

// --- The shared words and checks ---------------------------------------------

let logicReady = null;

const LOGIC_PY = `
import json, sys, types

def _load(name, source):
    module = types.ModuleType(name)
    sys.modules[name] = module
    exec(source, module.__dict__)
    return module

_briefing = _load("rlmness_briefing", _BRIEFING_SRC)
_text = _load("rlmness_loop_text", _LOOP_TEXT_SRC)

def _tools(specs):
    return [types.SimpleNamespace(**spec) for spec in specs]

_CALLS = {
    "system_prompt": lambda recurse, tools, structured, schema: _briefing.system_prompt(
        recurse, _tools(tools), sealed=True, structured=structured, schema=schema),
    "opening_code": _briefing.opening_code,
    "opening_message": lambda code, shown, instruction, limit, child: _briefing.opening_message(
        code, shown, instruction, limit, is_child=child),
    "shows_everything": _briefing.shows_everything,
    "preview": lambda: _briefing.PREVIEW,
    "words": lambda: {name: getattr(_text, name) for name in (
        "SCHEMA_SHOWN", "SCHEMA_REFUSED", "NO_CODE", "NO_TOOLS", "TOO_DEEP",
        "REFUSED_HANDOFF", "UNSEEN", "BLIND")},
    "runnable_code": _text.runnable_code,
    "answers_without_looking": _text.answers_without_looking,
    "writes_answer_blind": _text.writes_answer_blind,
    "approves": _text.approves,
    "gist": _text._gist,
    "budget_banner": _text.budget_banner,
    "label_output": _text.label_output,
    "handoff_question": _text.handoff_question,
    "as_text": lambda value: json.dumps(value, default=str),
    "facts": lambda value: [type(value).__name__, len(str(value))],
}

def _logic(name, args_json):
    return json.dumps(_CALLS[name](*json.loads(args_json)))
`;

let logicCall = null;
let words = null;

function startLogic(sources) {
  logicReady ??= (async () => {
    const py = await loadPyodide();
    py.setStdout({ batched: () => {} });
    py.setStderr({ batched: () => {} });
    py.globals.set("_BRIEFING_SRC", sources.briefing);
    py.globals.set("_LOOP_TEXT_SRC", sources.loop_text);
    await py.runPythonAsync(LOGIC_PY);
    const call = py.globals.get("_logic");
    logicCall = (name, ...args) => JSON.parse(call(name, JSON.stringify(args)));
    words = logicCall("words");
    words.PREVIEW = logicCall("preview");
  })();
  return logicReady;
}

const fill = (template, values) => template.replace(/\{(\w+)\}/g, (whole, name) => (name in values ? values[name] : whole));

// --- Native model routes -----------------------------------------------------

const RETRIABLE = new Set([408, 409, 425, 429, 500, 502, 503, 504]);
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

class HttpTimeout extends EngineError {
  constructor(message) { super("ReadTimeout", message); }
}

// One request, read under a bound on each read and on the whole reply.
async function sendOnce(route, body, token) {
  const control = new AbortController();
  const wake = () => control.abort();
  if (token) token.waiters.add(wake);
  let quiet = null;
  const arm = () => {
    clearTimeout(quiet);
    quiet = setTimeout(() => control.abort(new HttpTimeout(`no data for ${route.timeout}s`)), route.timeout * 1000);
  };
  const limit = route.deadline == null ? Infinity : performance.now() + route.deadline * 1000;
  try {
    arm();
    const response = await fetch(route.endpoint, {
      method: "POST",
      headers: { "content-type": "application/json", ...route.headers },
      body: JSON.stringify(body),
      signal: control.signal,
    });
    const pieces = [];
    for await (const piece of response.body ?? []) {
      arm();
      pieces.push(piece);
      if (performance.now() > limit) throw new HttpTimeout(`no complete reply within ${route.deadline}s`);
    }
    return { status: response.status, headers: response.headers, text: Buffer.concat(pieces).toString("utf8") };
  } catch (error) {
    if (token?.cancelled) throw new EngineError("Abandoned", token.reason);
    if (control.signal.reason instanceof HttpTimeout) throw control.signal.reason;
    if (error instanceof EngineError) throw error;
    if (error.name === "AbortError" || error.name === "TimeoutError") throw new HttpTimeout(String(error.message));
    throw new EngineError("TransportError", String(error.cause?.message ?? error.message));
  } finally {
    clearTimeout(quiet);
    if (token) token.waiters.delete(wake);
  }
}

function requestBody(route, messages, model) {
  if (route.shape === "anthropic") {
    const system = messages.filter((m) => m.role === "system").map((m) => m.content ?? "").join("\n\n");
    const body = {
      model,
      messages: messages.filter((m) => m.role !== "system"),
      max_tokens: route.max_tokens || route.default_max_tokens,
    };
    if (system) body.system = system;
    if (route.temperature !== null) body.temperature = route.temperature;
    return body;
  }
  const body = { model, messages };
  if (route.temperature !== null) body.temperature = route.temperature;
  if (route.reasoning_effort && !route.reasoningRefused) body.reasoning = { effort: route.reasoning_effort };
  if (route.max_tokens !== null) body.max_tokens = route.max_tokens;
  return body;
}

async function postWithRetries(route, messages, model, token) {
  for (let attempt = 0; attempt <= route.max_retries; attempt += 1) {
    const last = attempt === route.max_retries;
    let delay = route.backoff * 2 ** attempt;
    try {
      const response = await sendOnce(route, requestBody(route, messages, model), token);
      if (route.shape !== "anthropic" && response.status === 400 && route.reasoning_effort && !route.reasoningRefused) {
        route.reasoningRefused = true;
        continue;
      }
      const raw = Number.parseFloat(response.headers.get("retry-after") ?? "");
      const told = Number.isNaN(raw) ? null : Math.min(Math.max(raw, 0), route.retry_after_max);
      const retriable = RETRIABLE.has(response.status) || told !== null;
      if (!retriable || last) {
        if (response.status >= 400) {
          throw new EngineError("HTTPStatusError", `${response.status} from ${route.name}: ${response.text.slice(0, 500)}`);
        }
        return JSON.parse(response.text);
      }
      if (told !== null) delay = told;
    } catch (error) {
      if (error instanceof HttpTimeout) {
        if (last || attempt >= 1) throw error;
      } else if (error.name === "TransportError") {
        if (last) throw error;
      } else {
        throw error;
      }
    }
    await (token ? token.race(sleep(delay * 1000)) : sleep(delay * 1000));
  }
  throw new EngineError("AssertionError", "unreachable");
}

const num = (value) => (typeof value === "number" ? value : null);

function chatCost(usage) {
  const top = num(usage.cost);
  const upstream = usage.cost_details && typeof usage.cost_details === "object"
    ? num(usage.cost_details.upstream_inference_cost) : null;
  if (usage.is_byok === true && upstream !== null) return upstream + (top && top > 0 ? top : 0);
  if (top !== null && top > 0) return top;
  if (upstream !== null && upstream > 0) return upstream;
  return top;
}

function detail(usage, section, field) {
  const nested = usage[section];
  if (nested && typeof nested === "object" && nested[field] != null) return Math.trunc(nested[field]);
  return usage[field] != null ? Math.trunc(usage[field]) : null;
}

// Tool calls written out as text need Python's own repr of their arguments.
async function nativeCalls(run, agentId, calls) {
  return calls.length ? callback(run, agentId, "native_calls", { calls }) : "";
}

async function nativeReply(run, agentId, messages, model, token) {
  const route = run.route;
  const payload = await postWithRetries(route, messages, model, token);
  const raw = payload.usage ?? {};
  if (route.shape === "anthropic") {
    const blocks = payload.content ?? [];
    const text = blocks.filter((b) => b.type === "text").map((b) => b.text ?? "").join("\n")
      + await nativeCalls(run, agentId, blocks.filter((b) => b.type === "tool_use")
        .map((b) => [b.name ?? "?", b.input ?? {}]));
    const thinking = blocks.filter((b) => b.type === "thinking").map((b) => b.thinking ?? "").join("\n");
    const cached = num(raw.cache_read_input_tokens);
    const prompt = Math.trunc(raw.input_tokens ?? 0) + Math.trunc(cached ?? 0)
      + Math.trunc(num(raw.cache_creation_input_tokens) ?? 0);
    const completion = Math.trunc(raw.output_tokens ?? 0);
    return {
      text, reasoning: thinking || null,
      usage: {
        prompt_tokens: prompt, completion_tokens: completion, total_tokens: prompt + completion,
        cost: null, cached_tokens: cached === null ? null : Math.trunc(cached), reasoning_tokens: null,
      },
    };
  }
  const message = payload.choices[0].message;
  const text = (message.content ?? "") + await nativeCalls(run, agentId, (message.tool_calls ?? [])
    .map((call) => [call.function?.name ?? "?", call.function?.arguments ?? ""]));
  let reasoning = null;
  if (typeof message.reasoning === "string") reasoning = message.reasoning || null;
  else if (Array.isArray(message.reasoning)) {
    reasoning = message.reasoning.filter((b) => b && typeof b === "object")
      .map((b) => b.text || b.thinking || "").filter(Boolean).join("\n") || null;
  }
  return {
    text, reasoning,
    usage: {
      prompt_tokens: Math.trunc(raw.prompt_tokens ?? 0),
      completion_tokens: Math.trunc(raw.completion_tokens ?? 0),
      total_tokens: Math.trunc(raw.total_tokens ?? 0),
      cost: chatCost(raw),
      cached_tokens: detail(raw, "prompt_tokens_details", "cached_tokens"),
      reasoning_tokens: detail(raw, "completion_tokens_details", "reasoning_tokens"),
    },
  };
}

function modelReply(run, agentId, messages, model, token) {
  run.modelCalls += 1;
  if (run.route) return nativeReply(run, agentId, messages, model, token);
  return callback(run, agentId, "model", { messages, model }, token);
}

// --- Background agents -------------------------------------------------------

// Lives as long as the Python Background object it stands for, across runs.
class BackgroundAgents {
  constructor(maxAgents, records) {
    this.jobs = new Map();
    this.maxAgents = maxAgents;
    this.owners = new Map();
    for (const [key, record] of Object.entries(records ?? {})) {
      const job = newJob(key, record.parent, true, record.data, record.model);
      job.value = record.value ?? null;
      job.outbox = record.messages ?? [];
      for (const [receipt, item] of Object.entries(record.deliveries ?? {})) {
        job.deliveries.set(receipt, item.status === "running" ? { status: "interrupted" } : item);
      }
      job.status = (record.status ?? "idle") === "idle" ? "idle" : "failed";
      if (job.status === "failed") {
        job.error = new EngineError("RuntimeError", record.error ?? "child was interrupted");
      }
      job.finish();
      this.jobs.set(key, job);
    }
  }

  drain(runId) {
    const key = this.owners.get(runId);
    if (key === undefined) return [];
    const job = this.jobs.get(key);
    const items = job.inbox.filter((item) => item !== null);
    job.inbox.length = 0;
    return items;
  }

  close() {
    for (const job of this.jobs.values()) {
      if (job.sessionId !== null) closeSession(job.sessionId);
      job.token.cancel("agent cancelled");
      job.finish();
    }
  }
}

function newJob(key, parent, keep, data, model) {
  const job = {
    key, parent, keep, data, model, token: new Token(), task: null,
    sessionId: null, status: "starting", value: null, error: null, saveTail: Promise.resolve(),
    inbox: [], outbox: [], deliveries: new Map(), readyWaiters: [], isReady: false,
    finish() {
      job.isReady = true;
      for (const wake of job.readyWaiters.splice(0)) wake();
    },
    whenReady(seconds) {
      if (job.isReady) return Promise.resolve(true);
      return new Promise((resolve) => {
        let timer = null;
        const wake = () => { clearTimeout(timer); resolve(true); };
        job.readyWaiters.push(wake);
        if (seconds !== null) {
          timer = setTimeout(() => {
            job.readyWaiters.splice(job.readyWaiters.indexOf(wake), 1);
            resolve(false);
          }, seconds * 1000);
        }
      });
    },
  };
  return job;
}

function enqueueDelivery(job, text) {
  const receipt = randomUUID().replaceAll("-", "");
  job.deliveries.set(receipt, { status: "queued", text });
  return receipt;
}

function nextDelivery(job) {
  return [...job.deliveries.values()].find((item) => item.status === "queued");
}

function backgroundBridges(run, agent, background, childOf) {
  const parent = agent.owner ?? agent.runId;
  function save(job) {
    const record = { parent: job.parent, keep: job.keep, data: job.data, model: job.model,
      status: job.status, value: job.value, messages: [...job.outbox],
      deliveries: Object.fromEntries([...job.deliveries].map(([id, item]) => [id, { ...item }])),
      error: job.error?.message ?? null };
    const pending = job.saveTail.then(() => callback(run, agent.runId, "background_record", {
      key: job.key, record,
    }));
    job.saveTail = pending.catch(() => {});
    return pending;
  }
  function launch(job, question) {
    job.isReady = false;
    job.status = "starting";
    job.error = null;
    run.pendingJobs += 1;
    job.task = (async () => {
      let delivery;
      try {
        if (job.keep) {
          job.sessionId = await callback(run, agent.runId, "session_new", { key: job.key });
        }
        do {
          if (job.token.cancelled) return;
          if (job.keep) {
            delivery = nextDelivery(job);
            if (delivery === undefined) {
              job.status = "idle";
              await save(job);
              continue;
            }
            question = delivery.text;
            delete delivery.text;
            delivery.status = "running";
          }
          job.status = "running";
          await save(job);
          job.value = await childOf(job.data, question, job.token, {
            sessionId: job.sessionId, owner: job.key, model: job.model,
          });
          if (job.token.cancelled) return;
          if (delivery) delivery.status = "done";
          if (job.keep && nextDelivery(job) === undefined && !job.token.cancelled) {
            job.status = "idle";
            await save(job);
          }
        } while (job.keep && nextDelivery(job) !== undefined);
        if (!job.token.cancelled && !job.keep) {
          job.status = "done";
          await save(job);
        }
      } catch (error) {
        if (!job.token.cancelled) {
          job.error = error;
          job.status = "failed";
          if (delivery?.status === "running") delivery.status = "failed";
          try { await save(job); } catch { /* the original error remains visible */ }
        }
      } finally {
        job.task = null;
        job.finish();
        run.pendingJobs -= 1;
        releaseRun(run);
      }
    })();
  }
  const owned = (key) => {
    const job = background.jobs.get(key);
    if (!job) throw new EngineError("KeyError", String(key));
    if (job.parent !== parent) throw new EngineError("ValueError", "job does not belong to this agent");
    return job;
  };
  for (const job of background.jobs.values()) {
    if (job.parent === parent && job.keep && !job.token.cancelled && job.task === null && nextDelivery(job)) {
      launch(job);
    }
  }
  return {
    _spawn: named(["task", "data", "keep", "role", "model"], async ({ task, data = null, keep = false, role = null, model = null }) => {
      let question = task;
      if (role) question = `${await callback(run, agent.runId, "role", { role })}\n\n${task}`;
      if (background.jobs.size >= background.maxAgents) {
        throw new EngineError("RuntimeError", "background agent limit reached");
      }
      const key = randomUUID().replaceAll("-", "");
      const job = newJob(key, parent, keep, data, model);
      background.jobs.set(key, job);
      if (keep) enqueueDelivery(job, question);
      launch(job, question);
      return key;
    }),
    _result: named(["key", "timeout"], async ({ key, timeout = null }) => {
      const job = owned(key);
      if (!(await agent.token.race(job.whenReady(timeout)))) {
        throw new EngineError("TimeoutError", "agent still running");
      }
      if (job.status === "cancelled") throw new EngineError("RuntimeError", "agent cancelled");
      if (job.error) throw job.error;
      return job.value;
    }),
    _status: named(["key"], async ({ key }) => owned(key).status),
    _send: named(["key", "text"], async ({ key, text }) => {
      const job = owned(key);
      if (["done", "cancelled"].includes(job.status) || (job.status === "failed" && !job.keep)) {
        throw new EngineError("RuntimeError", "agent already finished");
      }
      if (!job.keep) {
        if (job.task === null) launch(job, text);
        else job.inbox.push(text);
        return null;
      }
      const receipt = enqueueDelivery(job, String(text));
      try { await save(job); } catch (error) {
        job.deliveries.delete(receipt);
        throw error;
      }
      if (job.task === null) launch(job);
      return receipt;
    }),
    _delivery: named(["key", "receipt"], async ({ key, receipt }) =>
      owned(key).deliveries.get(receipt)?.status ?? null),
    _cancel: named(["key"], async ({ key }) => {
      const jobs = [owned(key)];
      for (const job of jobs) {
        jobs.push(...[...background.jobs.values()].filter((child) => child.parent === job.key));
        job.token.cancel("agent cancelled");
        job.status = "cancelled";
        if (job.sessionId !== null) closeSession(job.sessionId);
      }
      for (const job of jobs) {
        await save(job);
        await job.task;
        job.finish();
      }
      return null;
    }),
    _messages: named(["key"], async ({ key }) => {
      const job = owned(key);
      const items = job.outbox.splice(0);
      try { await save(job); } catch (error) { job.outbox.unshift(...items); throw error; }
      return items;
    }),
    _children: named([], async () => [...background.jobs]
      .filter(([, job]) => job.parent === parent && job.status !== "cancelled").map(([key]) => key)),
    tell_parent: named(["text"], async ({ text }) => {
      const key = background.owners.get(agent.runId);
      if (key === undefined) throw new EngineError("RuntimeError", "no background parent mailbox");
      const job = background.jobs.get(key);
      job.outbox.push(text);
      await save(job);
      return null;
    }),
  };
}

// Positional and keyword arguments, sorted into one object the way a Python
// signature would, including its complaint about a name it does not have.
function named(parameters, work) {
  const bound = (args, kwargs) => {
    if (args.length > parameters.length) {
      throw new EngineError("TypeError", `takes ${parameters.length} positional arguments but ${args.length} were given`);
    }
    const values = {};
    args.forEach((value, index) => { values[parameters[index]] = value ?? undefined; });
    for (const [name, value] of Object.entries(kwargs)) {
      if (!parameters.includes(name)) throw new EngineError("TypeError", `got an unexpected keyword argument '${name}'`);
      values[name] = value ?? undefined;
    }
    return work(values);
  };
  return bound;
}

// --- The loop ----------------------------------------------------------------

function seconds(value) {
  return Number.isInteger(value) ? `${value}.0` : String(value);
}

function boundedOutput(text, limit) {
  if (!(limit > 0)) return text;
  const data = Buffer.from(text);
  if (data.length <= limit) return text;
  const notice = `\n[output truncated; ${data.length} bytes produced]\n`;
  const room = limit - Buffer.byteLength(notice);
  const boundary = (index) => {
    while (index > 0 && index < data.length && (data[index] & 0xc0) === 0x80) index -= 1;
    return index;
  };
  if (room <= 0) return data.subarray(0, boundary(limit)).toString();
  const head = boundary(Math.floor(room * 2 / 3));
  let tailStart = data.length - (room - head);
  while (tailStart < data.length && (data[tailStart] & 0xc0) === 0x80) tailStart += 1;
  return data.subarray(0, head).toString() + notice + data.subarray(tailStart).toString();
}

// A cell, bounded the way the other engine bounds one: quiet for too long with
// no helper call outstanding means hung, and waiting on a sub-agent does not.
function runBounded(run, agent, box, work) {
  if (agent.token.cancelled) {
    return Promise.reject(new EngineError("Abandoned", agent.token.reason));
  }
  return agent.token.race(new Promise((resolve, reject) => {
    box.quietSince = performance.now();
    const watch = setInterval(() => {
      if (box.inFlight > 0) { box.quietSince = performance.now(); return; }
      if (performance.now() - box.quietSince > run.config.timeout * 1000) {
        clearInterval(watch);
        box.hung = true;
        box.sandbox?.close();
        reject(new EngineError("CellTimeout", `no response within ${seconds(run.config.timeout)}s`));
      }
    }, Math.min(POLL_MS, run.config.timeout * 1000));
    work().then(
      (value) => { clearInterval(watch); resolve(value); },
      (error) => { clearInterval(watch); reject(error); },
    );
  }));
}

function grant(tools, asked, config) {
  if (asked === null || asked === undefined) {
    return config.inherit_tools ? tools : [];
  }
  const names = tools.map((tool) => tool.name);
  const unknown = asked.filter((name) => !names.includes(name));
  if (unknown.length) {
    const shown = unknown.map((name) => `'${name}'`).join(", ");
    const known = [...names].sort().map((name) => `'${name}'`).join(", ");
    throw new EngineError("RuntimeError", `cannot grant [${shown}]: this agent has no such tool. Available: [${known}]`);
  }
  return tools.filter((tool) => asked.includes(tool.name));
}

function pythonType(value) {
  if (value === null) return "NoneType";
  if (typeof value === "boolean") return "bool";
  if (typeof value === "number") return Number.isInteger(value) ? "int" : "float";
  return typeof value;
}

async function runAgent(run, spec) {
  const { prompt, instruction, depth, config, tools, parentRunId, sessionId, owner } = spec;
  const budget = run.budget;
  const runId = spec.runId ?? `${run.id}.${run.nextAgent++}`;
  const agent = { runId, token: spec.token, owner };
  const model = depth === 0 ? config.primary_agent : (config.sub_agent || config.primary_agent);
  const canRecurse = config.enable_delegation && depth + 1 <= budget.max_depth;
  const background = run.background;
  const emit = (name, fields) => {
    if (run.traced) send({ type: "event", runId: run.id, agentId: runId, name, fields });
  };
  const session = (op, args = {}) => callback(run, runId, "session", { id: sessionId, op, ...args });
  const hasSession = sessionId !== null && sessionId !== undefined;

  let shape = null;
  if (spec.schema !== null && spec.schema !== undefined && config.enable_structured_output) {
    shape = { schema: spec.schema, text: spec.schemaText ?? await callback(run, runId, "schema_text", { schema: spec.schema }) };
  }
  let given = prompt;
  if (!config.enable_structured_output && typeof given !== "string") given = logicCall("as_text", given);

  let total = emptySpend();
  const messages = [];

  const confirm = async (question) => {
    budget.reserve();
    const asked = [...messages, { role: "user", content: question }];
    const { text, usage } = await modelReply(run, runId, asked, model, agent.token);
    budget.settle(usage);
    total = addSpend(total, usage);
    return [logicCall("approves", text), logicCall("gist", text, 300)];
  };

  const permitted = async (pieces) => {
    if (!config.enable_handoff_guard) return;
    const question = logicCall("handoff_question", given, pieces, config.handoff_min_chars, config.handoff_share);
    if (question === null) return;
    const [allowed, reason] = await confirm(question);
    if (!allowed) throw new EngineError("RuntimeError", fill(words.REFUSED_HANDOFF, { reason }));
  };

  const handed = (piece) => {
    if (background && (piece === null || ["boolean", "number"].includes(typeof piece))) return piece;
    if (typeof piece === "string") return piece;
    if (Array.isArray(piece) || (piece && typeof piece === "object")) {
      return config.enable_structured_output ? piece : logicCall("as_text", piece);
    }
    throw new EngineError("TypeError", `a sub-agent is given a string, a dict or a list, not a ${pythonType(piece)}`);
  };

  // The recursion. No message leaves this process to make a child.
  const child = async (piece, childInstruction, token, granted, schema, extra = {}) => {
    const answer = await runAgent(run, {
      prompt: handed(piece),
      instruction: childInstruction ?? null,
      depth: depth + 1,
      config: extra.model ? { ...config, sub_agent: extra.model } : config,
      tools: grant(tools, granted, config),
      parentRunId: runId,
      schema: schema ?? null,
      sessionId: extra.sessionId ?? null,
      owner: extra.owner ?? null,
      lane: extra.lane,
      token,
    });
    total = addSpend(total, answer.usage);
    return answer.output;
  };

  const spread = async (items, work) => {
    if (!items.length) return [];
    const token = new Token(agent.token);
    const width = Math.min(items.length, Math.max(1, config.max_concurrent));
    const slots = new FanoutSlots(width, token);
    const results = new Array(items.length);
    let failure = null;
    await Promise.all(items.map(async (item, index) => {
      const lane = slots.lease(budget);
      try {
        await lane.acquire();
        if (!failure) results[index] = await work(item, token, lane);
      } catch (error) {
        if (!failure || failure.name === "Abandoned") failure = error;
        token.cancel("a sibling failed; this branch was abandoned");
      } finally {
        lane.release();
      }
    }));
    token.detach();
    if (failure) throw failure;
    return results;
  };

  const handlers = {};
  if (config.enable_delegation) {
    handlers.rlm = named(["piece", "instruction", "tools", "schema"], async ({ piece, instruction: asked, tools: granted, schema }) => {
      if (!canRecurse) throw new EngineError("RuntimeError", words.TOO_DEEP);
      await permitted([piece]);
      return child(piece, asked, agent.token, granted, schema);
    });
    handlers.gather_rlm = named(["pieces", "instruction", "tools", "schema"], async ({ pieces, instruction: asked, tools: granted, schema }) => {
      if (!canRecurse) throw new EngineError("RuntimeError", words.TOO_DEEP);
      await permitted(pieces);
      return spread(pieces, (item, token, lane) => child(item, asked, token, granted, schema, { lane }));
    });
  }
  if (background) {
    if (owner !== null && owner !== undefined) background.owners.set(runId, owner);
    Object.assign(handlers, backgroundBridges(run, agent, background, async (piece, asked, token, extra) => {
      if (!canRecurse) throw new EngineError("RuntimeError", words.TOO_DEEP);
      await permitted([piece]);
      return child(piece, asked, token, null, null, extra);
    }));
  }
  for (const name of run.hostFunctions) {
    handlers[name] = (args, kwargs) => callback(run, runId, "host", { name, args, kwargs }, agent.token);
  }

  const toolFacts = tools.map(({ name, description, signature, callable }) => ({ name, description, signature, callable }));
  messages.push({
    role: "system",
    content: logicCall("system_prompt", canRecurse, toolFacts, config.enable_structured_output, shape ? shape.text : null),
  });
  const [promptType, promptSize] = typeof given === "string"
    ? ["str", [...given].length] : logicCall("facts", given);
  emit("run_started", {
    run_id: runId, parent_run_id: parentRunId, depth, model,
    instruction, prompt_type: promptType, prompt_size: promptSize,
  });

  const failed = (error) => {
    emit("run_failed", { run_id: runId, error: describe(error) });
    return error;
  };
  const abortIfCancelled = () => {
    if (agent.token?.cancelled) {
      throw new EngineError("Abandoned", "a sibling failed; this branch was abandoned");
    }
  };
  try {
    abortIfCancelled();
    // Nothing is opened for a child the budget could never serve.
    budget.check();
  } catch (error) {
    throw failed(error);
  }

  const carried = hasSession ? await session("state_for_guest") : null;
  const box = { inFlight: 0, quietSince: 0, hung: false, sandbox: null };
  const callBridge = (name, args, kwargs) => {
    box.inFlight += 1;
    return Promise.resolve()
      .then(() => {
        const handler = handlers[name];
        if (!handler) throw new EngineError("RuntimeError", `no bridge named '${name}'`);
        return handler(args, kwargs);
      })
      .then(
        (value) => {
          if (name === activity?.schedule && value === true) activityScheduled = true;
          return JSON.stringify({ ok: true, value: value ?? null });
        },
        (error) => JSON.stringify({ ok: false, error: describe(error) }),
      )
      .finally(() => {
        box.inFlight -= 1;
        box.quietSince = performance.now();
      });
  };
  const batchOnly = config.enable_batching_guard && "rlm" in handlers ? { rlm: "gather_rlm" } : {};
  const prepared = run.runtimePrepare ? await callback(run, runId, "runtime_prepare", {
    prompt: given, bridges: Object.keys(handlers), sessionId, owner,
  }, agent.token) : {};
  const activity = prepared.activityWait;
  let activityScheduled = false;
  for (const name of prepared.hostFunctions ?? []) {
    if (name === activity?.status) continue;
    handlers[name] = (args, kwargs) => callback(run, runId, "host", { name, args, kwargs }, agent.token);
  }
  const pollActivity = async () => {
    while (true) {
      const status = await callback(run, runId, "host", { name: activity.status }, agent.token);
      if (typeof status?.event === "string") return status;
      const delay = status?.retry_after;
      if (!(typeof delay === "number" && delay > 0 && delay <= 300)) {
        throw new EngineError("RuntimeError", typeof status === "string" ? status : "invalid activity wait status");
      }
      await agent.token.race(sleep(delay * 1000));
    }
  };
  let setup;
  try {
    if (hasSession && activeSessions.has(sessionId)) {
      throw new EngineError("RuntimeError", "session interpreter is already in use");
    }
    if (hasSession) activeSessions.set(sessionId, { box, token: agent.token });
    setup = {
      prompt: given,
      bridges: Object.keys(handlers),
      tools: tools.map(({ name, source }) => ({ name, source })),
      summariser: run.sources.summariser,
      session: carried !== null ? run.sources.session : null,
      restore: carried,
      batch_only: batchOnly,
    };
    if (hasSession && resident.has(sessionId)) {
      box.sandbox = resident.get(sessionId);
      resident.delete(sessionId);
      try {
        await box.sandbox.rebind(setup, callBridge);
      } catch (error) {
        if (error.name !== "RuntimeGone") throw error;
        box.sandbox.close();
        box.sandbox = await openSandbox(setup, callBridge);
      }
    } else box.sandbox = await openSandbox(setup, callBridge);
  } catch (error) {
    if (hasSession && activeSessions.get(sessionId)?.box === box) activeSessions.delete(sessionId);
    box.sandbox?.close();
    throw failed(error instanceof EngineError ? error : new EngineError("RuntimeGone", `runtime failed to start: ${error}`));
  }
  budget.sandboxOpened();
  const executeRaw = (code) => runBounded(run, agent, box, () => box.sandbox.runCell(code));
  const startup = [prepared.startup, background ? run.sources.prelude : null].filter(Boolean);
  const install = async () => {
    for (const code of startup) {
      const installed = await executeRaw(code);
      if (installed.error) throw new EngineError("RuntimeError", installed.error);
    }
  };
  const policy = prepared.cellPolicy;
  const hints = (policy?.hints ?? []).map(([pattern, ignoreCase, hint]) =>
    [new RegExp(pattern, ignoreCase ? "i" : ""), hint]);
  let restarts = 0;
  const execute = async (code) => {
    let cell;
    try {
      cell = await executeRaw(code);
    } catch (error) {
      if (error.name !== "RuntimeGone" || agent.token.cancelled || restarts >= 3) throw error;
      restarts += 1;
      box.sandbox?.close();
      box.sandbox = await openSandbox({ ...setup,
        restore: hasSession ? await session("state_for_guest") : null }, callBridge);
      await install();
      return { stdout: "", final_given: false, error:
        "The kernel stopped and was restarted from its last checkpoint. The interrupted cell was not replayed." };
    }
    if (!policy) return cell;
    cell.stdout = boundedOutput(cell.stdout ?? "", policy.maxOutputBytes);
    if (cell.error) {
      const hint = hints.find(([pattern]) => pattern.test(`${cell.stdout}\n${cell.error}`))?.[1];
      if (hint) cell.error += `\nHint: ${hint}`;
    } else if (code !== null && !code.trim() && policy.emptyHint) {
      cell.stdout = `${cell.stdout}\nHint: ${policy.emptyHint}`.trimStart();
    }
    return cell;
  };

  const snapshot = async (step) => {
    if (!run.watchNamespace) return;
    try {
      const variables = await box.sandbox.snapshot();
      emit("namespace_changed", { run_id: runId, step, variables });
    } catch { /* looking at a run is never part of running one */ }
  };
  const log = (step, code, output, error, usage, timestamps, reasoning = null) => {
    emit("step", {
      step, code, output, error, usage, depth, run_id: runId,
      parent_run_id: parentRunId, timestamps, reasoning,
    });
  };
  const now = () => new Date().toISOString();

  try {
    await install();
    const preamble = hasSession ? await session("preamble") : "";
    if (hasSession) await session("ask", { question: instruction ?? given });

    const keep = async (step, code, ok) => {
      if (!hasSession) return;
      await session("absorb", {
        swept: box.hung ? {} : await box.sandbox.sweep(code),
        cell: { step, ok, code: code ?? "" },
        kept: box.sandbox.restoreFailed,
      });
    };

    // The opening look: step zero, free, and before the model is asked anything.
    {
      const opening = logicCall("opening_code");
      const started = now();
      emit("step_started", { run_id: runId, step: 0, started });
      emit("code_generated", { run_id: runId, step: 0, code: opening });
      const cell = await execute(opening);
      let shown = cell.stdout + (cell.error ? `\n${cell.error}` : "");
      if (shape) shown = fill(words.SCHEMA_SHOWN, { schema: shape.text }) + shown;
      if (hasSession) {
        shown += await session("probe", {
          restore_failed: box.sandbox.restoreFailed,
          taken: tools.map((tool) => tool.name),
        });
      }
      const stamps = { execution_start: started, execution_end: now() };
      emit("output_received", { run_id: runId, step: 0, output: shown, error: Boolean(cell.error) });
      log(0, opening, shown, Boolean(cell.error), emptySpend(), stamps);
      emit("step_completed", {
        run_id: runId, step: 0, usage: emptySpend(), error: Boolean(cell.error), ended: stamps.execution_end,
      });
      await snapshot(0);
      messages.push({
        role: "user",
        content: (preamble || "") + logicCall("opening_message", opening, shown, instruction, config.truncate_len, depth > 0),
      });
    }

    const showsAll = !config.enable_first_look_guard || logicCall("shows_everything", given);

    for (let step = 1; step <= config.max_steps; step += 1) {
      if (activityScheduled) {
        activityScheduled = false;
        emit("child_status", { run_id: runId, status: "waiting" });
        const outcome = await (spec.lane ? spec.lane.park(pollActivity) : pollActivity());
        emit("child_status", { run_id: runId, status: "running" });
        messages.push({ role: "user", content: "Activity wait ended: " + JSON.stringify(outcome) });
      }
      if (background) {
        for (const message of background.drain(runId)) messages.push({ role: "user", content: String(message) });
      }
      abortIfCancelled();
      budget.reserve();
      const callStart = now();
      emit("step_started", { run_id: runId, step, started: callStart });
      let { text, usage, reasoning } = await modelReply(run, runId, [...messages], model, agent.token);
      budget.settle(usage);
      total = addSpend(total, usage);
      if (config.enable_resample && logicCall("runnable_code", text) === null) {
        budget.reserve();
        ({ text, usage, reasoning } = await modelReply(run, runId, [...messages], model, agent.token));
        budget.settle(usage);
        total = addSpend(total, usage);
      }
      const timestamps = { llm_call_start: callStart, llm_call_end: now() };
      messages.push({ role: "assistant", content: text });
      const banner = config.enable_step_banner ? logicCall("budget_banner", step, config.max_steps) : "";
      const code = logicCall("runnable_code", text);

      const refuseUnrun = (notice, shownCode) => {
        emit("output_received", { run_id: runId, step, output: notice, error: true });
        log(step, shownCode, notice, true, usage, timestamps, reasoning);
        emit("step_completed", { run_id: runId, step, usage, error: true, ended: now() });
        messages.push({ role: "user", content: banner + notice });
      };

      if (code === null) {
        emit("code_generated", { run_id: runId, step, code: null });
        refuseUnrun(text.includes(run.nativeCallMark) ? words.NO_TOOLS : words.NO_CODE, null);
        continue;
      }
      emit("code_generated", { run_id: runId, step, code });

      if (config.enable_first_look_guard && step === 1 && !showsAll && logicCall("answers_without_looking", code)) {
        refuseUnrun(fill(words.UNSEEN, { preview: words.PREVIEW }), code);
        continue;
      }

      timestamps.execution_start = now();
      const cell = await execute(code);
      timestamps.execution_end = now();
      const output = cell.stdout + (cell.error ? `\n${cell.error}` : "");

      const held = cell.final_given && config.enable_blind_final_guard && logicCall("writes_answer_blind", code);
      const refused = shape && cell.final_given && !held
        ? await callback(run, runId, "schema_problems", { schema: shape.schema, value: cell.final })
        : [];

      if (cell.final_given && !held && !refused.length) {
        emit("output_received", { run_id: runId, step, output, error: false });
        log(step, code, output, false, usage, timestamps, reasoning);
        emit("step_completed", { run_id: runId, step, usage, error: false, ended: now() });
        await snapshot(step);
        emit("final", { result: cell.final, depth, run_id: runId, parent_run_id: parentRunId });
        emit("run_completed", { run_id: runId, result: cell.final });
        await keep(step, code, !cell.error);
        if (hasSession) {
          await session("settled", { question: instruction ?? given, answer: cell.final, run_id: runId });
        }
        return { output: cell.final, steps: step, usage: total };
      }

      await keep(step, code, !cell.error);
      let labelled = logicCall("label_output", output, config.truncate_len);
      if (held) labelled += `\n\n${words.BLIND}`;
      if (refused.length) {
        labelled += `\n\n${fill(words.SCHEMA_REFUSED, {
          schema: shape.text,
          problems: refused.map((line) => `  - ${line}`).join("\n"),
        })}`;
      }
      const failedStep = Boolean(cell.error) || refused.length > 0;
      emit("output_received", { run_id: runId, step, output: labelled, error: failedStep });
      log(step, code, labelled, failedStep, usage, timestamps, reasoning);
      emit("step_completed", { run_id: runId, step, usage, error: failedStep, ended: now() });
      await snapshot(step);
      messages.push({ role: "user", content: `${banner}Output:\n${labelled}` });
    }
    throw new EngineError("StepsUsedUp", `no FINAL() after ${config.max_steps} steps`);
  } catch (error) {
    throw failed(error);
  } finally {
    budget.sandboxClosed();
    if (hasSession) activeSessions.delete(sessionId);
    if (hasSession && !box.hung && !agent.token.cancelled) resident.set(sessionId, box.sandbox);
    else box.sandbox?.close();
    box.sandbox = null;
  }
}

// --- Runs, and the wire ------------------------------------------------------

const runs = new Map();
const backgrounds = new Map();

function releaseRun(run) {
  if (run.finished && run.pendingJobs === 0 && runs.delete(run.id)) {
    send({ type: "released", runId: run.id });
  }
}

async function start(record) {
  const run = {
    id: record.runId,
    config: record.config,
    budget: new Budget(record.budget, record.allowanceId),
    route: record.route ?? null,
    sources: record.sources,
    runtimePrepare: record.runtimePrepare ?? false,
    tools: record.tools,
    traced: record.traced,
    watchNamespace: record.watchNamespace,
    nativeCallMark: record.nativeCallMark,
    hostFunctions: record.hostFunctions ?? [],
    background: null,
    token: new Token(),
    nextAgent: 1,
    modelCalls: 0,
    pendingJobs: 0,
    finished: false,
  };
  runs.set(run.id, run);
  const callbacksBefore = stats.callbacks;
  try {
    await startLogic(record.sources);
    if (record.backgroundId !== null) {
      if (!backgrounds.has(record.backgroundId)) {
        backgrounds.set(record.backgroundId,
          new BackgroundAgents(record.maxAgents, record.backgroundRecords));
      }
      run.background = backgrounds.get(record.backgroundId);
      if (run.background.budget?.owner === record.allowanceId) {
        run.budget = run.background.budget;
        run.budget.sync(record.budget);
      } else run.background.budget = run.budget;
    }
    const answer = await runAgent(run, {
      prompt: record.prompt,
      instruction: record.instruction,
      depth: 0,
      config: record.config,
      tools: record.tools,
      parentRunId: null,
      schema: record.schema,
      schemaText: record.schemaText,
      sessionId: record.sessionId,
      owner: null,
      token: run.token,
      runId: record.agentId,
    });
    send({
      type: "outcome", runId: run.id, status: "completed", output: answer.output, steps: answer.steps,
      usage: answer.usage, budget: run.budget.snapshot(),
      counts: { callbacks: stats.callbacks - callbacksBefore, model_calls: run.modelCalls },
    });
  } catch (error) {
    send({
      type: "outcome", runId: run.id, status: error.name === "Abandoned" ? "cancelled" : "failed",
      error: { name: error.name, message: error.message, ref: error.ref ?? null },
      budget: run.budget.snapshot(),
      counts: { callbacks: stats.callbacks - callbacksBefore, model_calls: run.modelCalls },
    });
  } finally {
    run.finished = true;
    releaseRun(run);
  }
}

setInterval(() => send({ type: "heartbeat" }), HEARTBEAT_MS);

const lines = readline.createInterface({ input: process.stdin, crlfDelay: Infinity });
lines.on("line", (line) => {
  const record = JSON.parse(line);
  if (record.v !== PROTOCOL) {
    send({ type: "protocol_error", message: `unsupported protocol version ${record.v}` });
    return;
  }
  if (record.type === "start") {
    start(record);
  } else if (record.type === "callback_result") {
    answered(record);
  } else if (record.type === "cancel") {
    const run = runs.get(record.runId);
    if (run) {
      run.budget.cancelled = record.reason === "allowance";
      run.token.cancel(record.message);
    }
  } else if (record.type === "background_closed") {
    backgrounds.get(record.backgroundId)?.close();
    backgrounds.delete(record.backgroundId);
  } else if (record.type === "session_closed") {
    closeSession(record.sessionId);
  } else if (record.type === "shutdown") {
    process.exit(0);
  }
});
process.stdin.on("end", () => process.exit(0));
send({ type: "hello", protocol: PROTOCOL });
