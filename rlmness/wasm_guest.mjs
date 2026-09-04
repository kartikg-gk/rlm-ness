// The WASM guest side: one process hosting many sandboxes.
//
// Each sandbox is its own Pyodide interpreter with its own globals, so two
// agents can no more see each other than if they lived in separate processes.
// What they share is the Node runtime around them, which is the part that
// costs 34MB and buys nothing per agent: a second interpreter in this process
// costs about 45MB, against 122MB for a second process holding one.
//
// Every message carries `sid`, naming which sandbox it belongs to. Work is
// chained per sandbox rather than globally, so one agent's slow cell does not
// hold up another's — which is the whole point of a fan-out.

import { loadPyodide } from "pyodide";
import process from "node:process";
import readline from "node:readline";

function send(obj) {
  process.stdout.write(JSON.stringify(obj) + "\n");
}

const SETUP_PY = `
import ast, io, json, contextlib, traceback

PROMPT = json.loads(_PROMPT_JSON)

class _Answered(Exception):
    def __init__(self, value=None):
        self.value = value

def FINAL(value=None):
    raise _Answered(value)

def _make_proxy(name):
    async def proxy(*args, **kwargs):
        raw = await _bridge(
            name, json.dumps(list(args), default=str), json.dumps(kwargs, default=str)
        )
        reply = json.loads(raw)
        if not reply.get("ok"):
            raise RuntimeError(reply.get("error") or f"{name}() failed on the host")
        return reply.get("value")
    return proxy

def _install_tools(specs_json):
    # Defined here, inside the guest, so calling one never leaves WebAssembly.
    # The result is checked against what JSON can carry so a tool behaves the
    # same here as it does under the subprocess runtime.
    for spec in json.loads(specs_json):
        name = spec["name"]
        exec(spec["source"], globals())
        rebuilt = globals()[name]
        # Only a function has a result to check; a data tool is just a value.
        if callable(rebuilt):
            globals()[name] = _checked(name, rebuilt)

def _checked(name, function):
    def tool(*args, **kwargs):
        value = function(*args, **kwargs)
        try:
            json.dumps(value)
        except (TypeError, ValueError):
            raise TypeError(
                f"tool {name!r} returned {type(value).__name__}, which cannot be "
                f"carried as JSON. Return plain data — a string, number, list, "
                f"dict, bool or None."
            ) from None
        return value
    tool.__name__ = getattr(function, "__name__", name)
    tool.__doc__ = function.__doc__
    tool.__wrapped__ = function
    return tool

# Underscored, so the model never sees it listed and never sees the name at
# all in a namespace it did not build.
_summariser = {}

def _install_summariser(source):
    if source:
        exec(source, _summariser)

def _snapshot():
    describe = _summariser.get("summarise")
    if not describe:
        return json.dumps([])
    try:
        # globals() is the cell's namespace here, so this is the real thing
        # and not a copy kept alongside it.
        return json.dumps(describe(globals()), default=str)
    except Exception:
        return json.dumps([])

async def _exec_cell(src):
    buf = io.StringIO()
    final, has_final, error = None, False, None
    try:
        code = compile(src, "<cell>", "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
        with contextlib.redirect_stdout(buf):
            coro = eval(code, globals())
            if coro is not None:
                await coro
    except _Answered as f:
        has_final, final = True, f.value
    except BaseException:
        error = traceback.format_exc()
    return json.dumps(
        {"stdout": buf.getvalue(), "final": final, "has_final": has_final, "error": error},
        default=str,
    )
`;

// sid -> { py, chain, pending, nextId }
const sandboxes = new Map();

async function open(sid, msg) {
  const py = await loadPyodide();
  py.setStdout({ batched: () => {} });
  py.setStderr({ batched: () => {} });

  const box = { py, chain: Promise.resolve(), pending: new Map(), nextId: 1 };
  sandboxes.set(sid, box);

  // Arguments and results cross the FFI as JSON strings: auto-conversion turns
  // containers into proxies whose shape differs by type, and the protocol is
  // JSON on the wire anyway.
  function bridge(name, argsJson, kwargsJson) {
    const id = box.nextId++;
    send({
      sid,
      op: "bridge",
      name,
      args: JSON.parse(argsJson),
      kwargs: JSON.parse(kwargsJson),
      _id: id,
    });
    return new Promise((resolve) => box.pending.set(id, resolve));
  }

  py.globals.set("_PROMPT_JSON", JSON.stringify(msg.prompt ?? ""));
  py.globals.set("_bridge", bridge);
  await py.runPythonAsync(SETUP_PY);
  for (const name of msg.bridges || []) {
    py.globals.set("_bridge_name", name);
    await py.runPythonAsync("globals()[_bridge_name] = _make_proxy(_bridge_name)\n");
  }
  // Only the source crosses, and only now. A later call to a tool is an
  // ordinary Python call that never reaches back out here.
  py.globals.set("_TOOLS_JSON", JSON.stringify(msg.tools ?? []));
  await py.runPythonAsync("_install_tools(_TOOLS_JSON)\n");
  py.globals.set("_SUMMARISER_SRC", msg.summariser ?? "");
  await py.runPythonAsync("_install_summariser(_SUMMARISER_SRC)\n");
  send({ sid, op: "ready" });
}

async function handle(msg) {
  const sid = msg.sid;

  if (msg.op === "init") {
    try {
      await open(sid, msg);
    } catch (error) {
      send({ sid, op: "failed", error: String((error && error.stack) || error) });
    }
    return;
  }

  const box = sandboxes.get(sid);
  if (!box) return;

  if (msg.op === "exec") {
    box.py.globals.set("_CELL_SRC", msg.code ?? "");
    const resultJson = await box.py.runPythonAsync("await _exec_cell(_CELL_SRC)");
    send({ sid, op: "result", ...JSON.parse(resultJson) });
    return;
  }
  if (msg.op === "snapshot") {
    const variablesJson = await box.py.runPythonAsync("_snapshot()");
    send({ sid, op: "namespace", variables: JSON.parse(variablesJson) });
    return;
  }
  if (msg.op === "close") {
    sandboxes.delete(sid);
    send({ sid, op: "closed" });
    return;
  }
}

const lines = readline.createInterface({ input: process.stdin });
lines.on("line", (line) => {
  let msg;
  try {
    msg = JSON.parse(line);
  } catch {
    return;
  }
  if (msg.op === "shutdown") {
    process.exit(0);
  }
  // A bridge result is handed straight to the promise waiting for it, never
  // queued behind an exec — that is what lets an awaited bridge resolve while
  // the cell that called it is still running.
  if (msg.op === "bridge_result") {
    const box = sandboxes.get(msg.sid);
    const resolve = box && box.pending.get(msg._id);
    if (resolve) {
      box.pending.delete(msg._id);
      resolve(JSON.stringify({ ok: msg.ok, value: msg.value, error: msg.error }));
    }
    return;
  }
  if (msg.op === "init") {
    // A new sandbox has no chain yet, and opening it must not sit behind
    // another sandbox's work.
    handle(msg);
    return;
  }
  // Everything else is chained per sandbox: one agent's cells run in order,
  // and a slow cell holds up only its own agent.
  const box = sandboxes.get(msg.sid);
  if (!box) return;
  box.chain = box.chain.then(() => handle(msg)).catch((error) => {
    send({ sid: msg.sid, op: "result", stdout: "", final: null, has_final: false,
           error: String((error && error.stack) || error) });
  });
});

process.stdin.on("end", () => process.exit(0));
