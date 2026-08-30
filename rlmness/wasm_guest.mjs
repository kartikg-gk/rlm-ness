// The WASM guest side. Speaks the same protocol as cell_runner.py.

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

async function main() {
  const py = await loadPyodide();
  py.setStdout({ batched: () => {} });
  py.setStderr({ batched: () => {} });

  const pending = new Map();
  let nextId = 1;

  // Arguments and results cross the FFI as JSON strings: auto-conversion turns
  // containers into proxies whose shape differs by type, and the protocol is
  // JSON on the wire anyway.
  function bridge(name, argsJson, kwargsJson) {
    const id = nextId++;
    send({
      op: "bridge",
      name,
      args: JSON.parse(argsJson),
      kwargs: JSON.parse(kwargsJson),
      _id: id,
    });
    return new Promise((resolve) => pending.set(id, resolve));
  }

  async function handle(msg) {
    if (msg.op === "bridge_result") {
      const resolve = pending.get(msg._id);
      if (resolve) {
        pending.delete(msg._id);
        resolve(JSON.stringify({ ok: msg.ok, value: msg.value, error: msg.error }));
      }
      return;
    }
    if (msg.op === "init") {
      py.globals.set("_PROMPT_JSON", JSON.stringify(msg.prompt ?? ""));
      py.globals.set("_bridge", bridge);
      await py.runPythonAsync(SETUP_PY);
      for (const name of msg.bridges || []) {
        py.globals.set("_bridge_name", name);
        await py.runPythonAsync("globals()[_bridge_name] = _make_proxy(_bridge_name)\n");
      }
      send({ op: "ready" });
      return;
    }
    if (msg.op === "exec") {
      py.globals.set("_CELL_SRC", msg.code ?? "");
      const resultJson = await py.runPythonAsync("await _exec_cell(_CELL_SRC)");
      send({ op: "result", ...JSON.parse(resultJson) });
      return;
    }
    if (msg.op === "shutdown") {
      process.exit(0);
    }
  }

  // bridge_result is handled the moment it lands, never queued behind an exec —
  // that is what lets an awaited bridge resolve while a cell is still running.
  const lines = readline.createInterface({ input: process.stdin });
  let chain = Promise.resolve();
  lines.on("line", (line) => {
    let msg;
    try {
      msg = JSON.parse(line);
    } catch {
      return;
    }
    if (msg.op === "bridge_result") {
      handle(msg);
    } else {
      chain = chain.then(() => handle(msg));
    }
  });
}

main().catch((error) => {
  process.stderr.write("worker fatal: " + (error && error.stack ? error.stack : error) + "\n");
  process.exit(1);
});
