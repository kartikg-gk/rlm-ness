// A worker owns each interpreter; the coordinator keeps recursion and budgets.
import { Worker, isMainThread, parentPort } from "node:worker_threads";
import { loadPyodide, openSandbox as directSandbox } from "./wasm_sandbox.mjs";

export { loadPyodide };

const failure = (message) => Object.assign(new Error(message), { name: "RuntimeGone" });

export async function openSandbox(settings, bridge) {
  const worker = new Worker(new URL(import.meta.url), { stdout: true, stderr: true });
  worker.stdout.resume();
  worker.stderr.resume();
  const pending = new Map();
  let serial = 0;
  let closed = false;
  let currentBridge = bridge;

  function rejectAll(message) {
    closed = true;
    for (const request of pending.values()) request.reject(failure(message));
    pending.clear();
  }

  worker.on("error", (error) => rejectAll(error.message));
  worker.on("exit", () => rejectAll("interpreter worker exited"));
  worker.on("message", async (record) => {
    if (record.type === "bridge") {
      let reply;
      try {
        reply = { ok: true, value: await currentBridge(record.name, record.args, record.kwargs) };
      } catch (error) {
        reply = { ok: false, error: error.message };
      }
      if (!closed) worker.postMessage({ type: "bridge_result", id: record.id, ...reply });
      return;
    }
    const request = pending.get(record.id);
    if (!request) return;
    pending.delete(record.id);
    if (record.ok) request.resolve(record.value);
    else request.reject(failure(record.error));
  });

  function call(op, value) {
    if (closed) return Promise.reject(failure("interpreter worker is closed"));
    const id = ++serial;
    return new Promise((resolve, reject) => {
      pending.set(id, { resolve, reject });
      worker.postMessage({ type: "call", id, op, value });
    });
  }

  try {
    const restored = await call("open", settings);
    return {
      restoreFailed: restored,
      async rebind(next, bridge) {
        currentBridge = bridge;
        await call("rebind", next);
        this.restoreFailed = [];
      },
      runCell: (code) => call("cell", code),
      sweep: (code) => call("sweep", code),
      snapshot: () => call("snapshot", null),
      close() {
        if (!closed) {
          rejectAll("interpreter worker was closed");
          void worker.terminate();
        }
      },
    };
  } catch (error) {
    rejectAll(error.message);
    await worker.terminate();
    throw error;
  }
}

if (!isMainThread) {
  let box;
  let serial = 0;
  let chain = Promise.resolve();
  const waiting = new Map();
  const bridge = (name, args, kwargs) => new Promise((resolve, reject) => {
    const id = ++serial;
    waiting.set(id, { resolve, reject });
    parentPort.postMessage({ type: "bridge", id, name, args, kwargs });
  });

  parentPort.on("message", (record) => {
    if (record.type === "bridge_result") {
      const request = waiting.get(record.id);
      waiting.delete(record.id);
      if (request) {
        if (record.ok) request.resolve(record.value);
        else request.reject(new Error(record.error));
      }
      return;
    }
    chain = chain.then(async () => {
      let value;
      try {
        if (record.op === "open") {
          box = await directSandbox(record.value, bridge);
          value = box.restoreFailed;
        } else if (record.op === "rebind") value = await box.rebind(record.value);
        else if (record.op === "cell") value = await box.runCell(record.value);
        else if (record.op === "sweep") value = await box.sweep(record.value);
        else if (record.op === "snapshot") value = await box.snapshot();
        else throw new Error("unknown worker operation");
        parentPort.postMessage({ type: "result", id: record.id, ok: true, value });
      } catch (error) {
        parentPort.postMessage({ type: "result", id: record.id, ok: false, error: String(error) });
      }
    });
  });
}
