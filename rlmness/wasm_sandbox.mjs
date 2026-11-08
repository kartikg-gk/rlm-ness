// One sandbox: a Pyodide interpreter with its own globals, and the calls that
// drive it. Whoever opens one says how a helper call leaves it, so the same
// handling serves a guest behind a pipe and an engine in this same process.

import process from "node:process";

// Named by the host, because a bare "pyodide" is looked up beside this file,
// and an installed copy of this file has no node_modules beside it.
const { loadPyodide } = await import(process.env.RLMNESS_PYODIDE || "pyodide");

export { loadPyodide };

const SETUP_PY = `
import ast, io, json, contextlib, sys, traceback

# WebAssembly has no sockets, so the standard library's network calls already
# fail here. What does not fail is the bridge back into JavaScript: the host
# is an ordinary Node process with a network, and Pyodide hands the guest
# 'js' and 'pyodide.http' by default. Either one reaches the internet, which
# makes a runtime that calls itself sealed a liar -- and the briefing tells
# the model in as many words that there is no network.
#
# Shutting the door means putting None in sys.modules: an import of that name
# raises rather than resolving. This closes the documented ways out, which is
# what code written by a model will reach for. It is not a claim that a
# determined escape is impossible -- Pyodide shares one JavaScript context
# with its host and was never built to be a security boundary.
for _shut in ("js", "pyodide_js", "pyodide.http", "pyodide.ffi.wrappers"):
    sys.modules[_shut] = None
del _shut

PROMPT = json.loads(_PROMPT_JSON)

# The cell's own record of whether FINAL was called this turn -- reset before
# each cell runs, read after it finishes rather than raised, so code written
# after FINAL runs the same as any other code and a later FINAL call
# overwrites an earlier one the same way reassigning a variable would.
_outcome = {"final_given": False, "final": None}

def FINAL(value=None):
    _outcome["final_given"] = True
    _outcome["final"] = value

# Helpers whose single call must not be gathered by hand, and the code objects
# that let such a call be recognised. Filled before any helper is made.
_BATCH_ONLY = {}
_DELEGATION_CODES = set()

def _make_proxy(name):
    async def proxy(*args, **kwargs):
        raw = await _bridge(
            name, json.dumps(list(args), default=str), json.dumps(kwargs, default=str)
        )
        reply = json.loads(raw)
        if not reply.get("ok"):
            raise RuntimeError(reply.get("error") or f"{name}() failed on the host")
        return reply.get("value")
    if name not in _BATCH_ONLY:
        return proxy
    # A separate function, so a call to it can be told apart by its code.
    async def delegation(*args, **kwargs):
        return await proxy(*args, **kwargs)
    _DELEGATION_CODES.add(delegation.__code__)
    return delegation

def _install_batch_guard(replacements_json):
    # Each sandbox has an interpreter of its own, so replacing gather here
    # touches this agent and no other.
    import asyncio
    replacements = json.loads(replacements_json) if replacements_json else {}
    _BATCH_ONLY.update(replacements)
    if not replacements:
        return
    helpers = sorted(set(replacements.values()))
    instead = ", ".join(helpers)
    example = " or ".join(f"await {name}([a, b])" for name in helpers)

    def refuse(items, call):
        items = list(items)
        if not any(getattr(item, "cr_code", None) in _DELEGATION_CODES for item in items):
            return
        # None of them will run now, so none should be left behind to
        # warn that it was never awaited.
        for item in items:
            if asyncio.iscoroutine(item):
                item.close()
        raise RuntimeError(
            call + " was given sub-agent calls. Use " + instead
            + " with a list instead: it runs the same calls at the same "
            "time, keeps them inside the limit on how many may run at "
            "once, and stops the rest as soon as one fails. Example: "
            + example
        )

    def guarded(real, call, spread):
        def call_it(*args, **kwargs):
            refuse(args if spread else list(args[0]) if args else (), call)
            return real(*args, **kwargs)
        return call_it

    def one(real, call):
        def call_it(item, *args, **kwargs):
            refuse([item], call)
            return real(item, *args, **kwargs)
        return call_it

    asyncio.gather = guarded(asyncio.gather, "asyncio.gather", True)
    asyncio.wait = guarded(asyncio.wait, "asyncio.wait", False)
    asyncio.as_completed = guarded(asyncio.as_completed, "asyncio.as_completed", False)
    asyncio.create_task = one(asyncio.create_task, "asyncio.create_task")
    asyncio.ensure_future = one(asyncio.ensure_future, "asyncio.ensure_future")
    group = getattr(asyncio, "TaskGroup", None)
    if group is not None:
        real_start = group.create_task

        def start(self, item, *args, **kwargs):
            refuse([item], "TaskGroup.create_task")
            return real_start(self, item, *args, **kwargs)

        group.create_task = start

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

# Same treatment as the summariser, and one exception: \`commit\` is bound into
# the cell's namespace because the model is meant to call it. The rest stays
# out here, so a sweep can never pick up its own machinery.
_session = {}
_session_owned = set()

def _install_session(source, session_state):
    if not source:
        return json.dumps([])
    exec(source, _session)
    # Everything already here belongs to this guest, not to the model. In a
    # runtime with a private namespace for its machinery this is empty; here
    # the machinery lives alongside the cell, so it has to say so.
    _session_owned.update(globals())
    globals()["commit"] = _session["commit"]
    state = json.loads(session_state) if session_state else None
    if not state:
        return json.dumps([])
    # The same set serves both directions: names already here are the
    # caller's, so a saved value parks beside one rather than replacing it.
    return json.dumps(_session["restore"](globals(), state, _session_owned))

def _sweep(code):
    gather = _session.get("sweep")
    if not gather:
        return json.dumps({})
    try:
        return json.dumps(gather(globals(), code, _session_owned), default=str)
    except Exception as failure:
        return json.dumps(
            {"variables": {}, "functions": {}, "modules": {},
             "dropped": {"*": f"the sweep failed: {failure}"}}
        )

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

async def _run_cell(src):
    buf = io.StringIO()
    _outcome["final_given"] = False
    _outcome["final"] = None
    error = None
    try:
        code = compile(src, "<cell>", "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
        with contextlib.redirect_stdout(buf):
            coro = eval(code, globals())
            if coro is not None:
                await coro
    except BaseException:
        error = traceback.format_exc()
    return json.dumps(
        {
            "stdout": buf.getvalue(),
            "final": _outcome["final"],
            "final_given": _outcome["final_given"],
            "error": error,
        },
        default=str,
    )
`;

// `callBridge(name, args, kwargs)` answers a helper call with a promise of the
// JSON text {ok, value, error}.
export async function openSandbox(msg, callBridge) {
  const py = await loadPyodide();
  let helperNames = new Set(msg.bridges ?? []);
  py.setStdout({ batched: () => {} });
  py.setStderr({ batched: () => {} });

  // Arguments and results cross the FFI as JSON strings: auto-conversion turns
  // containers into proxies whose shape differs by type, and the protocol is
  // JSON on the wire anyway.
  function bridge(name, argsJson, kwargsJson) {
    return callBridge(name, JSON.parse(argsJson), JSON.parse(kwargsJson));
  }

  py.globals.set("_PROMPT_JSON", JSON.stringify(msg.prompt ?? ""));
  py.globals.set("_bridge", bridge);
  await py.runPythonAsync(SETUP_PY);
  // Before the helpers are made, so each knows whether it is a call that has
  // to go through the gather helper rather than asyncio.gather.
  py.globals.set("_BATCH_ONLY_JSON", JSON.stringify(msg.batch_only ?? {}));
  await py.runPythonAsync("_install_batch_guard(_BATCH_ONLY_JSON)");
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
  py.globals.set("_SESSION_SRC", msg.session ?? "");
  py.globals.set("_SESSION_STATE", JSON.stringify(msg.restore ?? null));
  const failedJson = await py.runPythonAsync(
    "_install_session(_SESSION_SRC, _SESSION_STATE)",
  );

  return {
    restoreFailed: JSON.parse(failedJson),
    async rebind(next) {
      py.globals.set("_PROMPT_JSON", JSON.stringify(next.prompt ?? ""));
      py.globals.set("_REMOVE_HELPERS", JSON.stringify(
        [...helperNames].filter((name) => !(next.bridges ?? []).includes(name))));
      await py.runPythonAsync("PROMPT = json.loads(_PROMPT_JSON)\n"
        + "for _name in json.loads(_REMOVE_HELPERS): globals().pop(_name, None)");
      py.globals.set("_BATCH_ONLY_JSON", JSON.stringify(next.batch_only ?? {}));
      await py.runPythonAsync("_install_batch_guard(_BATCH_ONLY_JSON)");
      for (const name of next.bridges ?? []) {
        py.globals.set("_bridge_name", name);
        await py.runPythonAsync("globals()[_bridge_name] = _make_proxy(_bridge_name)");
      }
      py.globals.set("_TOOLS_JSON", JSON.stringify(next.tools ?? []));
      await py.runPythonAsync("_install_tools(_TOOLS_JSON)");
      helperNames = new Set(next.bridges ?? []);
    },
    async runCell(code) {
      py.globals.set("_CELL_SRC", code ?? "");
      return JSON.parse(await py.runPythonAsync("await _run_cell(_CELL_SRC)"));
    },
    async sweep(code) {
      py.globals.set("_SWEEP_CODE", code ?? "");
      return JSON.parse(await py.runPythonAsync("_sweep(_SWEEP_CODE)"));
    },
    async snapshot() {
      return JSON.parse(await py.runPythonAsync("_snapshot()"));
    },
  };
}
