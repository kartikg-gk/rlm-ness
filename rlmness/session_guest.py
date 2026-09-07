"""Carry a namespace out of one run and back into the next.

This source is shipped into the runtime the same way the summariser is, and
for the same reason: the values live there, so the work of packing them
happens there and only plain data crosses back.

It is executed into a private dictionary, not into the cell's namespace, so
none of these names are visible to the model and none of them can be swept up
as though they were the model's own work. Only `commit` is bound across, and
only because the model is meant to call it.

What survives a run:

  - values that pickle, as base64
  - functions and classes, as the source that defined them, because a pickled
    function is a reference to a module that will not be there next time
  - the reason anything else was left behind, so the gap is legible rather
    than silent
"""

import ast
import base64
import importlib
import io
import pickle
import tokenize
import types

#: Past this, a value is left behind rather than carried. A namespace holding
#: a corpus is the normal case here, and writing it into the state file on
#: every step would cost more than rebuilding it.
MAX_BYTES = 5_000_000

PREVIEW = 200

#: Names the machinery owns. Sweeping them would save the engine's own
#: furniture as though the model had built it, and restoring them would
#: overwrite the live ones.
RESERVED = frozenset(
    {
        "PROMPT",
        "FINAL",
        "commit",
        "llm",
        "rlm",
        "gather_llm",
        "gather_rlm",
        "__builtins__",
        "__name__",
    }
)


class _State:
    """What this guest remembers between steps of one run.

    `sources` accumulates across steps: a function defined at step one is
    still live at step five, but only step one's code contains its text.

    `blobs` is a pickle cache. Re-pickling an unchanged corpus on every step
    is the sweep's dominant cost, and a value that cannot be mutated cannot
    have changed while it stayed the same object. The object itself is held,
    not its `id`, because ids are recycled and a recycled one would match a
    stale blob.
    """

    def __init__(self):
        self.sources = {}
        self.notes = {}
        self.committed = set()
        self.blobs = {}


_state = _State()

_IMMUTABLE = (str, bytes, int, float, bool, complex, type(None), tuple, frozenset)


def commit(name, note=None):
    """Keep this name whatever its size, and say why it mattered.

    A value over the size limit is dropped by default. This is how the model
    overrides that for the one it built the run around, and the note travels
    with it into the next run as a reminder of what it is.
    """
    _state.committed.add(name)
    if note is not None:
        _state.notes[name] = str(note)
    return name


def _read_code(code):
    """Pull definitions and trailing comments out of one step's source.

    A comment sitting on an assignment is the model's own description of what
    it just built, and it is the only description anyone will ever write.
    """
    if not code:
        return
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            text = ast.get_source_segment(code, node)
            if text:
                _state.sources[node.name] = text
    _read_comments(code)


def _read_comments(code):
    """Attach an inline comment to the name it sits beside."""
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(code).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return
    by_line = {}
    for token in tokens:
        if token.type == tokenize.COMMENT:
            by_line[token.start[0]] = token.string.lstrip("#").strip()
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return
    for node in ast.walk(tree):
        targets = []
        if isinstance(node, ast.Assign):
            targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            if isinstance(node.target, ast.Name):
                targets = [node.target.id]
        comment = by_line.get(getattr(node, "lineno", -1))
        if comment:
            for name in targets:
                _state.notes.setdefault(name, comment)


def _pack(name, value):
    """A value as bytes, or the reason it could not be.

    The cache is consulted only for types that cannot change underneath it.
    Anything mutable is packed again every time, because a list can be
    rewritten in place without ever becoming a different object.
    """
    cached = _state.blobs.get(name)
    if cached is not None and isinstance(value, _IMMUTABLE) and cached[0] is value:
        return cached[1], None
    try:
        blob = pickle.dumps(value, protocol=4)
    except Exception as failure:
        _state.blobs.pop(name, None)
        return None, f"does not pickle: {type(failure).__name__}"
    if len(blob) > MAX_BYTES and name not in _state.committed:
        _state.blobs.pop(name, None)
        return None, f"{len(blob)} bytes, over the {MAX_BYTES} limit"
    if isinstance(value, _IMMUTABLE):
        _state.blobs[name] = (value, blob)
    return blob, None


def _preview(value):
    try:
        text = repr(value)
    except Exception:
        return "<no repr>"
    return text if len(text) <= PREVIEW else text[:PREVIEW] + "..."


def sweep(namespace, code=None, skip=()):
    """Everything worth carrying, as it stands right now.

    A full picture rather than a change list: a name the model deleted should
    disappear from the session, and it can only do that if what is reported is
    what is actually there.
    """
    _read_code(code)
    variables, functions, dropped, modules = {}, {}, {}, {}
    # `skip` is for a runtime whose own machinery shares this namespace rather
    # than sitting in a private one. It hands over the names it already had,
    # so its imports are not saved as though the model had made them.
    owned = set(skip)
    for name, value in list(namespace.items()):
        if name in RESERVED or name in owned or name.startswith("_"):
            continue
        # A module does not pickle, but the import that produced it is one
        # line and can simply be run again. Without this a restored function
        # comes back referring to a name that is no longer bound, which is
        # worse than not restoring it at all: it looks fine until it is
        # called.
        if isinstance(value, types.ModuleType):
            modules[name] = value.__name__
            continue
        if callable(value) or isinstance(value, type):
            source = _state.sources.get(name)
            if source:
                functions[name] = source
            else:
                dropped[name] = "defined outside this session's code"
            continue
        blob, reason = _pack(name, value)
        if blob is None:
            dropped[name] = reason
            continue
        variables[name] = {
            "pickle_b64": base64.b64encode(blob).decode(),
            "type": type(value).__name__,
            "preview": _preview(value),
            "note": _state.notes.get(name),
            "committed": name in _state.committed,
        }
    return {
        "variables": variables,
        "functions": functions,
        "modules": modules,
        "dropped": dropped,
    }


def restore(namespace, state):
    """Rebuild a saved namespace, and report what would not come back.

    Definitions go first: an instance being unpickled may need its class to
    exist already. A name that collides with something the engine owns is
    parked beside it rather than written over it — the live PROMPT of this run
    is not the saved one, and losing it would be worse than losing the saved
    value.
    """
    failed = []
    # Imports first: a definition restored below may close over one, and a
    # value unpickled below may need its module to exist.
    for name, dotted in (state.get("modules") or {}).items():
        try:
            namespace[name] = importlib.import_module(dotted)
        except Exception:
            failed.append(name)
    for name, source in (state.get("functions") or {}).items():
        try:
            exec(source, namespace)
        except Exception:
            failed.append(name)
    for name, meta in (state.get("variables") or {}).items():
        target = f"{name}_saved" if name in RESERVED else name
        try:
            namespace[target] = pickle.loads(base64.b64decode(meta["pickle_b64"]))
        except Exception:
            failed.append(name)
    return failed
