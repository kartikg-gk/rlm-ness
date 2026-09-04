"""Describe a REPL namespace without carrying any of it out.

The summary is built where the namespace lives. Only names, type names, sizes
and short reprs cross back — all of them plain strings and integers. Nothing
that could hand the caller a live object goes anywhere, so a runtime that has
no channel for objects does not grow one to be inspectable.

Every runtime runs this same source, so the same variable reads the same way
whichever one is underneath.
"""

PREVIEW = 120

# Bound by the machinery rather than by the model. Listing them buries the two
# or three names that actually say what the agent is doing.
HELPERS = frozenset(
    {
        "PROMPT",
        "FINAL",
        "llm",
        "rlm",
        "gather_llm",
        "gather_rlm",
        "In",
        "Out",
    }
)


def _size(value):
    """Length when the value has one cheaply, else None.

    `len` is called rather than reasoned about, because a type that defines it
    is the only authority on whether it has one. Anything that raises is
    reported as having no size instead of taking the snapshot down.
    """
    try:
        return len(value)
    except Exception:
        return None


def _preview(value, width):
    try:
        text = repr(value)
    except Exception as error:
        return "<unreprable: %s>" % type(error).__name__
    text = " ".join(text.split())
    if len(text) > width:
        return text[: width - 1] + "…"
    return text


def _type_name(value):
    try:
        return type(value).__name__
    except Exception:
        return "?"


def summarise(namespace, width=PREVIEW, include_prompt=True):
    """One row per name the model bound, plus PROMPT.

    PROMPT is kept because it is the whole point of the run and its size is
    the number a reader wants first. The other helpers are dropped: they are
    the same in every namespace and say nothing about this one.
    """
    rows = []
    for name in namespace:
        if name.startswith("_"):
            continue
        if name in HELPERS and not (include_prompt and name == "PROMPT"):
            continue
        try:
            value = namespace[name]
        except Exception:
            continue
        if type(value).__name__ == "module":
            continue
        rows.append(
            {
                "name": name,
                "type": _type_name(value),
                "size": _size(value),
                "preview": _preview(value, width),
            }
        )
    rows.sort(key=lambda row: (row["name"] != "PROMPT", row["name"]))
    return rows
