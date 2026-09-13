"""Checking an answer against the shape a caller asked for."""

from __future__ import annotations

import json

_PRIMITIVES = {
    str: {"type": "string"},
    int: {"type": "integer"},
    float: {"type": "number"},
    bool: {"type": "boolean"},
    list: {"type": "array"},
    dict: {"type": "object"},
}


class SchemaError(ValueError):
    pass


def as_json_schema(spec) -> dict:
    """A JSON Schema dict from a dict, a plain type, or a pydantic type."""
    if isinstance(spec, dict):
        return spec
    if isinstance(spec, type) and spec in _PRIMITIVES:
        return dict(_PRIMITIVES[spec])
    try:
        from pydantic import BaseModel, TypeAdapter
    except ImportError:
        raise SchemaError(
            "output_schema must be a JSON Schema dict or one of str, int, float, "
            "bool, list, dict. Install pydantic to pass a model or a generic type."
        ) from None
    if isinstance(spec, type) and issubclass(spec, BaseModel):
        return spec.model_json_schema()
    return TypeAdapter(spec).json_schema()


class Shape:
    """A compiled output schema.

    Built before the run starts, so a schema that is itself malformed is the
    caller's error and costs no model call.
    """

    def __init__(self, spec):
        from jsonschema import exceptions, validators

        self.schema = as_json_schema(spec)
        checker = validators.validator_for(self.schema)
        try:
            checker.check_schema(self.schema)
        except exceptions.SchemaError as failure:
            raise SchemaError(f"invalid output schema: {failure.message}") from None
        self._validator = checker(self.schema)
        self.text = json.dumps(self.schema, indent=2)

    def problems(self, value) -> list[str]:
        """Every way `value` misses the schema, one line each, empty when it fits.

        The value is checked as JSON, since JSON is what an answer crosses a
        sandbox as. A runtime that hands back the object itself would otherwise
        pass a tuple where every other runtime passes a list.
        """
        try:
            plain = json.loads(json.dumps(value))
        except (TypeError, ValueError):
            return [
                f"(root): a {type(value).__name__} is not a JSON value. Pass dicts, "
                f"lists, strings, numbers, booleans or None."
            ]
        found = []
        for error in sorted(
            self._validator.iter_errors(plain), key=lambda e: [str(p) for p in e.absolute_path]
        ):
            where = "/" + "/".join(str(part) for part in error.absolute_path)
            found.append(f"{where if error.absolute_path else '(root)'}: {error.message}")
        return found
