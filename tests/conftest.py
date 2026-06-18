"""Shared test helpers: a scripted stand-in for the LLM client (no network, no credentials).

``ScriptedClient`` mimics ``LLMClient.generate_structured`` but produces rows from a ``value_fn`` you
supply, so tests can drive either valid or deliberately-invalid content.
"""

from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Literal, get_args, get_origin

REFERENCES = Path(__file__).resolve().parents[1] / "references"


class ScriptedClient:
    """Duck-types the content client: returns ``k`` dict rows built from ``value_fn(field, annotation)``."""

    def __init__(self, value_fn):
        self.value_fn = value_fn

    def generate_structured(self, prompt, schema, *, temperature, max_tokens):
        row_model = get_args(schema)[0]
        k = int(re.search(r"Generate (\d+)", prompt).group(1))
        return [
            {name: self.value_fn(name, info.annotation) for name, info in row_model.model_fields.items()}
            for _ in range(k)
        ]


def valid_value(field: str, annotation) -> object:
    """Produce a schema-valid value for a generated field (used by the integrity tests)."""
    args = get_args(annotation)
    non_none = [a for a in args if a is not type(None)] if args else []
    if get_origin(annotation) is Literal:
        return random.choice(list(args))
    for arg in non_none:
        if get_origin(arg) is Literal:
            return random.choice(list(get_args(arg)))
    base = non_none[0] if non_none else (annotation if not args else str)
    if base is int:
        return random.randint(1, 1000)
    if base is float:
        return round(random.uniform(1, 1000), 2)
    if base is bool:
        return random.choice([True, False])
    text = str(base)
    if "datetime" in text:
        return "2024-01-01T10:00:00"
    if "date" in text:
        return "2024-01-01"
    # Random suffix keeps UNIQUE string columns mostly collision-free (the engine dedupes the rest).
    return f"{field}_{random.randint(0, 1_000_000)}"
