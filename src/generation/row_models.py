"""Build a dynamic Pydantic model for the columns the LLM should fill for a given table.

Only *content* columns are included — primary keys, foreign keys and deferred columns are produced
deterministically by the engine, not the LLM.
"""

from __future__ import annotations

import datetime
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, Field, create_model

from schema.models import Column, Table

_NUMERIC_INT = {"INT", "INTEGER", "BIGINT", "SMALLINT", "TINYINT", "MEDIUMINT"}
_NUMERIC_FLOAT = {"DECIMAL", "NUMERIC", "FLOAT", "DOUBLE", "REAL"}
_BOOLEAN = {"BOOLEAN", "BOOL"}


def python_type(column: Column) -> object:
    """Map a column's SQL type to the Python/Pydantic type used for generation."""
    base = column.base_type
    if column.enum_values:
        return Literal[tuple(column.enum_values)]  # type: ignore[return-value]
    if base in _NUMERIC_INT:
        return int
    if base in _NUMERIC_FLOAT:
        return float
    if base in _BOOLEAN:
        return bool
    if base == "DATE":
        return datetime.date
    if base in {"DATETIME", "TIMESTAMP"}:
        return datetime.datetime
    if base == "TIME":
        return datetime.time
    return str


def _field(column: Column) -> tuple[object, object]:
    hints: list[str] = [f"Column '{column.name}' ({column.raw_type})."]
    if column.length:
        hints.append(f"Max length {column.length} characters.")
    if column.check_min is not None or column.check_max is not None:
        hints.append(f"Value between {column.check_min} and {column.check_max}.")
    if column.unique:
        hints.append("Must be unique across rows.")
    description = " ".join(hints)

    py_type = python_type(column)
    if column.nullable:
        return (py_type | None, Field(default=None, description=description))
    return (py_type, Field(description=description))


def build_row_model(table: Table, content_columns: list[Column]) -> type[BaseModel]:
    """Create a Pydantic model whose fields are the given content columns."""
    fields = {col.name: _field(col) for col in content_columns}
    return create_model(f"{table.name}Row", __base__=BaseModel, **fields)  # type: ignore[call-overload]


def coerce_value(column: Column, value: object) -> object:
    """Best-effort coercion of a generated value to the column's Python type."""
    if value is None:
        return None
    py_type = python_type(column)
    try:
        if py_type is int:
            return int(value)
        if py_type is float:
            return float(Decimal(str(value)))
        if py_type is bool:
            if isinstance(value, str):
                return value.strip().lower() in {"1", "true", "yes"}
            return bool(value)
    except (ValueError, TypeError, ArithmeticError):
        return value
    return value
