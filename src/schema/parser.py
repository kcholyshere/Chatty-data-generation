"""Hybrid DDL parser: deterministic by default, with an optional LLM fallback per statement.

The deterministic path uses ``sqlparse`` to strip comments and split statements, then targeted regex
to read ``CREATE TABLE`` / ``ALTER TABLE`` into :class:`~schema.models.Schema` metadata. If a
``CREATE TABLE`` statement cannot be parsed deterministically and an ``llm_fallback`` is supplied, it
is asked to produce a :class:`~schema.models.Table` for that statement instead.
"""

from __future__ import annotations

import re
from collections.abc import Callable

import sqlparse

from schema.models import Column, ForeignKey, Schema, Table

# A callable that turns one raw CREATE TABLE statement into a Table (or None if it can't).
LlmFallback = Callable[[str], Table | None]

_IDENT = r"[`\"\[]?(\w+)[`\"\]]?"
_CREATE_RE = re.compile(rf"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?{_IDENT}\s*\(", re.IGNORECASE)
_ALTER_FK_RE = re.compile(
    rf"ALTER\s+TABLE\s+{_IDENT}\s+ADD\s+(?:CONSTRAINT\s+\w+\s+)?"
    rf"FOREIGN\s+KEY\s*\(([^)]+)\)\s+REFERENCES\s+{_IDENT}\s*\(([^)]+)\)",
    re.IGNORECASE,
)
_CONSTRAINT_PREFIXES = ("PRIMARY KEY", "FOREIGN KEY", "UNIQUE", "CHECK", "KEY", "INDEX", "FULLTEXT", "SPATIAL")
_INDEX_PREFIXES = ("KEY", "INDEX", "FULLTEXT", "SPATIAL")


def parse_ddl(ddl: str, llm_fallback: LlmFallback | None = None) -> Schema:
    """Parse a DDL string into a :class:`Schema`."""
    cleaned = sqlparse.format(ddl, strip_comments=True).strip()
    statements = [s.strip() for s in sqlparse.split(cleaned) if s.strip()]

    schema = Schema()
    for stmt in statements:
        if _CREATE_RE.match(stmt):
            table = _try_parse_create(stmt, llm_fallback)
            if table is not None:
                schema.tables.append(table)
        elif stmt.upper().startswith("ALTER"):
            _apply_alter(stmt, schema)
    return schema


def _try_parse_create(stmt: str, llm_fallback: LlmFallback | None) -> Table | None:
    try:
        return _parse_create(stmt)
    except Exception:
        if llm_fallback is not None:
            return llm_fallback(stmt)
        raise


def _parse_create(stmt: str) -> Table:
    match = _CREATE_RE.match(stmt)
    if not match:
        raise ValueError("not a CREATE TABLE statement")
    name = match.group(1)
    body = _extract_parens(stmt, match.end() - 1)

    table = Table(name=name, columns=[])
    for definition in _split_top_level(body):
        _parse_definition(definition.strip(), table)
    if not table.primary_key:
        table.primary_key = [c.name for c in table.columns if c.is_primary_key]
    return table


def _parse_definition(definition: str, table: Table) -> None:
    """Parse one comma-separated entry from a CREATE TABLE body (a column or a table-level constraint)."""
    text = definition
    if text.upper().startswith("CONSTRAINT"):
        text = re.sub(r"^CONSTRAINT\s+" + _IDENT + r"\s+", "", text, flags=re.IGNORECASE)
    upper = text.upper()

    if upper.startswith("PRIMARY KEY"):
        cols = _cols_in_parens(text)
        table.primary_key = cols
        for name in cols:
            if (col := table.column(name)) is not None:
                col.is_primary_key = True
                col.nullable = False
    elif upper.startswith("FOREIGN KEY"):
        fk = _parse_table_fk(text)
        if fk is not None:
            table.foreign_keys.append(fk)
    elif upper.startswith("UNIQUE"):
        for name in _cols_in_parens(text):
            if (col := table.column(name)) is not None:
                col.unique = True
    elif upper.startswith("CHECK"):
        expr = _extract_parens(text, text.upper().index("CHECK") + 5)
        _apply_check_to_columns(expr, table)
    elif upper.startswith(_INDEX_PREFIXES):
        return  # plain index/key — not a data constraint
    else:
        column, inline_fk = _parse_column(text)
        table.columns.append(column)
        if inline_fk is not None:
            table.foreign_keys.append(inline_fk)


def _parse_column(text: str) -> tuple[Column, ForeignKey | None]:
    match = re.match(_IDENT + r"\s+(.*)", text, re.DOTALL)
    if not match:
        raise ValueError(f"cannot parse column: {text!r}")
    name, rest = match.group(1), match.group(2)

    base_type, raw_type, length, precision, scale, enum_values, type_end = _parse_type(rest)
    remainder = rest[type_end:]
    upper = remainder.upper()

    column = Column(
        name=name,
        base_type=base_type,
        raw_type=raw_type,
        length=length,
        precision=precision,
        scale=scale,
        enum_values=enum_values,
        nullable="NOT NULL" not in upper,
        auto_increment=bool(re.search(r"AUTO_?INCREMENT", upper)),
        unique="UNIQUE" in upper,
    )
    if "PRIMARY KEY" in upper:
        column.is_primary_key = True
        column.nullable = False

    if (default_match := re.search(r"DEFAULT\s+('[^']*'|\"[^\"]*\"|\w+\([^)]*\)|[^\s,]+)", remainder, re.IGNORECASE)):
        column.default = default_match.group(1).strip("'\"")

    inline_fk: ForeignKey | None = None
    if (ref := re.search(rf"REFERENCES\s+{_IDENT}\s*\(\s*(\w+)\s*\)", remainder, re.IGNORECASE)):
        inline_fk = ForeignKey(columns=[name], ref_table=ref.group(1), ref_columns=[ref.group(2)])

    if (check := re.search(r"CHECK\s*\(", remainder, re.IGNORECASE)):
        expr = _extract_parens(remainder, check.end() - 1)
        column.check_raw = expr
        column.check_min, column.check_max = _parse_check_bounds(expr)

    return column, inline_fk


def _parse_type(rest: str) -> tuple[str, str, int | None, int | None, int | None, list[str], int]:
    match = re.match(r"\s*(\w+)\s*(\(([^)]*)\))?", rest)
    if not match:
        raise ValueError(f"cannot parse type: {rest!r}")
    base_type = match.group(1).upper()
    params = match.group(3)
    raw_type = match.group(0).strip()

    length = precision = scale = None
    enum_values: list[str] = []
    if params is not None:
        if base_type in {"ENUM", "SET"}:
            enum_values = re.findall(r"'([^']*)'", params)
        elif base_type in {"DECIMAL", "NUMERIC", "FLOAT", "DOUBLE"}:
            nums = [int(p) for p in re.findall(r"\d+", params)]
            precision = nums[0] if nums else None
            scale = nums[1] if len(nums) > 1 else None
        else:
            nums = re.findall(r"\d+", params)
            length = int(nums[0]) if nums else None
    return base_type, raw_type, length, precision, scale, enum_values, match.end()


def _parse_table_fk(text: str) -> ForeignKey | None:
    match = re.match(
        rf"FOREIGN\s+KEY\s*\(([^)]+)\)\s+REFERENCES\s+{_IDENT}\s*\(([^)]+)\)",
        text,
        re.IGNORECASE,
    )
    if not match:
        return None
    return ForeignKey(
        columns=_split_names(match.group(1)),
        ref_table=match.group(2),
        ref_columns=_split_names(match.group(3)),
    )


def _apply_alter(stmt: str, schema: Schema) -> None:
    match = _ALTER_FK_RE.search(stmt)
    if not match:
        return
    table = schema.table(match.group(1))
    if table is None:
        return
    table.foreign_keys.append(
        ForeignKey(
            columns=_split_names(match.group(2)),
            ref_table=match.group(3),
            ref_columns=_split_names(match.group(4)),
        )
    )


def _apply_check_to_columns(expr: str, table: Table) -> None:
    """Attach a table-level CHECK to whichever column it names (best effort)."""
    mn, mx = _parse_check_bounds(expr)
    for col in table.columns:
        if re.search(rf"\b{re.escape(col.name)}\b", expr):
            col.check_raw = expr
            if mn is not None:
                col.check_min = mn
            if mx is not None:
                col.check_max = mx


def _parse_check_bounds(expr: str) -> tuple[float | None, float | None]:
    between = re.search(r"BETWEEN\s+(-?\d+(?:\.\d+)?)\s+AND\s+(-?\d+(?:\.\d+)?)", expr, re.IGNORECASE)
    if between:
        return float(between.group(1)), float(between.group(2))
    mn = mx = None
    for op, raw in re.findall(r"(>=|<=|>|<)\s*(-?\d+(?:\.\d+)?)", expr):
        val = float(raw)
        if op in (">=", ">"):
            mn = val if mn is None else max(mn, val)
        else:
            mx = val if mx is None else min(mx, val)
    return mn, mx


# --- low-level text helpers -------------------------------------------------

def _extract_parens(text: str, open_index: int) -> str:
    """Return the content between the parenthesis at ``open_index`` and its matching close."""
    depth = 0
    in_quote: str | None = None
    for i in range(open_index, len(text)):
        ch = text[i]
        if in_quote:
            if ch == in_quote:
                in_quote = None
            continue
        if ch in "'\"":
            in_quote = ch
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return text[open_index + 1 : i]
    raise ValueError("unbalanced parentheses")


def _split_top_level(body: str) -> list[str]:
    """Split a CREATE TABLE body on commas that are not inside parentheses or quotes."""
    parts: list[str] = []
    depth = 0
    in_quote: str | None = None
    current: list[str] = []
    for ch in body:
        if in_quote:
            current.append(ch)
            if ch == in_quote:
                in_quote = None
            continue
        if ch in "'\"":
            in_quote = ch
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(ch)
    if "".join(current).strip():
        parts.append("".join(current))
    return parts


def _cols_in_parens(text: str) -> list[str]:
    open_index = text.index("(")
    return _split_names(_extract_parens(text, open_index))


def _split_names(text: str) -> list[str]:
    return [n.strip().strip("`\"[]") for n in text.split(",") if n.strip()]
