"""Natural-language querying of a generated dataset (Phase 2).

The model is given two tools and the dataset's schema, then answers questions in natural language:

- ``run_sql`` - runs one read-only SELECT and returns rows for the model to reason over.
- ``plot_chart`` - runs a SELECT and records a chart spec the UI renders locally with plotly.

Safety: each query runs in a PostgreSQL read-only transaction (``postgresql_readonly=True``) so a
malformed prompt can never mutate the data; a single-SELECT guard rejects anything else before it
reaches the database.

A dataset lives in its own PostgreSQL schema (set as the ``search_path``). Relationships are grounded
from the parsed schema stored in ``_meta.datasets`` (``to_sql`` creates plain tables, so the live
database exposes no foreign keys); ``information_schema`` columns are the fallback when it's absent.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Literal

import pandas as pd
import sqlparse
from pydantic import BaseModel
from sqlalchemy import text

from llm.client import LLMClient
from schema.models import Schema
from storage.db import get_engine
from storage.writer import load_schema

# Rows handed back to the model per query. The full result is kept for the UI; this only caps the
# token cost of feeding rows into the conversation.
_MAX_ROWS_TO_MODEL = 50

ChartType = Literal["bar", "line", "scatter", "pie"]
_CHART_TYPES: set[str] = {"bar", "line", "scatter", "pie"}


class ChartSpec(BaseModel):
    """A chart the UI should render from ``data`` using plotly."""

    chart_type: ChartType
    x: str
    y: str | None = None
    title: str = ""


@dataclass
class AnswerResult:
    """Everything one NL turn produced: the model's prose, the table to show, an optional chart."""

    text: str
    table: pd.DataFrame | None = None
    chart: ChartSpec | None = None
    sql: list[str] = field(default_factory=list)


def _json_safe_records(df: pd.DataFrame) -> list[dict]:
    """Records with missing/non-finite values coerced to null so the tool result is valid JSON.

    Pandas NULLs surface as ``float('nan')``, which ``json`` emits as the literal ``NaN`` - invalid
    JSON that the Gemini API rejects (``Invalid JSON payload … Unexpected token``) when it serialises
    the tool's return value. Round-tripping through ``to_json`` maps them to null.
    """
    return json.loads(df.to_json(orient="records", date_format="iso"))


def _is_read_only(sql: str) -> bool:
    """True only if ``sql`` is a single SELECT/WITH statement (no DML/DDL/PRAGMA/ATTACH)."""
    statements = [s for s in sqlparse.parse(sql) if s.token_first(skip_cm=True) is not None]
    if len(statements) != 1:
        return False
    first = statements[0].token_first(skip_cm=True)
    return first.ttype in (sqlparse.tokens.DML, sqlparse.tokens.CTE) and first.normalized.upper() in {
        "SELECT",
        "WITH",
    }


class QueryService:
    def __init__(self, dataset: str, client: LLMClient) -> None:
        self.dataset = dataset
        self.client = client
        self.schema_summary = self._build_schema_summary()

    # --- data access -------------------------------------------------------------------------

    def _run_select(self, sql: str) -> pd.DataFrame:
        """Run one SELECT inside a read-only transaction, scoped to the dataset's schema."""
        if not _is_read_only(sql):
            raise ValueError("Only a single read-only SELECT statement is allowed.")
        engine = get_engine()
        with engine.connect().execution_options(postgresql_readonly=True) as conn:
            conn.execute(text(f'SET search_path TO "{self.dataset}"'))
            return pd.read_sql_query(text(sql), conn)

    # --- schema grounding --------------------------------------------------------------------

    def _build_schema_summary(self) -> str:
        schema = load_schema(self.dataset)
        if schema is not None:
            return self._summarise_schema(schema)
        return self._summarise_information_schema()

    @staticmethod
    def _summarise_schema(schema: Schema) -> str:
        lines: list[str] = []
        for table in schema.tables:
            cols = ", ".join(f"{c.name} {c.base_type}" for c in table.columns)
            lines.append(f"TABLE {table.name} ({cols})")
            if table.primary_key:
                lines.append(f"  PRIMARY KEY: {', '.join(table.primary_key)}")
            for fk in table.foreign_keys:
                src = ", ".join(fk.columns)
                ref = ", ".join(f"{fk.ref_table}.{c}" for c in fk.ref_columns)
                lines.append(f"  FOREIGN KEY: {src} -> {ref}")
        return "\n".join(lines)

    def _summarise_information_schema(self) -> str:
        engine = get_engine()
        with engine.connect() as conn:
            rows = conn.execute(
                text(
                    "SELECT table_name, column_name, data_type FROM information_schema.columns "
                    "WHERE table_schema = :s ORDER BY table_name, ordinal_position"
                ),
                {"s": self.dataset},
            ).all()
        tables: dict[str, list[str]] = {}
        for table_name, column_name, data_type in rows:
            tables.setdefault(table_name, []).append(f"{column_name} {data_type}")
        return "\n".join(f"TABLE {t} ({', '.join(cols)})" for t, cols in tables.items())

    # --- the conversational turn -------------------------------------------------------------

    def ask_stream(
        self, question: str, history: list[tuple[str, str]] | None = None
    ) -> tuple[AnswerResult, Iterator[str]]:
        """Answer ``question`` against the dataset, streaming the text.

        Returns the (initially empty) :class:`AnswerResult` and a generator of text deltas. The tool
        calls run while the generator is consumed, so ``result.table``/``chart``/``sql`` are populated
        by the time it is exhausted; ``result.text`` accumulates the streamed deltas. Optional prior
        ``(role, text)`` turns give the model conversation context.
        """
        result = AnswerResult(text="")

        def run_sql(query: str) -> dict:
            """Run a single read-only SQL SELECT against the dataset and return the rows.

            Args:
                query: One PostgreSQL SELECT (or WITH) statement using the exact table and column names
                    from the schema. INSERT/UPDATE/DELETE/DDL are not permitted.

            Returns:
                A dict with the column names, the rows (capped), and the total row count.
            """
            df = self._run_select(query)
            result.table = df
            result.sql.append(query)
            return {
                "columns": list(df.columns),
                "rows": _json_safe_records(df.head(_MAX_ROWS_TO_MODEL)),
                "row_count": int(len(df)),
            }

        def plot_chart(query: str, chart_type: str, x: str, y: str = "", title: str = "") -> dict:
            """Run a SELECT and record a chart for the UI to render from its result.

            Args:
                query: A read-only SELECT producing the columns to plot.
                chart_type: One of "bar", "line", "scatter", "pie".
                x: Column for the x-axis (category/labels for a pie chart).
                y: Column for the y-axis (values). Optional for some charts.
                title: A short chart title.

            Returns:
                A confirmation dict with the row count plotted.
            """
            df = self._run_select(query)
            kind = chart_type if chart_type in _CHART_TYPES else "bar"
            result.table = df
            result.sql.append(query)
            result.chart = ChartSpec(chart_type=kind, x=x, y=y or None, title=title)
            return {"plotted": True, "chart_type": kind, "row_count": int(len(df))}

        # ``from __future__ import annotations`` stringises these closures' type hints, which breaks the
        # SDK's automatic-function-calling introspection (it calls ``isinstance`` with the unresolved
        # string). Restore real types so the SDK can build the function declarations.
        run_sql.__annotations__ = {"query": str, "return": dict}
        plot_chart.__annotations__ = {
            "query": str,
            "chart_type": str,
            "x": str,
            "y": str,
            "title": str,
            "return": dict,
        }

        contents = self._build_contents(question, history)

        def deltas() -> Iterator[str]:
            for delta in self.client.chat_with_tools_stream(
                contents,
                tools=[run_sql, plot_chart],
                system_instruction=self._system_instruction(),
            ):
                result.text += delta
                yield delta

        return result, deltas()

    def _system_instruction(self) -> str:
        return (
            "You are a precise data analyst for a PostgreSQL dataset. Answer the user's questions by "
            "querying the data with the run_sql tool (PostgreSQL dialect, read-only SELECT only). When a "
            "chart would help, call plot_chart with a SELECT and the columns to plot instead of "
            "describing the chart. Use the exact table and column names below, and always wrap every "
            "table and column name in double quotes (identifiers are case-sensitive in PostgreSQL, e.g. "
            'SELECT COUNT(*) FROM "Employees"). When the user asks to "add", "create", or "fill" a '
            "column, or otherwise transform the data, satisfy it as a derived/computed column in your "
            "SELECT (expressions, CASE, COALESCE, or window functions like row_number()) - this returns "
            "a shaped result and never alters the stored data, so do NOT refuse such requests as "
            "DDL/DML; you are only shaping the query output. If a request is "
            "ambiguous, make a reasonable assumption and state it. After querying, give a concise, "
            "direct natural-language answer grounded in the results.\n\nSCHEMA:\n" + self.schema_summary
        )

    @staticmethod
    def _build_contents(question: str, history: list[tuple[str, str]] | None) -> list[Any]:
        from google.genai import types

        contents: list[Any] = []
        for role, text_part in history or []:
            sdk_role = "model" if role == "assistant" else "user"
            contents.append(types.Content(role=sdk_role, parts=[types.Part(text=text_part)]))
        contents.append(types.Content(role="user", parts=[types.Part(text=question)]))
        return contents
