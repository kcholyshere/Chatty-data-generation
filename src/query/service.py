"""Natural-language querying of a generated dataset (Phase 2).

The model is given two tools and the dataset's schema, then answers questions in natural language:

- ``run_sql`` — runs one read-only SELECT and returns rows for the model to reason over.
- ``plot_chart`` — runs a SELECT and records a chart spec the UI renders locally with plotly.

Safety: the query connection is opened read-only (``mode=ro``) so a malformed prompt can never mutate
the data; a SELECT-only guard rejects anything else before it reaches SQLite (belt and braces).

Relationships are grounded from the schema sidecar JSON written next to the ``.db`` (``df.to_sql``
creates plain tables, so ``PRAGMA`` exposes no foreign keys). When the sidecar is absent we fall back to
column names/types from ``PRAGMA``.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import sqlparse
from pydantic import BaseModel

from llm.client import LLMClient
from schema.models import Schema
from storage.writer import schema_sidecar_path

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
    def __init__(self, sqlite_path: str | Path, client: LLMClient) -> None:
        self.sqlite_path = Path(sqlite_path)
        self.client = client
        self.schema_summary = self._build_schema_summary()

    # --- data access -------------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        """Open the database read-only so no tool call can ever mutate it."""
        return sqlite3.connect(f"file:{self.sqlite_path}?mode=ro", uri=True)

    def _run_select(self, sql: str) -> pd.DataFrame:
        if not _is_read_only(sql):
            raise ValueError("Only a single read-only SELECT statement is allowed.")
        with self._connect() as conn:
            return pd.read_sql_query(sql, conn)

    # --- schema grounding --------------------------------------------------------------------

    def _build_schema_summary(self) -> str:
        sidecar = schema_sidecar_path(self.sqlite_path)
        if sidecar.exists():
            schema = Schema.model_validate_json(sidecar.read_text(encoding="utf-8"))
            return self._summarise_schema(schema)
        return self._summarise_pragma()

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

    def _summarise_pragma(self) -> str:
        with self._connect() as conn:
            tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
            lines: list[str] = []
            for table in tables:
                cols = ", ".join(f"{r[1]} {r[2]}" for r in conn.execute(f"PRAGMA table_info('{table}')"))
                lines.append(f"TABLE {table} ({cols})")
        return "\n".join(lines)

    # --- the conversational turn -------------------------------------------------------------

    def ask(self, question: str, history: list[tuple[str, str]] | None = None) -> AnswerResult:
        """Answer ``question`` against the dataset, optionally with prior ``(role, text)`` turns."""
        result = AnswerResult(text="")

        def run_sql(query: str) -> dict:
            """Run a single read-only SQL SELECT against the dataset and return the rows.

            Args:
                query: One SQLite SELECT (or WITH) statement using the exact table and column names
                    from the schema. INSERT/UPDATE/DELETE/DDL are not permitted.

            Returns:
                A dict with the column names, the rows (capped), and the total row count.
            """
            df = self._run_select(query)
            result.table = df
            result.sql.append(query)
            return {
                "columns": list(df.columns),
                "rows": df.head(_MAX_ROWS_TO_MODEL).to_dict(orient="records"),
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
        response = self.client.chat_with_tools(
            contents,
            tools=[run_sql, plot_chart],
            system_instruction=self._system_instruction(),
        )
        result.text = (response.text or "").strip()
        return result

    def _system_instruction(self) -> str:
        return (
            "You are a precise data analyst for a SQLite dataset. Answer the user's questions by "
            "querying the data with the run_sql tool (SQLite dialect, read-only SELECT only). When a "
            "chart would help, call plot_chart with a SELECT and the columns to plot instead of "
            "describing the chart. Use the exact table and column names below. If a request is "
            "ambiguous, make a reasonable assumption and state it. After querying, give a concise, "
            "direct natural-language answer grounded in the results.\n\nSCHEMA:\n" + self.schema_summary
        )

    @staticmethod
    def _build_contents(question: str, history: list[tuple[str, str]] | None) -> list[Any]:
        from google.genai import types

        contents: list[Any] = []
        for role, text in history or []:
            sdk_role = "model" if role == "assistant" else "user"
            contents.append(types.Content(role=sdk_role, parts=[types.Part(text=text)]))
        contents.append(types.Content(role="user", parts=[types.Part(text=question)]))
        return contents
