"""Offline tests for the Phase 2 query service: read-only guard, SQL execution, schema grounding.

These cover everything that doesn't need the LLM (the tool closures call ``_run_select`` directly).
"""

from __future__ import annotations

import sqlite3

import pandas as pd
import pytest

from query.service import QueryService, _is_read_only
from schema.models import Column, ForeignKey, Schema, Table
from storage.writer import schema_sidecar_path


@pytest.fixture
def db_path(tmp_path):
    path = tmp_path / "shop.db"
    with sqlite3.connect(path) as conn:
        pd.DataFrame({"id": [1, 2], "name": ["A", "B"]}).to_sql("customers", conn, index=False)
        pd.DataFrame({"id": [1, 2], "customer_id": [1, 1], "total": [10.0, 5.0]}).to_sql(
            "orders", conn, index=False
        )
    return path


def test_is_read_only_accepts_select_and_with():
    assert _is_read_only("SELECT * FROM customers")
    assert _is_read_only("WITH c AS (SELECT 1) SELECT * FROM c")


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM customers",
        "DROP TABLE customers",
        "INSERT INTO customers VALUES (3, 'C')",
        "SELECT 1; DROP TABLE customers",  # stacked statements
        "PRAGMA table_info('customers')",
    ],
)
def test_is_read_only_rejects_mutations(sql):
    assert not _is_read_only(sql)


def test_run_select_returns_rows(db_path):
    service = QueryService(db_path, client=None)
    df = service._run_select("SELECT name FROM customers ORDER BY id")
    assert list(df["name"]) == ["A", "B"]


def test_run_select_rejects_non_select(db_path):
    service = QueryService(db_path, client=None)
    with pytest.raises(ValueError):
        service._run_select("DELETE FROM customers")


def test_connection_is_read_only(db_path):
    """Even a guard bypass can't mutate: the connection itself is opened read-only."""
    service = QueryService(db_path, client=None)
    with pytest.raises(sqlite3.OperationalError), service._connect() as conn:
        conn.execute("INSERT INTO customers VALUES (3, 'C')")


def test_schema_summary_uses_sidecar_foreign_keys(db_path):
    schema = Schema(
        tables=[
            Table(name="customers", columns=[Column(name="id", base_type="INT", raw_type="INT")]),
            Table(
                name="orders",
                columns=[Column(name="customer_id", base_type="INT", raw_type="INT")],
                foreign_keys=[
                    ForeignKey(columns=["customer_id"], ref_table="customers", ref_columns=["id"])
                ],
            ),
        ]
    )
    schema_sidecar_path(db_path).write_text(schema.model_dump_json(), encoding="utf-8")
    summary = QueryService(db_path, client=None).schema_summary
    assert "FOREIGN KEY: customer_id -> customers.id" in summary


def test_schema_summary_falls_back_to_pragma(db_path):
    summary = QueryService(db_path, client=None).schema_summary
    assert "TABLE customers" in summary
    assert "TABLE orders" in summary
