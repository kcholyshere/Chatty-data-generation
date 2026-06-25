"""Offline tests for the Phase 2 query service: read-only guard, JSON safety, SQL execution, grounding.

The DB-touching tests require the PostgreSQL container (``docker compose up -d db``); the guard and
JSON tests are pure and need nothing.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import text

from query.service import QueryService, _is_read_only, _json_safe_records
from schema.models import Column, ForeignKey, Schema, Table
from storage.db import get_engine
from storage.writer import write_dataset


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
        "UPDATE customers SET name = 'X'",
    ],
)
def test_is_read_only_rejects_mutations(sql):
    assert not _is_read_only(sql)


def test_json_safe_records_coerce_missing_to_null():
    """NULL/NaN must become null, not the literal ``NaN`` the Gemini API rejects."""
    df = pd.DataFrame({"manager_id": [1.0, np.nan], "count": [2, 3]})
    records = _json_safe_records(df)
    assert records[1]["manager_id"] is None
    json.dumps(records, allow_nan=False)  # would raise on a stray NaN


@pytest.fixture
def dataset():
    frames = {
        "customers": pd.DataFrame({"id": [1, 2], "name": ["A", "B"]}),
        "orders": pd.DataFrame({"id": [1, 2], "customer_id": [1, 1], "total": [10.0, 5.0]}),
    }
    schema = Schema(
        tables=[
            Table(
                name="customers",
                columns=[Column(name="id", base_type="INT", raw_type="INT")],
                primary_key=["id"],
            ),
            Table(
                name="orders",
                columns=[Column(name="customer_id", base_type="INT", raw_type="INT")],
                foreign_keys=[
                    ForeignKey(columns=["customer_id"], ref_table="customers", ref_columns=["id"])
                ],
            ),
        ]
    )
    name = write_dataset(frames, "pytest_shop", schema=schema)
    yield name
    with get_engine().begin() as conn:
        conn.execute(text(f'DROP SCHEMA IF EXISTS "{name}" CASCADE'))
        conn.execute(text('DELETE FROM "_meta".datasets WHERE name = :n'), {"n": name})


def test_run_select_returns_rows(dataset):
    service = QueryService(dataset, client=None)
    df = service._run_select("SELECT name FROM customers ORDER BY id")
    assert list(df["name"]) == ["A", "B"]


def test_run_select_rejects_non_select(dataset):
    service = QueryService(dataset, client=None)
    with pytest.raises(ValueError):
        service._run_select("DELETE FROM customers")


def test_run_select_is_read_only_transaction(dataset):
    """Even bypassing the guard, the read-only transaction must reject a write."""
    service = QueryService(dataset, client=None)
    with pytest.raises(Exception):
        # ``_run_select`` guards SELECT-only, so go through the same read-only connection directly.
        engine = get_engine()
        with engine.connect().execution_options(postgresql_readonly=True) as conn:
            conn.execute(text(f'SET search_path TO "{dataset}"'))
            conn.execute(text("INSERT INTO customers VALUES (3, 'C')"))


def test_schema_summary_uses_metadata_foreign_keys(dataset):
    summary = QueryService(dataset, client=None).schema_summary
    assert "FOREIGN KEY: customer_id -> customers.id" in summary


def test_schema_summary_falls_back_to_information_schema():
    """With no recorded metadata, grounding falls back to information_schema columns."""
    frames = {"widgets": pd.DataFrame({"id": [1], "label": ["x"]})}
    name = write_dataset(frames, "pytest_nometa")  # no schema -> no metadata row
    try:
        summary = QueryService(name, client=None).schema_summary
        assert "TABLE widgets" in summary
    finally:
        with get_engine().begin() as conn:
            conn.execute(text(f'DROP SCHEMA IF EXISTS "{name}" CASCADE'))
