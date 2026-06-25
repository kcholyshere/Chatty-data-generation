"""Persist generated tables to PostgreSQL (plus CSV/Parquet for inspection and download).

Each dataset is written into its own PostgreSQL schema (namespace) named after the upload. The parsed
SQL schema (columns/PKs/FKs) is stored in a ``_meta.datasets`` table so the Phase 2 "Talk to your data"
tab can ground JOINs correctly - ``to_sql`` creates plain tables with no FK constraints, so the live
database can't recover relationships on its own.
"""

from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path

import pandas as pd
from sqlalchemy import text

from schema.models import Schema
from storage.db import get_engine

DEFAULT_ROOT = Path("data/processed")
_META_SCHEMA = "_meta"


def safe_identifier(name: str) -> str:
    """Sanitise a dataset name into a safe lowercase SQL identifier ([a-z0-9_], not starting a digit)."""
    ident = re.sub(r"\W", "_", name.strip().lower())
    if not ident or ident[0].isdigit():
        ident = f"d_{ident}"
    return ident[:63]  # PostgreSQL identifier length limit


def write_dataset(
    frames: dict[str, pd.DataFrame],
    schema_name: str,
    root: Path = DEFAULT_ROOT,
    schema: Schema | None = None,
) -> str:
    """Write each table into PostgreSQL schema ``<dataset>`` (and CSV/Parquet under ``root``).

    Returns the sanitised dataset name (the PostgreSQL schema it was written to).
    """
    dataset = safe_identifier(schema_name)

    out_dir = root / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    for table, df in frames.items():
        df.to_csv(out_dir / f"{table}.csv", index=False)
        df.to_parquet(out_dir / f"{table}.parquet", index=False)

    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(text(f'DROP SCHEMA IF EXISTS "{dataset}" CASCADE'))
        conn.execute(text(f'CREATE SCHEMA "{dataset}"'))
    for table, df in frames.items():
        df.to_sql(table, engine, schema=dataset, if_exists="replace", index=False)

    if schema is not None:
        _save_metadata(dataset, schema)
    return dataset


def _save_metadata(dataset: str, schema: Schema) -> None:
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{_META_SCHEMA}"'))
        conn.execute(
            text(
                f'CREATE TABLE IF NOT EXISTS "{_META_SCHEMA}".datasets '
                "(name text PRIMARY KEY, schema_json text NOT NULL, "
                "created_at timestamptz NOT NULL DEFAULT now())"
            )
        )
        conn.execute(
            text(
                f'INSERT INTO "{_META_SCHEMA}".datasets (name, schema_json) VALUES (:n, :j) '
                "ON CONFLICT (name) DO UPDATE SET schema_json = EXCLUDED.schema_json, created_at = now()"
            ),
            {"n": dataset, "j": schema.model_dump_json()},
        )


def list_datasets() -> list[str]:
    """All dataset names, newest first (empty if none generated yet)."""
    engine = get_engine()
    with engine.connect() as conn:
        if conn.execute(text(f"SELECT to_regclass('{_META_SCHEMA}.datasets')")).scalar() is None:
            return []
        rows = conn.execute(
            text(f'SELECT name FROM "{_META_SCHEMA}".datasets ORDER BY created_at DESC')
        )
        return [r[0] for r in rows]


def load_schema(dataset: str) -> Schema | None:
    """Load the parsed schema metadata for a dataset, or None if it wasn't recorded."""
    engine = get_engine()
    with engine.connect() as conn:
        if conn.execute(text(f"SELECT to_regclass('{_META_SCHEMA}.datasets')")).scalar() is None:
            return None
        row = conn.execute(
            text(f'SELECT schema_json FROM "{_META_SCHEMA}".datasets WHERE name = :n'),
            {"n": dataset},
        ).first()
    return Schema.model_validate_json(row[0]) if row else None


def build_csv_zip(frames: dict[str, pd.DataFrame]) -> bytes:
    """Return an in-memory zip of one CSV per table (for a Streamlit download button)."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for table, df in frames.items():
            archive.writestr(f"{table}.csv", df.to_csv(index=False))
    return buffer.getvalue()
