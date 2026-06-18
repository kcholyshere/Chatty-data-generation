"""Persist generated tables to local files (CSV + Parquet) and a SQLite database.

SQLite is the queryable store the Phase 2 "Talk to your data" tab will read. CSV/Parquet are for
inspection and download. PostgreSQL is deferred (see architectural decisions).
"""

from __future__ import annotations

import io
import sqlite3
import zipfile
from pathlib import Path

import pandas as pd

DEFAULT_ROOT = Path("data/processed")


def write_dataset(
    frames: dict[str, pd.DataFrame],
    schema_name: str,
    root: Path = DEFAULT_ROOT,
) -> dict[str, Path]:
    """Write each table to ``<root>/<schema_name>/<table>.{csv,parquet}`` and load all into a SQLite DB.

    Returns a dict of useful output paths (the directory and the ``.db`` file).
    """
    out_dir = root / schema_name
    out_dir.mkdir(parents=True, exist_ok=True)

    for table, df in frames.items():
        df.to_csv(out_dir / f"{table}.csv", index=False)
        df.to_parquet(out_dir / f"{table}.parquet", index=False)

    db_path = root / f"{schema_name}.db"
    with sqlite3.connect(db_path) as conn:
        for table, df in frames.items():
            df.to_sql(table, conn, if_exists="replace", index=False)

    return {"directory": out_dir, "sqlite": db_path}


def build_csv_zip(frames: dict[str, pd.DataFrame]) -> bytes:
    """Return an in-memory zip of one CSV per table (for a Streamlit download button)."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for table, df in frames.items():
            archive.writestr(f"{table}.csv", df.to_csv(index=False))
    return buffer.getvalue()
