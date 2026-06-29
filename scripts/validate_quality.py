"""Quantitative validation of generated data: referential integrity and content duplication.

Generates a dataset from a sample DDL (live Gemini) and reports two numbers that back the
project's claims:

- Integrity: orphan FK rows - foreign-key values with no matching parent primary key (should be 0).
- Duplication: mean duplication rate across non-key, non-unique content columns, where a column's
  duplication rate is ``1 - distinct / total`` (lower is more varied).

Run: ``uv run python scripts/validate_quality.py [path/to/schema.ddl] [rows]``
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from generation.engine import GenerationConfig, generate  # noqa: E402
from llm.client import LLMClient  # noqa: E402
from schema.models import Schema  # noqa: E402
from schema.parser import parse_ddl  # noqa: E402


def count_orphan_fk_rows(schema: Schema, frames: dict[str, pd.DataFrame]) -> tuple[int, int]:
    """Return (orphan_rows, total_fk_values_checked) across every single-column FK in the schema."""
    orphans = total = 0
    for table in schema.tables:
        df = frames[table.name]
        for fk in table.foreign_keys:
            if len(fk.columns) != 1:  # composite FKs are a known limitation; report separately if present
                continue
            col, ref_col = fk.columns[0], fk.ref_columns[0]
            parent_keys = set(frames[fk.ref_table][ref_col].dropna())
            values = df[col].dropna()
            total += len(values)
            orphans += int((~values.isin(parent_keys)).sum())
    return orphans, total


def content_duplication(schema: Schema, frames: dict[str, pd.DataFrame]) -> list[tuple[str, float]]:
    """Per non-key, non-unique content column: (name, duplication rate = 1 - distinct/total)."""
    out: list[tuple[str, float]] = []
    for table in schema.tables:
        df = frames[table.name]
        fk_cols = {c for fk in table.foreign_keys for c in fk.columns}
        for col in table.columns:
            if col.is_primary_key or col.unique or col.name in fk_cols:
                continue
            series = df[col.name].dropna()
            if len(series) < 2:
                continue
            rate = 1 - series.nunique() / len(series)
            out.append((f"{table.name}.{col.name}", rate))
    return out


def main() -> None:
    ddl_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("references/library_mgm_schema.ddl")
    rows = int(sys.argv[2]) if len(sys.argv) > 2 else 50

    schema = parse_ddl(ddl_path.read_text())
    print(f"Schema: {ddl_path.name} - {len(schema.tables)} tables, {rows} rows/table")

    config = GenerationConfig(rows_per_table=rows)
    frames = generate(schema, config, LLMClient())

    orphans, total = count_orphan_fk_rows(schema, frames)
    print("\n=== Referential integrity ===")
    print(f"Orphan FK rows: {orphans} / {total} FK values checked "
          f"({0.0 if total == 0 else 100 * orphans / total:.2f}% broken)")

    dup = content_duplication(schema, frames)
    print("\n=== Content duplication (lower = more varied) ===")
    for name, rate in sorted(dup, key=lambda x: -x[1]):
        print(f"  {name:40s} {rate:6.1%}")
    if dup:
        mean = sum(r for _, r in dup) / len(dup)
        print(f"\nMean duplication across {len(dup)} content columns: {mean:.1%}")


if __name__ == "__main__":
    main()
