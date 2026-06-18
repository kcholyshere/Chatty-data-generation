"""Generation tests: referential integrity (valid input) and the validation backstop (bad input)."""

from __future__ import annotations

import pytest
from conftest import REFERENCES, ScriptedClient, valid_value

from generation.engine import GenerationConfig, generate
from schema.parser import parse_ddl

SCHEMAS = ["restrurants_schema.ddl", "library_mgm_schema.ddl", "company_employee_schema.ddl"]


@pytest.mark.parametrize("filename", SCHEMAS)
def test_referential_integrity(filename):
    schema = parse_ddl((REFERENCES / filename).read_text())
    frames = generate(schema, GenerationConfig(rows_per_table=25, seed=7), ScriptedClient(valid_value))

    pk_values = {
        t.name: set(frames[t.name][t.primary_key[0]]) for t in schema.tables if len(t.primary_key) == 1
    }

    for table in schema.tables:
        df = frames[table.name]
        assert len(df) == 25
        if len(table.primary_key) == 1:
            assert df[table.primary_key[0]].is_unique

        for fk in table.foreign_keys:
            col, ref = fk.columns[0], fk.ref_table
            present = [v for v in df[col].tolist() if v is not None]
            # Every non-null FK value must exist in the parent's primary key set.
            assert all(v in pk_values[ref] for v in present)


# A focused schema that exercises every branch of the validation backstop.
_BAD_DDL = """
CREATE TABLE T (
    id INT PRIMARY KEY AUTO_INCREMENT,
    status ENUM('A', 'B', 'C') NOT NULL,
    score INT CHECK (score >= 1 AND score <= 5),
    code VARCHAR(5) UNIQUE NOT NULL,
    note VARCHAR(3),
    opt_status ENUM('X', 'Y'),
    opt_code VARCHAR(10) UNIQUE
);
"""


def _bad_value(field: str, annotation) -> object:
    # Deliberately invalid: out-of-enum, out-of-range, over-length, duplicate, and NULLs.
    return {
        "status": "ZZZ",       # not a valid ENUM member
        "score": 999,          # exceeds CHECK max of 5
        "code": "DUP",         # identical for every row -> must be de-duplicated
        "note": "toolong",     # exceeds VARCHAR(3)
        "opt_status": None,    # nullable ENUM -> must stay NULL
        "opt_code": None,      # nullable UNIQUE, all NULL -> must not hang or dedupe NULLs
    }[field]


def test_validation_backstop_cleans_bad_content():
    schema = parse_ddl(_BAD_DDL)
    frames = generate(schema, GenerationConfig(rows_per_table=10, seed=1), ScriptedClient(_bad_value))
    df = frames["T"]

    assert len(df) == 10
    assert set(df["status"]) <= {"A", "B", "C"}          # ENUM enforced
    assert df["score"].between(1, 5).all()               # CHECK clamp enforced
    assert (df["note"].str.len() <= 3).all()             # VARCHAR length enforced
    assert df["code"].is_unique                          # UNIQUE enforced via de-dup
    assert (df["code"].str.len() <= 5).all()             # de-dup respects length budget
    assert df["opt_status"].isna().all()                 # nullable ENUM can be NULL
    assert df["opt_code"].isna().all()                   # nullable UNIQUE all-NULL: no hang, stays NULL
