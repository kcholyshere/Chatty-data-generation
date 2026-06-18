"""Internal representation of a parsed SQL schema.

These objects describe the *shape* of the data (tables, columns, keys, constraints) — they hold no
rows. The generation engine reads them to produce the actual data.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ForeignKey(BaseModel):
    """A foreign-key relationship: ``columns`` in this table reference ``ref_columns`` in ``ref_table``."""

    columns: list[str]
    ref_table: str
    ref_columns: list[str]


class Column(BaseModel):
    """A single column and the constraints that apply to it."""

    name: str
    # ``base_type`` is the normalised type family (INT, VARCHAR, DECIMAL, DATE, DATETIME, TEXT,
    # BOOLEAN, ENUM, ...). ``raw_type`` keeps the original text for reference.
    base_type: str
    raw_type: str
    length: int | None = None          # VARCHAR(length)
    precision: int | None = None       # DECIMAL(precision, scale)
    scale: int | None = None
    nullable: bool = True
    is_primary_key: bool = False
    auto_increment: bool = False
    unique: bool = False
    default: str | None = None
    enum_values: list[str] = Field(default_factory=list)
    check_raw: str | None = None       # raw CHECK(...) expression
    check_min: float | None = None     # parsed simple numeric bound, if any
    check_max: float | None = None


class Table(BaseModel):
    """A table: its columns, primary key, and foreign keys."""

    name: str
    columns: list[Column]
    primary_key: list[str] = Field(default_factory=list)
    foreign_keys: list[ForeignKey] = Field(default_factory=list)

    def column(self, name: str) -> Column | None:
        return next((c for c in self.columns if c.name == name), None)

    @property
    def fk_column_names(self) -> set[str]:
        names: set[str] = set()
        for fk in self.foreign_keys:
            names.update(fk.columns)
        return names


class Schema(BaseModel):
    """A whole schema: an ordered collection of tables."""

    tables: list[Table] = Field(default_factory=list)

    def table(self, name: str) -> Table | None:
        return next((t for t in self.tables if t.name == name), None)

    def topo_order(self) -> tuple[list[str], set[tuple[str, str]]]:
        """Order tables so a table comes after the tables it references.

        Returns ``(ordered_table_names, deferred_fks)`` where ``deferred_fks`` is the set of
        ``(table_name, fk_column)`` pairs whose foreign key was broken to resolve a dependency cycle
        (e.g. Employees <-> Departments). Deferred FK columns must be filled as NULL initially and
        may be backfilled once the referenced table exists.
        """
        names = [t.name for t in self.tables]
        deferred: set[tuple[str, str]] = set()

        # Build dependency edges: table -> set of tables it references (ignoring self-references).
        deps: dict[str, set[str]] = {name: set() for name in names}
        for table in self.tables:
            for fk in table.foreign_keys:
                if fk.ref_table in deps and fk.ref_table != table.name:
                    deps[table.name].add(fk.ref_table)

        ordered: list[str] = []
        remaining = set(names)
        while remaining:
            ready = sorted(n for n in remaining if not (deps[n] & remaining))
            if not ready:
                # A cycle remains; break it by deferring one nullable FK edge.
                broke = self._break_cycle(remaining, deps, deferred)
                if not broke:
                    # No nullable FK to defer — break arbitrarily so we still make progress.
                    victim = sorted(remaining)[0]
                    for ref in list(deps[victim] & remaining):
                        deps[victim].discard(ref)
                        for fk in self.table(victim).foreign_keys:
                            if fk.ref_table == ref:
                                for col in fk.columns:
                                    deferred.add((victim, col))
                continue
            for n in ready:
                ordered.append(n)
                remaining.discard(n)
        return ordered, deferred

    def _break_cycle(
        self,
        remaining: set[str],
        deps: dict[str, set[str]],
        deferred: set[tuple[str, str]],
    ) -> bool:
        """Defer the first nullable FK edge found within the remaining cycle. Returns True if one was broken."""
        for name in sorted(remaining):
            table = self.table(name)
            for fk in table.foreign_keys:
                if fk.ref_table not in remaining or fk.ref_table == name:
                    continue
                if all((table.column(c) and table.column(c).nullable) for c in fk.columns):
                    deps[name].discard(fk.ref_table)
                    for col in fk.columns:
                        deferred.add((name, col))
                    return True
        return False
