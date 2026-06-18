"""Parser tests against the three reference schemas (and their tricky bits)."""

from __future__ import annotations

from conftest import REFERENCES

from schema.parser import parse_ddl


def _load(name: str):
    return parse_ddl((REFERENCES / name).read_text())


def test_restaurants_tables_keys_and_constraints():
    schema = _load("restrurants_schema.ddl")
    assert {t.name for t in schema.tables} == {
        "Restaurants", "Customers", "Orders", "Menu", "Order_Items", "Reviews", "Delivery_Drivers",
    }

    orders = schema.table("Orders")
    assert orders.primary_key == ["order_id"]
    assert {fk.ref_table for fk in orders.foreign_keys} == {"Customers", "Restaurants"}

    # ENUM values are extracted in order.
    assert schema.table("Restaurants").column("cuisine_type").enum_values[:2] == ["Italian", "Mexican"]

    # CHECK bounds are parsed (rating BETWEEN-style 1..5).
    rating = schema.table("Reviews").column("rating")
    assert (rating.check_min, rating.check_max) == (1.0, 5.0)

    # UNIQUE + NOT NULL flags on a column.
    email = schema.table("Customers").column("email")
    assert email.unique and not email.nullable


def test_library_alter_table_fk_and_cycle_detection():
    schema = _load("library_mgm_schema.ddl")
    assert len(schema.tables) == 9

    # The trailing ALTER TABLE ... ADD FOREIGN KEY is captured.
    branches = schema.table("Library_Branches")
    assert any(fk.ref_table == "Employees" and "manager_id" in fk.columns for fk in branches.foreign_keys)

    # The Employees <-> Library_Branches <-> Departments cycle is broken via deferred FKs.
    order, deferred = schema.topo_order()
    assert set(order) == {t.name for t in schema.tables}
    assert deferred  # at least one FK had to be deferred

    # Non-deferred dependencies are respected in the ordering.
    position = {name: i for i, name in enumerate(order)}
    for table in schema.tables:
        for fk in table.foreign_keys:
            if (table.name, fk.columns[0]) in deferred or fk.ref_table == table.name:
                continue
            assert position[fk.ref_table] < position[table.name]


def test_company_employee_self_referencing_fks():
    schema = _load("company_employee_schema.ddl")
    reviews = schema.table("Performance_Reviews")
    # Two FKs both point at Employees (employee_id and reviewer_id).
    assert sum(fk.ref_table == "Employees" for fk in reviews.foreign_keys) == 2
    assert reviews.column("rating").check_max == 5.0
