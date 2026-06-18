"""Schema parsing: turn raw DDL text into internal schema metadata objects."""

from schema.models import Column, ForeignKey, Schema, Table
from schema.parser import parse_ddl

__all__ = ["Column", "ForeignKey", "Schema", "Table", "parse_ddl"]
