"""Shared SQLAlchemy engine for the PostgreSQL store.

One engine per process (cached). Pandas uses it to read/write tables; the query service uses it for
read-only SELECTs.
"""

from __future__ import annotations

from functools import lru_cache

from sqlalchemy import Engine, create_engine

from config import get_settings


@lru_cache
def get_engine() -> Engine:
    """Build the process-wide engine from settings (pre-ping so stale connections are recycled)."""
    return create_engine(get_settings().database_url, pool_pre_ping=True)
