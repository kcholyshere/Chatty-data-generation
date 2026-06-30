"""Application settings, loaded from environment / ``.env``.

Uses ``python-dotenv`` to read ``.env`` and a plain Pydantic model to validate. Auth follows the Google
GenAI SDK conventions: either Vertex AI (GCP project + location) or a Gemini API key.
"""

from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv
from pydantic import BaseModel


class Settings(BaseModel):
    # --- Gemini / Vertex ---
    google_cloud_project: str | None = None
    google_cloud_location: str = "global"
    use_vertexai: bool = True
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-3.5-flash"

    # --- Langfuse (optional observability) ---
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_host: str | None = None

    # --- PostgreSQL ---
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "chatty"
    postgres_user: str = "chatty"
    postgres_password: str = "chatty"

    @property
    def langfuse_enabled(self) -> bool:
        return bool(self.langfuse_public_key and self.langfuse_secret_key)

    @property
    def database_url(self) -> str:
        """SQLAlchemy URL for the psycopg (v3) driver."""
        return (
            f"postgresql+psycopg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


def _bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@lru_cache
def get_settings() -> Settings:
    """Load settings once from the environment (after reading ``.env``)."""
    # ``override=True`` so the project's ``.env`` is authoritative: a stale value left in the shell
    # (e.g. from a previously sourced ``.env``) must not silently shadow the file.
    load_dotenv(override=True)
    return Settings(
        google_cloud_project=os.getenv("GOOGLE_CLOUD_PROJECT"),
        google_cloud_location=os.getenv("GOOGLE_CLOUD_LOCATION", "global"),
        use_vertexai=_bool(os.getenv("GOOGLE_GENAI_USE_VERTEXAI"), default=True),
        gemini_api_key=os.getenv("GEMINI_API_KEY"),
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-3.5-flash"),
        langfuse_public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
        langfuse_secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
        langfuse_host=os.getenv("LANGFUSE_HOST"),
        postgres_host=os.getenv("POSTGRES_HOST", "localhost"),
        postgres_port=int(os.getenv("POSTGRES_PORT", "5432")),
        postgres_db=os.getenv("POSTGRES_DB", "chatty"),
        postgres_user=os.getenv("POSTGRES_USER", "chatty"),
        postgres_password=os.getenv("POSTGRES_PASSWORD", "chatty"),
    )
