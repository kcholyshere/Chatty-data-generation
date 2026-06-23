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
    google_cloud_location: str = "us-central1"
    use_vertexai: bool = True
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-3.5-flash"

    # --- Langfuse (optional observability) ---
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_host: str | None = None

    @property
    def langfuse_enabled(self) -> bool:
        return bool(self.langfuse_public_key and self.langfuse_secret_key)


def _bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@lru_cache
def get_settings() -> Settings:
    """Load settings once from the environment (after reading ``.env``)."""
    load_dotenv()
    return Settings(
        google_cloud_project=os.getenv("GOOGLE_CLOUD_PROJECT"),
        google_cloud_location=os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1"),
        use_vertexai=_bool(os.getenv("GOOGLE_GENAI_USE_VERTEXAI"), default=True),
        gemini_api_key=os.getenv("GEMINI_API_KEY"),
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-3.5-flash"),
        langfuse_public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
        langfuse_secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
        langfuse_host=os.getenv("LANGFUSE_HOST"),
    )
