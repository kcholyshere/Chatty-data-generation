"""Wrapper around the Google GenAI SDK for structured (JSON-schema-constrained) generation.

The wrapper exposes one core method, :meth:`LLMClient.generate_structured`, which asks Gemini to
return data matching a Pydantic schema, and wraps each call in best-effort Langfuse tracing.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from typing import Any, TypeVar

from google import genai
from google.genai import types
from pydantic import BaseModel

from config import Settings, get_settings
from schema.models import Table

T = TypeVar("T", bound=BaseModel)


class LLMClient:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._client = self._build_client()
        self._langfuse = self._build_langfuse()

    def _build_client(self) -> genai.Client:
        s = self.settings
        if s.use_vertexai and s.google_cloud_project:
            return genai.Client(
                vertexai=True,
                project=s.google_cloud_project,
                location=s.google_cloud_location,
            )
        if s.gemini_api_key:
            return genai.Client(api_key=s.gemini_api_key)
        # Fall back to ambient credentials / env (e.g. GOOGLE_API_KEY).
        return genai.Client()

    def _build_langfuse(self) -> Any | None:
        if not self.settings.langfuse_enabled:
            return None
        try:
            from langfuse import get_client

            return get_client()
        except Exception:
            return None

    def generate_structured(
        self,
        prompt: str,
        schema: type[T] | list[type[T]],
        *,
        temperature: float = 1.0,
        max_tokens: int | None = None,
        system_instruction: str | None = None,
    ) -> Any:
        """Generate a response constrained to ``schema`` and return the parsed Pydantic object(s)."""
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=schema,
            temperature=temperature,
            max_output_tokens=max_tokens,
            system_instruction=system_instruction,
        )
        with self._trace(prompt, temperature):
            response = self._client.models.generate_content(
                model=self.settings.gemini_model,
                contents=prompt,
                config=config,
            )
        if response.parsed is not None:
            return response.parsed
        # Fall back to manual validation if the SDK did not populate ``parsed``.
        target = schema[0] if isinstance(schema, list) else schema
        return target.model_validate_json(response.text)

    @contextlib.contextmanager
    def _trace(self, prompt: str, temperature: float) -> Iterator[None]:
        if self._langfuse is None:
            yield
            return
        try:
            with self._langfuse.start_as_current_generation(
                name="generate_structured",
                model=self.settings.gemini_model,
                input=prompt,
                model_parameters={"temperature": temperature},
            ):
                yield
        except Exception:
            # Never let observability break generation.
            yield


def make_ddl_fallback(client: LLMClient):
    """Return a callable that asks the LLM to parse one CREATE TABLE statement into a Table.

    Wired into :func:`schema.parser.parse_ddl` as the deterministic parser's fallback.
    """

    def _fallback(statement: str) -> Table | None:
        prompt = (
            "Convert the following SQL CREATE TABLE statement into the structured schema object. "
            "Capture every column, its type, nullability, primary key, unique, default, ENUM values, "
            "CHECK bounds, and any foreign keys.\n\n" + statement
        )
        try:
            return client.generate_structured(prompt, Table, temperature=0.0)
        except Exception:
            return None

    return _fallback
