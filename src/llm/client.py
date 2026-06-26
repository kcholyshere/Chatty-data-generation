"""Wrapper around the Google GenAI SDK for structured (JSON-schema-constrained) generation.

The wrapper exposes one core method, :meth:`LLMClient.generate_structured`, which asks Gemini to
return data matching a Pydantic schema, and wraps each call in best-effort Langfuse tracing.
"""

from __future__ import annotations

import contextlib
import random
import time
from collections.abc import Iterator
from typing import Any, TypeVar, get_args, get_origin

from google import genai
from google.genai import errors, types
from pydantic import BaseModel, TypeAdapter

from config import Settings, get_settings
from schema.models import Table

T = TypeVar("T", bound=BaseModel)


def _parse_response_text(text: str, schema: object) -> Any:
    """Parse JSON text into ``schema``, salvaging complete objects if the array was truncated."""
    try:
        return TypeAdapter(schema).validate_json(text)
    except Exception:
        if get_origin(schema) is list:
            return _salvage_objects(text, get_args(schema)[0])
        raise


def _salvage_objects(text: str, model: type[BaseModel]) -> list:
    """Extract and validate every *complete* top-level JSON object from a (possibly truncated) array."""
    objects: list = []
    depth = 0
    start: int | None = None
    in_string = False
    escaped = False
    for i, ch in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                with contextlib.suppress(Exception):
                    objects.append(model.model_validate_json(text[start : i + 1]))
                start = None
    return objects


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
        disable_thinking: bool = False,
    ) -> Any:
        """Generate a response constrained to ``schema`` and return the parsed Pydantic object(s).

        Thread-safe: the underlying client may be shared across worker threads. Set
        ``disable_thinking`` to skip the model's reasoning step (much faster on flash thinking models).
        """
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=schema,
            temperature=temperature,
            max_output_tokens=max_tokens,
            system_instruction=system_instruction,
            thinking_config=types.ThinkingConfig(thinking_budget=0) if disable_thinking else None,
        )
        with self._trace("generate_structured", prompt, temperature) as generation:
            response = self._call_with_retry(prompt, config)
            self._record_output(generation, response.text or "", response)
        if response.parsed is not None:
            return response.parsed
        # The SDK did not populate ``parsed`` (e.g. the JSON was truncated at ``max_tokens``).
        return _parse_response_text(response.text or "", schema)

    def chat_with_tools_stream(
        self,
        contents: Any,
        tools: list[Any],
        *,
        system_instruction: str | None = None,
        temperature: float = 0.0,
        disable_thinking: bool = True,
        max_rounds: int = 8,
    ) -> Iterator[str]:
        """Stream a tool-using turn, yielding the final answer's text deltas as they arrive.

        ``tools`` are plain Python callables. We run the function-calling loop manually (automatic
        function calling is disabled) so the final answer can stream rather than only being returned
        whole: each round streams the model's parts, executes any function calls it requests (by
        name), feeds the results back, and loops until a round makes no calls. The model's raw parts
        are appended verbatim so Gemini's required ``thought_signature`` on function-call parts is
        preserved (rebuilding the parts strips it and the API rejects the next turn). Bounded by
        ``max_rounds`` so a model that keeps calling tools cannot loop forever. Low temperature by
        default for deterministic tool selection (see ``function_calling.md`` best practices).
        """
        fns = {t.__name__: t for t in tools}
        config = types.GenerateContentConfig(
            tools=tools,
            system_instruction=system_instruction,
            temperature=temperature,
            thinking_config=types.ThinkingConfig(thinking_budget=0) if disable_thinking else None,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        contents = list(contents)
        with self._trace("chat_with_tools_stream", str(contents), temperature) as generation:
            rendered: list[str] = []
            for _ in range(max_rounds):
                model_parts: list[Any] = []
                calls: list[Any] = []
                for chunk in self._stream_with_retry(contents, config):
                    candidate = chunk.candidates[0] if chunk.candidates else None
                    for part in candidate.content.parts if candidate and candidate.content else []:
                        model_parts.append(part)
                        if part.text:
                            rendered.append(part.text)
                            yield part.text
                        if part.function_call is not None:
                            calls.append(part.function_call)
                if not calls:
                    break
                contents.append(types.Content(role="model", parts=model_parts))
                contents.append(
                    types.Content(
                        role="user",
                        parts=[
                            types.Part(
                                function_response=types.FunctionResponse(
                                    name=c.name, response=fns[c.name](**dict(c.args))
                                )
                            )
                            for c in calls
                        ],
                    )
                )
            self._record_output(generation, "".join(rendered))

    def _stream_with_retry(self, contents: Any, config: types.GenerateContentConfig, attempts: int = 4):
        """Yield streamed chunks, retrying transient 429/5xx errors only before the first chunk.

        Once a chunk has been yielded, mid-stream errors propagate so the caller never sees the
        partial text of a failed attempt repeated by a retry.
        """
        for attempt in range(attempts):
            try:
                stream = self._client.models.generate_content_stream(
                    model=self.settings.gemini_model, contents=contents, config=config
                )
                first = next(stream, None)
            except (errors.ClientError, errors.ServerError) as exc:
                transient = isinstance(exc, errors.ServerError) or getattr(exc, "code", None) == 429
                if not transient or attempt == attempts - 1:
                    raise
                time.sleep(min(2**attempt + random.uniform(0, 1), 30))
                continue
            if first is not None:
                yield first
            yield from stream
            return

    def _call_with_retry(self, contents: Any, config: types.GenerateContentConfig, attempts: int = 4):
        """Call the model, retrying transient rate-limit (429) and server (5xx) errors with backoff."""
        for attempt in range(attempts):
            try:
                return self._client.models.generate_content(
                    model=self.settings.gemini_model, contents=contents, config=config
                )
            except (errors.ClientError, errors.ServerError) as exc:
                transient = isinstance(exc, errors.ServerError) or getattr(exc, "code", None) == 429
                if not transient or attempt == attempts - 1:
                    raise
                time.sleep(min(2**attempt + random.uniform(0, 1), 30))

    @contextlib.contextmanager
    def _trace(self, name: str, input_: str, temperature: float) -> Iterator[Any | None]:
        """Open a Langfuse generation span for one model call, yielding it so the caller can record
        the output (``langfuse>=3`` renamed ``start_as_current_generation`` to
        ``start_as_current_observation(as_type="generation")``). A no-op yielding ``None`` when
        Langfuse is unconfigured; any tracing error is swallowed so observability never breaks
        generation."""
        if self._langfuse is None:
            yield None
            return
        try:
            with self._langfuse.start_as_current_observation(
                name=name,
                as_type="generation",
                input=input_,
                model=self.settings.gemini_model,
                model_parameters={"temperature": temperature},
            ) as generation:
                yield generation
        except Exception:
            yield None

    @staticmethod
    def _record_output(generation: Any | None, output: str, response: Any | None = None) -> None:
        """Record the model output (and token usage if available) on a Langfuse generation span."""
        if generation is None:
            return
        usage = getattr(response, "usage_metadata", None)
        usage_details = (
            {"input": usage.prompt_token_count, "output": usage.candidates_token_count}
            if usage is not None
            else None
        )
        with contextlib.suppress(Exception):
            generation.update(output=output, usage_details=usage_details)


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
