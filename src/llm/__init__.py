"""LLM access: a thin wrapper over the Google GenAI SDK for structured output."""

from llm.client import LLMClient, make_ddl_fallback

__all__ = ["LLMClient", "make_ddl_fallback"]
