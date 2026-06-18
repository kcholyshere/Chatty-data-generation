"""Synthetic data generation: turn a parsed Schema into rows that respect its constraints."""

from generation.engine import GenerationConfig, generate, regenerate_table

__all__ = ["GenerationConfig", "generate", "regenerate_table"]
