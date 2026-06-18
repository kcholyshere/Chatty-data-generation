# Chatty Data Generation

A conversational AI app with two functions:

1. **Synthetic data generation** — parse a SQL DDL schema and generate valid synthetic data that respects all constraints (especially foreign keys).
2. **Talk to your data** — query the generated data in natural language, with results rendered as text, tables, and plots.

## Tech stack

- **LLM:** Gemini 2.0 Flash (or newer) — streaming, function calling, structured/JSON output.
- **SDK:** Google GenAI SDK (Vertex AI auth via a GCP project).
- **UI:** Streamlit or Gradio.
- **DB:** PostgreSQL.
- **Container:** Docker.
- **Observability:** Langfuse.

## Project layout

```
src/              application source code
data/             generated/working data (raw → interim → processed)
references/       sample schemas and prompting/background material
reports/figures/  generated plots & figures
```

## Getting started

```bash
uv sync --extra dev          # create .venv and install dependencies
cp .env.example .env         # then fill in your credentials
uv run streamlit run src/app.py
```

Run tests with `uv run pytest`.

## Configuration

Copy `.env.example` to `.env` (not tracked) and fill in:

| Key | Purpose |
|---|---|
| `GOOGLE_GENAI_USE_VERTEXAI` | `true` for Vertex AI auth, `false` to use an API key |
| `GOOGLE_CLOUD_PROJECT` / `GOOGLE_CLOUD_LOCATION` | GCP project + region (Vertex auth) |
| `GEMINI_API_KEY` | API key (when not using Vertex) |
| `GEMINI_MODEL` | Model id (default `gemini-2.0-flash`) |
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` / `LANGFUSE_HOST` | Optional observability |
