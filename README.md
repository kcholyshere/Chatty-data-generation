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

> Setup is a work in progress — steps will be filled in as `src/` takes shape.

1. Copy `.env` and fill in your GCP project and credentials.
2. Install dependencies (a `requirements.txt` / `pyproject.toml` will be added).
3. Run the UI.

## Configuration

Environment variables live in `.env` (not tracked). Expected keys will be documented here as they are introduced.
