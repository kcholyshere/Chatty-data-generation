# Chatty Data Generation

A conversational AI app with two functions:

1. **Synthetic data generation** - parse a SQL DDL schema and generate valid synthetic data that respects all constraints (especially foreign keys).
2. **Talk to your data** - query the generated data in natural language, with results rendered as text, tables, and plots.

<br>

![Data generation tab](assets/ss1.png)

<details>
<summary>Show "Talk to your data" tab screenshot</summary>

<br>

![Talk to your data tab](assets/ss2.png)

</details>

## Tech stack

- **LLM:** Gemini 3.5 Flash (2.0+ supported) - function calling, structured/JSON output.
- **SDK:** Google GenAI SDK (Vertex AI auth via a GCP project).
- **UI:** Streamlit.
- **DB:** PostgreSQL.
- **Container:** Docker.
- **Observability:** Langfuse.

## Project layout

```
src/        application source code
examples/   sample SQL schemas to try
assets/     screenshots used in this README
tests/      test suite
```

## Quick start

Everything (app + PostgreSQL) runs in Docker. Authentication uses **Vertex AI via Application Default Credentials** - no plain API keys. Prerequisites: [Docker](https://www.docker.com/products/docker-desktop), the [`gcloud` CLI](https://cloud.google.com/sdk/docs/install), and access to a GCP project with Vertex AI enabled.

1. Clone and enter the repo.

   ```bash
   git clone <repository-url>
   cd Chatty-data-generation
   ```

2. Authenticate to Google Cloud. This writes the Application Default Credentials that the container mounts.

   ```bash
   gcloud auth application-default login
   ```

3. Configure your environment.

   ```bash
   cp .env.example .env
   ```

   The Vertex AI settings (project, location) live in `docker-compose.yml`. It defaults to project `gd-gcp-gridu-genai`; to use your own, edit `GOOGLE_CLOUD_PROJECT` there. `.env` only carries optional Langfuse keys.

4. Run it.

   ```bash
   docker compose up --build
   ```

   Once it's up, open [http://localhost:8501](http://localhost:8501). Stop with `Ctrl+C`.

> Tip: sample schemas live in `examples/` (`library_mgm.ddl`, `restaurants.ddl`, `company_employee.ddl`). Upload one in the *Data Generation* tab to try it out.

> Note: data refinement is **per-table** - each feedback box regenerates only the selected table. A single global edit across all tables at once (e.g. "replace X with Y everywhere") is not supported; apply such changes one table at a time.

## Configuration

Full reference for the keys in `.env`. The Docker quick start authenticates via Vertex AI (settings baked into `docker-compose.yml`); these keys matter mainly for the non-Docker dev path.

| Key | Purpose |
|---|---|
| `GOOGLE_GENAI_USE_VERTEXAI` | `true` (default) for Google Cloud (Vertex AI) auth; `false` only for the dev-only API-key path |
| `GOOGLE_CLOUD_PROJECT` / `GOOGLE_CLOUD_LOCATION` | GCP project + region for Vertex AI auth |
| `GEMINI_API_KEY` | Gemini API key, used only when `GOOGLE_GENAI_USE_VERTEXAI=false` (local dev only, not spec-conformant) |
| `GEMINI_MODEL` | Model id (default `gemini-3.5-flash`) |
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` / `LANGFUSE_HOST` | Optional observability (leave blank to disable) |

> **Auth:** the project standard is Vertex AI (no plain API keys). Run `gcloud auth application-default login` once so the credentials are available. The plain API-key route (`GOOGLE_GENAI_USE_VERTEXAI=false`) is a local-dev convenience only and does not work with the Docker setup, which forces Vertex AI.

## Running for development (without Docker)

For working on the code directly. Requires [uv](https://docs.astral.sh/uv/) and a local PostgreSQL database.

```bash
uv sync --extra dev          # create .venv and install dependencies
cp .env.example .env         # then fill in your credentials
uv run streamlit run src/app.py
```

Run tests with `uv run pytest`.
