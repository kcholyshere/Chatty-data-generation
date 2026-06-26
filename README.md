# Chatty Data Generation

A conversational AI app with two functions:

1. **Synthetic data generation** — parse a SQL DDL schema and generate valid synthetic data that respects all constraints (especially foreign keys).
2. **Talk to your data** — query the generated data in natural language, with results rendered as text, tables, and plots.

## Tech stack

- **LLM:** Gemini 3.5 Flash (2.0+ supported) — function calling, structured/JSON output.
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

## Quick start (recommended)

This is the easiest way to run the app. You do **not** need to know how to code — just follow the steps in order. Everything runs inside Docker, so you don't have to install Python, a database, or anything else by hand.

**1. Install Docker Desktop**

Download it from [docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop), install it, and open it. Wait until it says it is running (a whale icon appears in your menu bar / system tray).

**2. Download this project**

If you have the project as a `.zip`, unzip it. Otherwise, in a terminal:

```bash
git clone <repository-url>
cd Chatty-data-generation
```

**3. Get a Gemini API key** (one minute, free)

Go to [aistudio.google.com/apikey](https://aistudio.google.com/apikey), sign in with a Google account, and click **Create API key**. Copy the key it gives you.

**4. Add your key to the settings file**

In the project folder, make a copy of `.env.example` and rename the copy to `.env`. Open `.env` in any text editor and set these two lines (leave everything else as-is):

```
GOOGLE_GENAI_USE_VERTEXAI=false
GEMINI_API_KEY=paste-your-key-here
```

**5. Start the app**

In a terminal, from inside the project folder, run:

```bash
docker compose up --build
```

The first run takes a few minutes while it sets things up. When you see a line mentioning `8501`, open your web browser and go to:

```
http://localhost:8501
```

That's it — the app is running. To stop it, press `Ctrl+C` in the terminal.

> **Tip:** sample schemas to try are in the `references/` folder (`library_mgm`, `restaurants`, `company_employee`). Upload one in the *Data Generation* tab to see it work.

## Configuration

All settings live in a `.env` file (copy `.env.example` to `.env`). The quick start above only needs the first two keys; the rest have sensible defaults.

| Key | Purpose |
|---|---|
| `GOOGLE_GENAI_USE_VERTEXAI` | `false` to use a simple Gemini API key, `true` for Google Cloud (Vertex AI) auth |
| `GEMINI_API_KEY` | Your Gemini API key (when `GOOGLE_GENAI_USE_VERTEXAI=false`) |
| `GOOGLE_CLOUD_PROJECT` / `GOOGLE_CLOUD_LOCATION` | GCP project + region (only for Vertex AI auth) |
| `GEMINI_MODEL` | Model id (default `gemini-3.5-flash`) |
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` / `LANGFUSE_HOST` | Optional observability (leave blank to disable) |

> Using Google Cloud / Vertex AI instead of an API key requires the `gcloud` CLI and running `gcloud auth application-default login` first. The API key route above is simpler.

## Running for development (without Docker)

For working on the code directly. Requires [uv](https://docs.astral.sh/uv/) and a local PostgreSQL database.

```bash
uv sync --extra dev          # create .venv and install dependencies
cp .env.example .env         # then fill in your credentials
uv run streamlit run src/app.py
```

Run tests with `uv run pytest`.
