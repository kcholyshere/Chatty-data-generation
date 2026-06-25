FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

WORKDIR /app

# Install dependencies first (cached) then the project source.
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev

EXPOSE 8501

CMD ["uv", "run", "--no-dev", "streamlit", "run", "src/app.py", \
     "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
