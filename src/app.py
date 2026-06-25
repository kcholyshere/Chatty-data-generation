"""Streamlit UI.

Sidebar with two tabs: *Data Generation* (functional) and *Talk to your data* (Phase 2 stub).
Run with: ``uv run streamlit run src/app.py``.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from generation.engine import GenerationConfig, generate, regenerate_table
from query.guardrails import REFUSAL, check_input
from query.service import ChartSpec, QueryService
from schema.parser import parse_ddl
from storage.writer import build_csv_zip, list_datasets, write_dataset

st.set_page_config(page_title="Chatty Data Generation", layout="wide")


def _get_client():
    """Build the LLM client lazily so the app loads without credentials configured."""
    from llm.client import LLMClient

    return LLMClient()


def data_generation_tab() -> None:
    st.header("Data Generation")

    uploaded = st.file_uploader("Upload a DDL schema", type=["sql", "txt", "ddl"])
    prompt = st.text_area("Instructions (optional)", placeholder="e.g. UK-based data, realistic names")

    col_a, col_b, col_c = st.columns(3)
    rows = col_a.number_input("Rows per table", min_value=1, max_value=5000, value=50, step=10)
    temperature = col_b.slider("Temperature", 0.0, 2.0, 1.0, 0.1)
    max_tokens = col_c.number_input("Max output tokens", min_value=256, max_value=32000, value=8192, step=256)

    with st.expander("Speed settings"):
        s1, s2 = st.columns(2)
        concurrency = s1.slider("Parallel requests", 1, 24, 8, help="How many LLM batches run at once.")
        disable_thinking = s2.checkbox("Disable model 'thinking' (faster)", value=True)

    if uploaded is not None and st.button("Generate", type="primary"):
        ddl = uploaded.getvalue().decode("utf-8", errors="replace")
        schema = parse_ddl(ddl)
        if not schema.tables:
            st.error("No tables found in the uploaded DDL.")
            return
        st.success(f"Parsed {len(schema.tables)} tables: " + ", ".join(t.name for t in schema.tables))

        config = GenerationConfig(
            rows_per_table=int(rows),
            temperature=float(temperature),
            max_tokens=int(max_tokens),
            user_prompt=prompt,
            concurrency=int(concurrency),
            disable_thinking=bool(disable_thinking),
        )
        st.session_state["gen_config"] = config
        try:
            with st.spinner("Generating synthetic data…"):
                frames = generate(schema, config, _get_client())
        except Exception as exc:  # noqa: BLE001 — surface any generation/auth error to the user
            st.error(f"Generation failed: {exc}")
            return

        st.session_state["schema"] = schema
        st.session_state["frames"] = frames
        st.session_state["schema_name"] = uploaded.name.rsplit(".", 1)[0]
        st.session_state["dataset"] = write_dataset(
            frames, st.session_state["schema_name"], schema=schema
        )

    _render_results(temperature, max_tokens)


def _render_results(temperature: float, max_tokens: int) -> None:
    frames = st.session_state.get("frames")
    if not frames:
        return

    st.divider()
    st.subheader("Generated data")
    table_name = st.selectbox("Table", list(frames.keys()))
    st.dataframe(frames[table_name], use_container_width=True)

    feedback = st.text_area("Refine this table via feedback", key=f"fb_{table_name}")
    if st.button("Submit refinement") and feedback.strip():
        base = st.session_state.get("gen_config", GenerationConfig())
        config = GenerationConfig(
            rows_per_table=len(frames[table_name]),
            temperature=float(temperature),
            max_tokens=int(max_tokens),
            user_prompt=feedback,
            concurrency=base.concurrency,
            disable_thinking=base.disable_thinking,
        )
        try:
            with st.spinner(f"Refining {table_name}…"):
                updated = regenerate_table(st.session_state["schema"], table_name, frames, config, _get_client())
            frames[table_name] = updated
            write_dataset(frames, st.session_state["schema_name"], schema=st.session_state.get("schema"))
            st.rerun()
        except Exception as exc:  # noqa: BLE001
            st.error(f"Refinement failed: {exc}")

    st.download_button(
        "Download all tables (CSV zip)",
        data=build_csv_zip(frames),
        file_name=f"{st.session_state.get('schema_name', 'data')}.zip",
        mime="application/zip",
    )
    if st.session_state.get("dataset"):
        st.caption(f"Stored in PostgreSQL schema `{st.session_state['dataset']}`")


def _available_datasets() -> list[str]:
    """All generated datasets, newest first; the freshly generated one (if any) leads."""
    datasets = list_datasets()
    current = st.session_state.get("dataset")
    if current and current in datasets:
        datasets.remove(current)
        datasets.insert(0, current)
    return datasets


def _render_chart(df: pd.DataFrame, spec: ChartSpec) -> None:
    import plotly.express as px

    title = spec.title or None
    if spec.chart_type == "pie":
        fig = px.pie(df, names=spec.x, values=spec.y, title=title)
    elif spec.chart_type == "line":
        fig = px.line(df, x=spec.x, y=spec.y, title=title)
    elif spec.chart_type == "scatter":
        fig = px.scatter(df, x=spec.x, y=spec.y, title=title)
    else:
        fig = px.bar(df, x=spec.x, y=spec.y, title=title)
    st.plotly_chart(fig, use_container_width=True)


def _render_turn(entry: dict) -> None:
    with st.chat_message(entry["role"]):
        if entry.get("text"):
            st.markdown(entry["text"])
        table = entry.get("table")
        if table is not None and not table.empty:
            st.dataframe(table, use_container_width=True)
        if entry.get("chart") is not None and table is not None and not table.empty:
            _render_chart(table, entry["chart"])
        for sql in entry.get("sql", []):
            st.caption(f"```sql\n{sql}\n```")


def talk_to_your_data_tab() -> None:
    st.header("Talk to your data")

    datasets = _available_datasets()
    if not datasets:
        st.info("Generate a dataset first — it will appear here for natural-language querying.")
        return

    chosen = st.selectbox("Dataset", datasets)

    history_key = f"chat_{chosen}"
    history: list[dict] = st.session_state.setdefault(history_key, [])

    for entry in history:
        _render_turn(entry)

    question = st.chat_input("Ask a question about your data…", max_chars=500)

    # Starter prompts (schema-agnostic) to kick off a conversation in one click.
    if not history:
        st.caption("Try one:")
        examples = [
            "How many rows are in each table? Show it as a bar chart.",
            "Give me a quick summary of what this dataset contains.",
        ]
        cols = st.columns(len(examples))
        for i, example in enumerate(examples):
            if cols[i].button(example, key=f"example_{i}"):
                question = example

    if not question:
        return

    history.append({"role": "user", "text": question})
    _render_turn(history[-1])

    # Input guardrail: screen for prompt-injection / jailbreak / off-topic before querying.
    if check_input(question, _get_client()).verdict == "unsafe":
        history.append({"role": "assistant", "text": REFUSAL})
        _render_turn(history[-1])
        return

    prior = [(e["role"], e["text"]) for e in history[:-1] if e.get("text")]
    try:
        with st.spinner("Thinking…"):
            service = QueryService(chosen, _get_client())
            answer = service.ask(question, history=prior)
    except Exception as exc:  # noqa: BLE001 — surface any query/auth error to the user
        history.append({"role": "assistant", "text": f"Sorry, that query failed: {exc}"})
        _render_turn(history[-1])
        return

    history.append(
        {
            "role": "assistant",
            "text": answer.text or "(no answer)",
            "table": answer.table,
            "chart": answer.chart,
            "sql": answer.sql,
        }
    )
    _render_turn(history[-1])


def main() -> None:
    st.sidebar.title("Chatty Data Generation")
    tab = st.sidebar.radio("Navigate", ["Data Generation", "Talk to your data"])
    if tab == "Data Generation":
        data_generation_tab()
    else:
        talk_to_your_data_tab()


main()
