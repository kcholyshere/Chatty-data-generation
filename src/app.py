"""Streamlit UI.

Sidebar with two tabs: *Data Generation* (functional) and *Talk to your data* (Phase 2 stub).
Run with: ``uv run streamlit run src/app.py``.
"""

from __future__ import annotations

import streamlit as st

from generation.engine import GenerationConfig, generate, regenerate_table
from schema.parser import parse_ddl
from storage.writer import build_csv_zip, write_dataset

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
        paths = write_dataset(frames, st.session_state["schema_name"])
        st.session_state["sqlite_path"] = str(paths["sqlite"])

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
            write_dataset(frames, st.session_state["schema_name"])
            st.rerun()
        except Exception as exc:  # noqa: BLE001
            st.error(f"Refinement failed: {exc}")

    st.download_button(
        "Download all tables (CSV zip)",
        data=build_csv_zip(frames),
        file_name=f"{st.session_state.get('schema_name', 'data')}.zip",
        mime="application/zip",
    )
    if st.session_state.get("sqlite_path"):
        st.caption(f"SQLite database written to `{st.session_state['sqlite_path']}`")


def talk_to_your_data_tab() -> None:
    st.header("Talk to your data")
    st.info("Coming in Phase 2 — natural-language querying of the generated data.")


def main() -> None:
    st.sidebar.title("Chatty Data Generation")
    tab = st.sidebar.radio("Navigate", ["Data Generation", "Talk to your data"])
    if tab == "Data Generation":
        data_generation_tab()
    else:
        talk_to_your_data_tab()


main()
