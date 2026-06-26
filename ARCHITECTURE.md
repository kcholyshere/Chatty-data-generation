# Architecture (current)

How a DDL upload becomes generated, persisted data (Phase 1), and how that data is then queried in
natural language (Phase 2). The Phase 2 flow is at the bottom of the diagram.

```mermaid
flowchart TD
    user([User]) -->|uploads .sql/.txt/.ddl + prompt + params| app["Streamlit UI<br/>src/app.py"]

    app -->|raw DDL text| parser["Hybrid DDL parser<br/>src/schema/parser.py"]
    parser -->|sqlparse + regex| schema[["Schema model<br/>tables, columns, keys, FKs<br/>src/schema/models.py"]]
    parser -. on parse failure .-> llm
    llm -. Table fragment .-> schema

    schema -->|topo_order: dependency order<br/>+ deferred cyclic FKs| engine["Generation engine<br/>src/generation/engine.py"]

    subgraph gen ["Per table (in dependency order)"]
        direction TB
        pk["PKs: sequential ints"]
        fk["FK values: sampled from<br/>parent table's generated PKs"]
        content["Content columns:<br/>LLM structured output (batched)"]
        validate["Backstop validation<br/>ENUM / CHECK / length / NOT NULL / UNIQUE"]
        pk --> validate
        fk --> validate
        content --> validate
    end

    engine --> gen
    content <-->|generate_structured<br/>response_schema = list of row model| llm["LLM client<br/>src/llm/client.py<br/>(Gemini via Google GenAI)"]
    config[".env settings<br/>src/config.py"] --> llm
    llm -. traces .-> langfuse([Langfuse])

    gen --> frames[["DataFrames<br/>one per table"]]
    engine -->|backfill deferred FKs| frames

    frames --> storage["Storage writer<br/>src/storage/writer.py"]
    storage --> csv[(CSV + Parquet<br/>data/processed/&lt;schema&gt;/)]
    storage --> sqlite[(SQLite DB<br/>data/processed/&lt;schema&gt;.db)]
    storage --> sidecar[(Schema sidecar JSON<br/>&lt;schema&gt;.schema.json<br/>cols/PKs/FKs for grounding)]

    frames --> preview["UI: table preview,<br/>per-table refine, CSV-zip download"]

    %% --- Phase 2: Talk to your data ---
    question([User question]) --> chat["Chat UI<br/>src/app.py"]
    chat --> service["Query service<br/>src/query/service.py"]
    sidecar -->|relationships| service
    service <-->|chat_with_tools_stream<br/>manual tool loop, streamed text| llm
    service -->|run_sql / plot_chart| roconn["read-only SELECT<br/>(mode=ro + guard)"]
    roconn --> sqlite
    service --> answer["Answer: text + table +<br/>plotly chart spec"]
    answer --> chat
```

**Key principle:** the LLM generates *content only*; keys and relationships are produced in code
(sequential PKs, FK values sampled from parent PKs), so referential integrity is guaranteed rather
than trusted to the model.
