# Project Report: Conversational Synthetic Data Generation and Querying

A conversational AI application with two functions: (1) generating valid synthetic data from a SQL
schema, and (2) querying that data in natural language ("talk to your data"). Built in Python with
Gemini (via the Google GenAI SDK on Vertex AI), Streamlit, PostgreSQL, Docker, and Langfuse.

## 1. Approach

### Overall design
The system is split into two phases that share a persistent store. Phase 1 turns an uploaded SQL DDL
schema into synthetic data and writes it to PostgreSQL. Phase 2 lets the user ask questions in plain
language; the model writes and runs SQL against that stored data and answers with text, tables, and
charts. A Streamlit sidebar exposes both as separate tabs.

The guiding principle throughout is "the model proposes, code executes". The LLM is used where it is
strong (producing realistic free-text content, and translating questions into SQL) and kept away from
anything where correctness must be guaranteed (keys, referential integrity, data mutation).

### Phase 1: synthetic data generation
- DDL parsing is hybrid. By default `sqlparse` plus regular expressions extract tables, columns, types,
  and constraints deterministically; only when a statement fails to parse does the system fall back to
  the LLM. This keeps structural understanding reliable and testable.
- The key design decision is that the LLM generates content columns only. Primary keys are assigned as
  sequential integers in code, foreign-key values are sampled from the parent table's already-generated
  primary keys, and a deterministic validation pass enforces ENUM, CHECK, length, NOT NULL, and UNIQUE.
  Referential integrity is therefore guaranteed by construction rather than trusted to the model.
- Generation runs in parallel: content batches across all tables are independent (because keys and FKs
  are not produced by the model), so they run concurrently in a bounded thread pool, cutting generation
  time substantially. A distinct variety seed per batch decorrelates batches that would otherwise reach
  for the same common values.
- Tables with cyclic foreign keys are handled by deferring one nullable edge during assembly and
  backfilling it with real parent keys once every table exists.
- Output is persisted to PostgreSQL (one schema per dataset, with parsed schema metadata in a
  `_meta.datasets` table) and also written as CSV and Parquet for inspection and download. Users can
  refine any table through a free-text feedback box that regenerates just that table's content columns.

### Phase 2: talk to your data
- The query model is given two tools, `run_sql` and `plot_chart`, plus the dataset schema, and drives a
  manual tool-calling loop that streams its answer token by token.
- Every query runs inside a read-only PostgreSQL transaction behind a single-SELECT guard, so a prompt
  can never modify or drop data even if it tries. Aggregations and JOINs are reads and so are supported;
  transformations the user asks for (for example a derived or computed column) are satisfied as part of
  a SELECT and never touch the stored rows.
- Results render inline as text, a table, and, where useful, a Plotly chart. The executed SQL is shown
  alongside the answer so the user can see exactly what ran.
- A guardrail screens each message before it reaches the query model. An LLM-as-judge classifies the
  message as safe or unsafe (prompt injection, jailbreak, or off-topic) and refuses unsafe ones. It
  fails open on any judge error, since the data layer is already hardened and breaking a session on a
  transient error would be worse.

### Infrastructure and observability
The whole stack runs in Docker (app plus PostgreSQL via docker-compose), authenticating to Vertex AI
with the host's application default credentials mounted read-only. Both the generation and query paths
are traced to Langfuse via OpenInference auto-instrumentation of the GenAI SDK, capturing model, token
usage, input/output, and tool calls; chat turns are grouped into a single trace per conversation.

## 2. Results

A complete, working application was delivered against every required feature. All required scope is
met; only items explicitly marked optional in the brief (PII masking, UI-editable queries, jailbreak
alerts, online evaluations) were left out.

Delivered:
- DDL parsing with constraint extraction and an LLM fallback.
- Constraint-respecting synthetic data with guaranteed referential integrity, configurable volume, and
  per-table refinement.
- Persistence to PostgreSQL plus CSV/Parquet, and CSV-zip download.
- A two-tab Streamlit UI with upload, prompt, generation parameters, generate button, per-table preview,
  and per-table editing.
- Natural-language querying with streamed responses, conversation history, automatic SQL generation and
  execution (including JOINs and aggregation), the source query shown with the result, and inline charts.
- Input guardrails against prompt injection and jailbreaks, on-topic enforcement, and Langfuse tracing
  on both paths.
- Full Docker deployment with Vertex AI authentication.

The codebase is roughly 1,900 lines of source across cleanly separated modules (schema, generation,
storage, query, llm), with 24 automated tests covering parsing, generation integrity, the guardrail,
and the read-only query guard.

### Quantitative validation
A live measurement (`scripts/validate_quality.py`) was run on the `library_mgm` schema (9 tables, 50
rows per table, gemini-3.5-flash):

- Referential integrity: 0 orphan foreign-key rows out of 500 FK values checked (0.00 percent broken).
  Because FK values are sampled from parent keys in code, this holds regardless of the schema.
- Content duplication: a mean of 17.7 percent across 56 non-key, non-unique content columns
  (duplication rate = 1 - distinct/total). This figure is dominated by genuinely low-cardinality columns
  where repetition is correct (gender at 94 percent, state and surnames around 18 percent), while
  high-cardinality fields such as names, emails, addresses, and dates sit at or near 0 percent.

## 3. Challenges

- Referential integrity was the hardest requirement, and the central insight that resolved it was to
  treat integrity as a code problem rather than a prompting problem. Early designs that asked the model
  to emit whole rows produced foreign keys that did not match any parent. Moving key and FK assignment
  into deterministic code removed the class of bug entirely, at the cost of a more involved generation
  pipeline (dependency ordering, FK sampling, deferred cyclic edges).
- Content repetition emerged once generation was parallelised. Independent batches given the same prompt
  reached for the same common values, producing near-duplicate non-unique columns. A distinct variety
  seed injected into each batch's prompt decorrelates them while keeping generation reproducible when a
  fixed seed is set.
- Streaming with tool calls was awkward. The SDK's automatic function calling returns empty output under
  streaming, so the loop is driven manually: stream each round, execute the model's tool calls, feed the
  results back, and repeat. The model's raw response parts must be fed back verbatim to preserve Gemini's
  required thought-signature on function-call parts.
- Langfuse integration involved real friction: a v4 API rename meant the first manual-span approach
  silently traced nothing, and the auto-instrumentor wraps the streaming call in a non-iterator iterable
  that had to be handled. Switching to framework auto-instrumentation, as the Langfuse guidance
  recommends, was more robust than hand-written spans.
- Two subtler issues shaped the query path: PostgreSQL identifiers are case-sensitive, so all table and
  column names are quoted in generated SQL; and `from __future__ import annotations` breaks the SDK's
  tool introspection, so tool functions reset their annotations to real types.

## 4. Limitations and future work

- Content variety is model-dependent and there is no statistical realism (distributions or cross-column
  correlations are not modelled).
- Composite (multi-column) foreign keys are sampled per column independently, so a sampled tuple may not
  exist in the parent; only single-column foreign keys are fully safe today.
- There is no automated evaluation of answer correctness; the tests cover the plumbing while the live
  model paths are stub-tested, so a wrong JOIN or a thin answer is not caught automatically.
- When the model issues several queries in one turn, the table displayed is the last one, which is
  usually but not always the one that answers the question.

Natural next steps would be whole-tuple sampling for composite FKs, an automated answer-quality
evaluation harness, and richer statistical control over generated distributions.
