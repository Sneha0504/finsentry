# Deploying FinSentry on Snowflake

FinSentry was **built and validated on local DuckDB** so every figure reproduces
from one command with no credentials. This document is the **Snowflake-equivalent
path**: the same raw schema, the same dbt project, the same Streamlit app, and an
optional Snowflake Cortex path for the analyst. It is **Snowflake-ready, not
Snowflake-run** — treat it as the port guide, honestly labeled.

Everything below changes only *where the data lives and which engine runs the SQL*.
The dbt models, the tests, the anomaly detector, the Streamlit views and the
analyst tools are engine-agnostic.

---

## 0. Prerequisites

- A **Snowflake trial account** (https://signup.snowflake.com/ — 30-day free trial).
- `snowsql` CLI (or the Snowsight web worksheet) for running DDL and `PUT`.
- The repo cloned locally with the raw CSVs generated once:
  ```bash
  make reproduce            # builds data/raw/*.csv (+ the whole DuckDB run)
  ```

---

## 1. Create warehouse / database / schema and the raw tables (setup)

Run the shipped **setup** DDL. It creates `FINSENTRY_WH`, the `FINSENTRY`
database, the `RAW` schema, the four landing tables, the CSV file format and the
internal stage. It is **idempotent and non-destructive** — safe to re-run; it
never touches staged files or table data (the stage uses `CREATE … IF NOT EXISTS`):

```bash
snowsql -a <account> -u <user> -f snowflake/ddl.sql
```

(dbt will create the `ANALYTICS` schema itself on first run.)

---

## 2. Load the CSVs

Loading is a **separate** file (`snowflake/load.sql`) so re-running setup can
never wipe uploaded data. Run it from the repo **root** in a `snowsql` session so
the `file://data/raw/...` paths resolve:

```bash
snowsql -a <account> -u <user> -f snowflake/load.sql
```

`load.sql` does the `PUT`s (with `OVERWRITE=TRUE`), `TRUNCATE`s each table so a
reload is idempotent, `COPY INTO`s, and prints the row-count sanity check
(`21900 / 1825 / 1825 / 18`). Re-run it any time to reload from fresh CSVs.

---

## 3. Point dbt at Snowflake

The project's `dbt/finsentry/profiles.yml` ships a DuckDB `dev` target. Add a
`snowflake` output with the **same schema names**. The models reference
`source('raw', …)` and build into `analytics`; all but one are portable ANSI SQL
that runs as-is. The one dialect-specific model — `dim_date` — is already written
to branch on `target.type` (see the dialect note below), so **no model edits are
needed**: selecting the `snowflake` target is enough.

```yaml
# ~/.dbt/profiles.yml  (or dbt/finsentry/profiles.yml with a new output)
finsentry:
  target: snowflake
  outputs:
    snowflake:
      type: snowflake
      account: "{{ env_var('SNOWFLAKE_ACCOUNT') }}"
      user: "{{ env_var('SNOWFLAKE_USER') }}"
      password: "{{ env_var('SNOWFLAKE_PASSWORD') }}"   # or authenticator: externalbrowser
      role: ACCOUNTADMIN            # or a dedicated FINSENTRY_ROLE
      warehouse: FINSENTRY_WH
      database: FINSENTRY
      schema: ANALYTICS             # dbt builds staging + marts here
      threads: 4                    # Snowflake handles concurrency (DuckDB used 1)
```

Install the Snowflake adapter in a **separate** virtualenv (do not mix with the
pinned `dbt-duckdb` in `requirements.txt`):

```bash
pip install "dbt-snowflake==1.8.*"
```

Then run the identical build + tests:

```bash
dbt run  --project-dir dbt/finsentry --profiles-dir dbt/finsentry --target snowflake
dbt test --project-dir dbt/finsentry --profiles-dir dbt/finsentry --target snowflake
```

Expected: **9 models**, **65 data tests** green (uniqueness, not-null,
referential integrity, the cross-source reconciliation singular test, and the
missing-source-day test) — the same model and test counts as on DuckDB.

> **Note on dialect — the one engine-specific model.** Every model is portable
> ANSI SQL (window functions, `coalesce`, `extract`) **except the date spine in
> `dim_date`**. DuckDB builds it with `generate_series(start, end, interval '1 day')`,
> which Snowflake does not have. `dim_date.sql` therefore branches on `target.type`:
> the Snowflake branch builds the spine with `TABLE(GENERATOR(ROWCOUNT => 20000))` +
> `DATEADD(day, seq4(), start_date)` and clips to the end date, e.g.
>
> ```sql
> -- Snowflake branch (already in models/marts/dim_date.sql):
> select dateadd(day, seq4(), (select start_date from bounds)) as date_ts
> from table(generator(rowcount => 20000))
> -- ... final select ... where date_ts <= (select end_date from bounds)
> ```
>
> So selecting the `snowflake` target is sufficient; no manual edit is required.
> The reconciliation singular test is data-driven (allowlist from the labels), so
> it ports as-is.

---

## 4. Deploy the Streamlit app as Streamlit-in-Snowflake

`streamlit/app.py` is written so the port touches **two lines, both in the data
layer** — `_get_connection()` and the one query line in `run_query()`:

```python
# _get_connection()  --  Local (this repo):
import duckdb
return duckdb.connect(str(DUCKDB_PATH), read_only=True)

# _get_connection()  --  Streamlit-in-Snowflake (injected session, no creds in code):
from snowflake.snowpark.context import get_active_session
return get_active_session()

# run_query()  --  change the ONE query line:
#   DuckDB:     df = con.execute(sql).fetchdf()
#   Snowflake:  df = con.sql(sql).to_pandas()
```

`run_query()` already calls `df.columns = df.columns.str.lower()`, which normalizes
Snowflake's UPPER-CASE result columns down to the lower-case names every loader
uses (`df["date"]`, `df["total_billed_cost"]`, …) — so with those two line changes
none of the three views change. Every query is plain `SELECT … FROM analytics.*`.

Create the app from Snowsight:

1. **Projects → Streamlit → + Streamlit App**, pick the `FINSENTRY` database,
   `ANALYTICS` schema and `FINSENTRY_WH` warehouse.
2. Paste `streamlit/app.py` (with the two data-layer changes above).
3. The committed figures (`assets/*.png`) and the `reports/*.json` the app reads
   for the eval/verification summaries are repo files, not warehouse data. Either
   upload them to a Snowflake **internal stage** and read them in the app with
   `SnowflakeFile.open('@FINSENTRY.ANALYTICS.<stage>/precision_recall.png')` (the
   `snowflake.snowpark.files` API), or skip them — the app already **guards every
   missing file** (`st.image`/`st.markdown`/JSON panels show a friendly "run the
   pipeline" / "generate it" note) so those panels simply degrade gracefully.

Streamlit-in-Snowflake pins the runtime; the app sticks to Streamlit APIs
available there (`st.tabs`, `st.metric`, `st.dataframe`, `st.plotly_chart`).

---

## 5. (Optional) Analyst on Snowflake Cortex

The local analyst runs a tool-calling loop against **Groq** (live) or the
deterministic **offline** planner (committed results). On Snowflake you can keep
the exact same three tools (read-only SQL, anomaly check, chart) and swap the LLM
for **Snowflake Cortex**:

- Replace the Groq chat call in `src/finsentry/analyst/llm.py` with
  `SNOWFLAKE.CORTEX.COMPLETE('<model>', <prompt/messages>)` (or the Cortex REST
  API). Cortex supports tool/function calling on the newer models; wire the same
  `TOOL_SCHEMAS` from `analyst/tools.py`.
- Keep `verify_numbers.py` as the grounding guardrail — it re-queries every number
  in the generated review from the warehouse, unchanged.
- The read-only SQL tool's validator (single `SELECT`, no DDL, no file access)
  applies equally; on Snowflake also run the analyst under a **read-only role**
  granted only `SELECT` on `ANALYTICS`.

This keeps the whole agent inside Snowflake (no external API egress), at the cost
of Cortex availability/region. The offline planner remains the reproducible path.

---

## 6. Orchestration

Locally, `python -m finsentry.pipeline_runner` (or the Airflow DAG in `airflow/`)
runs generate → load → dbt run → dbt test → monitoring → analyst. On Snowflake,
schedule the dbt build with a **Snowflake Task** (or dbt Cloud / Airflow against
the Snowflake target). The monitoring + analyst stages run as Python (Snowpark
stored procedures, or an external runner) against the `ANALYTICS` schema.

---

## Honesty note

Row counts, model/test counts and every metric in the README come from the **real
DuckDB run**. The Snowflake path above is the validated-by-construction port (same
SQL, same dbt graph, same app code) — stand it up in a trial to reproduce the same
`analytics` marts and tests on Snowflake.
