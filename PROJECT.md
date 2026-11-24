# FinSentry — Finance Data Quality Monitoring & AI Analyst

> Portfolio project. **Snowflake-ready, runs locally on DuckDB.** Timeline: Oct 2026 – Nov 2026.
> Author: Sneha Mishra ([@Sneha0504](https://github.com/Sneha0504)).

## Original spec

Build a portfolio project called "Finance Data Quality Monitoring and AI Analyst on Snowflake".

**Data:** public FOCUS-format cloud billing sample data, or a synthetic revenue and
consumption dataset with documented assumptions and planted anomalies (label synthetic
data clearly in the README). Use a Snowflake free trial account.

**Build:**
1. **Pipeline:** raw files land in the warehouse, then dbt staging and mart models,
   orchestrated by an Airflow DAG (run locally in Docker). Include dbt tests for
   uniqueness, nulls, referential integrity and reconciliation between two source tables.
2. **Data quality monitoring:** statistical anomaly detection on daily revenue or
   consumption (median and MAD plus Isolation Forest), evaluated for precision and recall
   against the planted anomalies, with a log of each failure and its cause and a short
   runbook for fixing breakages.
3. **Streamlit app** showing data health, anomalies and a weekly revenue review view.
4. **AI analyst agent:** an LLM tool-calling loop with tools for running read-only SQL,
   running the anomaly check and creating charts. It automates the weekly business review
   write-up, with a verification script that checks every number in the generated text
   against the data.
5. **Evaluation:** 15 test questions with known answers, accuracy and failure cases
   reported, and every tool call logged.
6. **README** with architecture, design decisions, limitations and failure cases found.
   Pin dependencies, use fixed seeds and give one command to reproduce results. Every
   figure must come from a real run.

## Build decisions (this session)

- **Warehouse:** local **DuckDB** via `dbt-duckdb` so every figure comes from a real run
  with one command, no external credentials. Parallel **Snowflake DDL + deploy notes**
  ship alongside so it drops into a Snowflake trial unchanged where possible.
- **LLM provider:** **Groq** (free tier) via `GROQ_API_KEY`. A deterministic offline
  analyst mode (fixed seed) lets the 15-question eval and weekly-review number-verification
  reproduce with **no API key**; set the key to run the live Groq loop.
- **Orchestration:** real **Airflow DAG + docker-compose** (deploy target) PLUS a plain
  Python runner executing the identical steps, so the one-command repro runs without Docker.
- **Data:** synthetic revenue + consumption with documented assumptions, fixed seed, and
  **planted, labeled anomalies** used as ground truth for precision/recall. Clearly marked
  SYNTHETIC in README and data dictionary.

## Non-negotiables

- No Claude watermark anywhere (commits, PRs, README). Author = Sneha Mishra / @Sneha0504.
- Pinned dependencies, fixed seeds, one command to reproduce, every figure from a real run.
- Synthetic data labeled clearly.
