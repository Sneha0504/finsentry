"""Airflow DAG mirroring the FinSentry local pipeline runner (deploy target).

This DAG runs the identical stages as ``finsentry.pipeline_runner`` so the Docker
deploy and the one-command local fallback stay in lock-step:

    generate_data -> load_raw -> dbt_run -> dbt_test -> [anomaly/review placeholder]

Paths are configurable via Airflow Variables (preferred) or environment variables,
with sane defaults for the bundled docker-compose (repo mounted at /opt/finsentry).

Full validation requires an Airflow environment (see airflow/requirements note and
airflow/README.md); do NOT install Airflow into the core venv. A standalone parse /
AST check (``python -m py_compile airflow/dags/finsentry_dag.py``) validates syntax
without Airflow installed.
"""

from __future__ import annotations

import os
import pendulum

from airflow import DAG
from airflow.models import Variable
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator


def _cfg(key: str, env_key: str, default: str) -> str:
    """Resolve a config value: Airflow Variable -> env var -> default.

    Variable.get can touch the metadata DB; it is wrapped so DAG parsing never
    fails if the Variable is unset or the backend is unavailable.
    """
    try:
        return Variable.get(key, default_var=os.environ.get(env_key, default))
    except Exception:
        return os.environ.get(env_key, default)


# --- Configurable paths (defaults match the docker-compose repo mount) ---
PROJECT_ROOT = _cfg("finsentry_project_root", "FINSENTRY_PROJECT_ROOT", "/opt/finsentry")
PYTHON_BIN = _cfg("finsentry_python_bin", "FINSENTRY_PYTHON_BIN", "python")
DBT_BIN = _cfg("finsentry_dbt_bin", "FINSENTRY_DBT_BIN", "dbt")

DBT_PROJECT_DIR = f"{PROJECT_ROOT}/dbt/finsentry"
DUCKDB_PATH = f"{PROJECT_ROOT}/data/warehouse.duckdb"

# dbt tasks need an absolute DuckDB path independent of the working directory.
_DBT_ENV = {"FINSENTRY_DUCKDB": DUCKDB_PATH}

_DBT_FLAGS = f"--project-dir {DBT_PROJECT_DIR} --profiles-dir {DBT_PROJECT_DIR}"

default_args = {
    "owner": "finsentry",
    "retries": 0,
}

with DAG(
    dag_id="finsentry_pipeline",
    description="FinSentry data-quality pipeline: generate -> load -> dbt run -> dbt test.",
    schedule=None,  # manual trigger; set a cron (e.g. "0 6 * * *") to run daily.
    start_date=pendulum.datetime(2025, 1, 1, tz="UTC"),
    catchup=False,
    default_args=default_args,
    tags=["finsentry", "dbt", "data-quality"],
) as dag:

    generate_data = BashOperator(
        task_id="generate_data",
        bash_command=f"cd {PROJECT_ROOT} && {PYTHON_BIN} -m finsentry.generate_data",
    )

    load_raw = BashOperator(
        task_id="load_raw",
        bash_command=f"cd {PROJECT_ROOT} && {PYTHON_BIN} -m finsentry.load_raw",
    )

    dbt_run = BashOperator(
        task_id="dbt_run",
        bash_command=f"cd {PROJECT_ROOT} && {DBT_BIN} run {_DBT_FLAGS}",
        env=_DBT_ENV,
        append_env=True,
    )

    dbt_test = BashOperator(
        task_id="dbt_test",
        bash_command=f"cd {PROJECT_ROOT} && {DBT_BIN} test {_DBT_FLAGS}",
        env=_DBT_ENV,
        append_env=True,
    )

    # Placeholder for later FinSentry units (anomaly detection + weekly review).
    # Replace with real operators when those units land; wiring/order already set.
    anomaly_and_review = EmptyOperator(task_id="anomaly_and_review_placeholder")

    generate_data >> load_raw >> dbt_run >> dbt_test >> anomaly_and_review
