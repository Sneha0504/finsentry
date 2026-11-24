# FinSentry orchestration — Airflow (deploy target)

The **deploy path** for FinSentry is a real Airflow DAG (`finsentry_pipeline`) run in
Docker. For everyday, credential-free, one-command reproduction you do **not** need
Airflow — the plain Python runner executes the identical stages:

```bash
# One-command local fallback (no Docker, no Airflow):
.venv/bin/python -m finsentry.pipeline_runner
```

The DAG mirrors the runner exactly:

```
generate_data → load_raw → dbt_run → dbt_test → [anomaly/review placeholder]
```

## Why Airflow is separate

Airflow 2.9.3's dependency constraints pin different `pandas`/`numpy` versions than
the core `requirements.txt`. **Do not install Airflow into the core `.venv`.** It runs
in its own Docker containers (this folder) or a dedicated virtualenv
(`requirements-airflow.txt`).

## Bring up local Airflow

Requires Docker + Docker Compose. From the repo root:

```bash
cd airflow

# First run only: initialise the metadata DB and admin user.
AIRFLOW_UID=$(id -u) docker compose up airflow-init

# Start the webserver + scheduler.
AIRFLOW_UID=$(id -u) docker compose up
```

- UI: http://localhost:8080  (login `airflow` / `airflow`)
- The repo is mounted read/write at `/opt/finsentry` inside the containers, so the
  DAG regenerates data, loads DuckDB and runs dbt against the repo in place.
- Enable and trigger the **`finsentry_pipeline`** DAG from the UI.

Tear down (and wipe the metadata volume):

```bash
docker compose down -v
```

## Configuration

Paths are resolved as **Airflow Variable → environment variable → default**:

| Airflow Variable            | Env var                   | Default            |
| --------------------------- | ------------------------- | ------------------ |
| `finsentry_project_root`    | `FINSENTRY_PROJECT_ROOT`  | `/opt/finsentry`   |
| `finsentry_python_bin`      | `FINSENTRY_PYTHON_BIN`    | `python`           |
| `finsentry_dbt_bin`         | `FINSENTRY_DBT_BIN`       | `dbt`              |

The compose file installs dbt + the scientific stack into the Airflow image via
`_PIP_ADDITIONAL_REQUIREMENTS` and sets `PYTHONPATH=/opt/finsentry/src` so the
`finsentry` package is importable.

## Validate the DAG parses (without a running Airflow)

```bash
# Syntax / AST check — no Airflow needed:
python -m py_compile airflow/dags/finsentry_dag.py

# Full import/parse validation (needs the Airflow env, e.g. inside the container):
docker compose run --rm airflow-scheduler airflow dags list
docker compose run --rm airflow-scheduler airflow dags list-import-errors
```
