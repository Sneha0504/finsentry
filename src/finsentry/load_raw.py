"""Load the raw synthetic CSVs into DuckDB under a ``raw`` schema.

Creates (or replaces) ``raw.billing_line_items``, ``raw.revenue_ledger``,
``raw.consumption`` and ``raw.anomaly_labels`` in the DuckDB database at
``config.paths.duckdb_path``. Idempotent: the ``raw`` schema is dropped and
recreated on every run, so re-running yields the same state.

Run from the project root (with the venv active)::

    python -m finsentry.load_raw
"""

from __future__ import annotations

from pathlib import Path

import duckdb

from finsentry.config import Config, load_config

# Logical table name -> raw CSV filename.
_TABLES: dict[str, str] = {
    "billing_line_items": "billing_line_items.csv",
    "revenue_ledger": "revenue_ledger.csv",
    "consumption": "consumption.csv",
    "anomaly_labels": "anomaly_labels.csv",
}


def load_raw(cfg: Config | None = None) -> dict[str, int]:
    """Load raw CSVs into DuckDB. Returns table name -> row count."""
    cfg = cfg or load_config()

    missing = [
        fname for fname in _TABLES.values() if not (cfg.raw_dir / fname).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing raw CSVs: "
            + ", ".join(missing)
            + ". Run `python -m finsentry.generate_data` first."
        )

    cfg.duckdb_path.parent.mkdir(parents=True, exist_ok=True)

    counts: dict[str, int] = {}
    con = duckdb.connect(str(cfg.duckdb_path))
    try:
        # Fresh raw schema every run (idempotent).
        con.execute("DROP SCHEMA IF EXISTS raw CASCADE;")
        con.execute("CREATE SCHEMA raw;")

        for table, fname in _TABLES.items():
            csv_path = (cfg.raw_dir / fname).resolve()
            con.execute(
                f"CREATE OR REPLACE TABLE raw.{table} AS "
                "SELECT * FROM read_csv_auto(?, header=true);",
                [str(csv_path)],
            )
            counts[table] = con.execute(
                f"SELECT COUNT(*) FROM raw.{table};"
            ).fetchone()[0]
    finally:
        con.close()

    return counts


def main() -> None:
    cfg = load_config()
    counts = load_raw(cfg)
    print(f"Loaded raw CSVs into DuckDB at {cfg.duckdb_path}")
    for table, n in counts.items():
        print(f"  raw.{table:<20} {n:>7,} rows")


if __name__ == "__main__":
    main()
