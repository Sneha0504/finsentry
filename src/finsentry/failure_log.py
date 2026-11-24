"""FinSentry Unit 3 - unified data-quality FAILURE LOG.

Collects every DQ failure from a monitoring run into one table, each row carrying a
plain-language *cause*:

* **series anomalies** - every day flagged by the MAD and/or Isolation Forest
  detectors on revenue/consumption, tagged with the detector(s), the robust z /
  forest score, whether it matches a planted label, and - when it does not - the
  likely confounder cause (holiday dip, one-off shock, neighbor spillover, residual
  noise);
* **reconciliation breaks** - days where ``mart_daily_health.recon_breach`` is true,
  with billed vs recognized and the signed percentage divergence.

The log is written to BOTH the DuckDB table ``analytics.dq_failure_log`` (so the
Streamlit app and the AI analyst can read it) and ``reports/dq_failure_log.csv``.

Determinism: ``run_ts`` is set to the DATA end-date (``config.end_date``), never
wall-clock, so committed figures/CSVs reproduce byte-for-byte.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from finsentry.anomaly import connect
from finsentry.anomaly_eval import (
    _key,
    attribute_false_positive,
    build_confounder_index,
)
from finsentry.config import Config

FAILURE_LOG_COLUMNS = [
    "run_ts",
    "failure_id",
    "date",
    "failure_type",
    "metric",
    "detector",
    "severity",
    "value",
    "expected",
    "deviation",
    "cause",
    "is_known_label",
]


def _severity_from_z(z: float) -> str:
    """Derive a severity band from the robust z when there is no label."""
    az = abs(z)
    if az >= 6.0:
        return "high"
    if az >= 4.5:
        return "medium"
    return "low"


def build_failure_log(
    detections: pd.DataFrame,
    labels: pd.DataFrame,
    health: pd.DataFrame,
    cfg: Config,
    params: Any,
) -> pd.DataFrame:
    """Assemble the unified failure log as a deterministic DataFrame."""
    run_ts = cfg.end_date.isoformat()  # deterministic, data-driven (not wall-clock)
    conf_idx = build_confounder_index(cfg)

    # Metric-anomaly label lookup (exclude recon_break).
    metric_labels = labels[labels["anomaly_type"] != "recon_break"]
    label_keys = {_key(r.anomaly_date, r.metric) for r in metric_labels.itertuples()}
    label_info = {
        _key(r.anomaly_date, r.metric): {
            "severity": r.severity,
            "note": r.note,
            "magnitude": float(r.magnitude),
            "anomaly_type": r.anomaly_type,
        }
        for r in metric_labels.itertuples()
    }

    rows: list[dict[str, Any]] = []

    # --- Series anomalies (any flag) ---
    flagged = detections[detections["any_flag"]].copy()
    for r in flagged.itertuples():
        k = _key(r.date, r.metric)
        dets = []
        if r.mad_flag:
            dets.append("mad")
        if r.iforest_flag:
            dets.append("iforest")
        detector = "+".join(dets)

        # expected baseline (deseason == 1) and signed pct deviation.
        expected = r.value / r.deseason if r.deseason else float("nan")
        deviation = r.deseason - 1.0  # fractional deviation from the deseasonalized norm

        is_known = k in label_keys
        if is_known:
            info = label_info[k]
            severity = info["severity"]
            cause = (
                f"matches planted {r.metric} {info['anomaly_type']} label "
                f"(mag {info['magnitude']:+.2f}): {info['note']}"
            )
        else:
            severity = _severity_from_z(r.robust_z)
            attr = attribute_false_positive(
                pd.Timestamp(r.date), r.metric, conf_idx, label_keys
            )
            cause = f"suspected false positive [{attr['cause_kind']}]: {attr['cause']}"

        rows.append(
            {
                "date": pd.Timestamp(r.date).date().isoformat(),
                "failure_type": f"{r.metric}_anomaly",
                "metric": r.metric,
                "detector": detector,
                "severity": severity,
                "value": round(float(r.value), 2),
                "expected": round(float(expected), 2),
                "deviation": round(float(deviation), 4),
                "cause": cause,
                "is_known_label": bool(is_known),
                "_robust_z": float(r.robust_z),
            }
        )

    # --- Reconciliation breaks ---
    recon_label_dates = {
        pd.Timestamp(d).date().isoformat()
        for d in labels[labels["anomaly_type"] == "recon_break"]["anomaly_date"]
    }
    recon_notes = {
        pd.Timestamp(r.anomaly_date).date().isoformat(): r.note
        for r in labels[labels["anomaly_type"] == "recon_break"].itertuples()
    }
    breaches = health[health["recon_breach"] == True]  # noqa: E712
    for r in breaches.itertuples():
        d_str = pd.Timestamp(r.date).date().isoformat()
        billed = float(r.total_billed_cost)
        recognized = float(r.recognized_revenue)
        signed_dev = (recognized - billed) / billed if billed else float("nan")
        is_known = d_str in recon_label_dates
        if is_known:
            cause = f"matches planted recon_break label: {recon_notes[d_str]}"
        else:
            cause = (
                "UNEXPLAINED reconciliation break - ledger diverges from billing "
                "beyond tolerance with no planted label; investigate (would also "
                "fail the dbt assert_revenue_reconciliation test)"
            )
        rows.append(
            {
                "date": d_str,
                "failure_type": "reconciliation_break",
                "metric": "reconciliation",
                "detector": "dbt_recon_rule",
                "severity": "large",
                "value": round(recognized, 2),  # recognized revenue (ledger, B)
                "expected": round(billed, 2),  # billed cost (billing, A)
                "deviation": round(float(signed_dev), 4),
                "cause": cause,
                "is_known_label": bool(is_known),
                "_robust_z": float("nan"),
            }
        )

    # --- Missing day (present in dim_date but absent from a source) ---
    # mart_daily_health is built from dim_date, so a day missing from both billing
    # and ledger (-> no fct_daily_revenue row) carries NULL total_billed_cost; a day
    # missing from the consumption source carries NULL total_consumption_units.
    missing = health[
        health["total_billed_cost"].isna() | health["total_consumption_units"].isna()
    ]
    for r in missing.itertuples():
        d_str = pd.Timestamp(r.date).date().isoformat()
        gaps = []
        if pd.isna(r.total_billed_cost):
            gaps.append("revenue (billing+ledger)")
        if pd.isna(r.total_consumption_units):
            gaps.append("consumption")
        cause = (
            "missing day - no data from "
            + " and ".join(gaps)
            + " source(s); day present in dim_date but absent from the mart "
            "(also fails the mart_daily_health not_null / assert_no_missing_source_day tests)"
        )
        rows.append(
            {
                "date": d_str,
                "failure_type": "missing_day",
                "metric": "pipeline",
                "detector": "dim_date_coverage",
                "severity": "large",
                "value": float("nan"),
                "expected": float("nan"),
                "deviation": float("nan"),
                "cause": cause,
                "is_known_label": False,
                "_robust_z": float("nan"),
            }
        )

    if not rows:
        df = pd.DataFrame(columns=FAILURE_LOG_COLUMNS)
        return df

    df = pd.DataFrame(rows)
    # Deterministic ordering: date, then failure_type, then metric.
    df = df.sort_values(["date", "failure_type", "metric"], kind="mergesort").reset_index(
        drop=True
    )
    df.insert(0, "failure_id", [f"F{i + 1:04d}" for i in range(len(df))])
    df.insert(0, "run_ts", run_ts)
    df = df.drop(columns=["_robust_z"])
    return df[FAILURE_LOG_COLUMNS]


def write_failure_log(
    df: pd.DataFrame,
    cfg: Config,
    con: duckdb.DuckDBPyConnection | None = None,
) -> dict[str, Path]:
    """Write the log to DuckDB (``analytics.dq_failure_log``) and CSV."""
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    csv_path = cfg.reports_dir / "dq_failure_log.csv"
    df.to_csv(csv_path, index=False)

    owned = con is None
    con = con or connect(cfg, read_only=False)
    try:
        con.execute("CREATE SCHEMA IF NOT EXISTS analytics;")
        con.register("dq_failure_log_df", df)
        con.execute(
            "CREATE OR REPLACE TABLE analytics.dq_failure_log AS "
            "SELECT * FROM dq_failure_log_df"
        )
        con.unregister("dq_failure_log_df")
    finally:
        if owned:
            con.close()
    return {"csv": csv_path, "table": "analytics.dq_failure_log"}
