"""FinSentry Unit 3 - data-quality monitoring: anomaly detection + orchestration.

Two deterministic detectors run over the daily revenue and consumption marts in
DuckDB (schema ``analytics``):

1. **Robust median/MAD** (statistical).  The daily series is deseasonalized with a
   centered rolling-median trend plus a robust day-of-week factor, then scored with
   the Iglewicz-Hoaglin robust z = ``0.6745 * (x - median) / MAD``.  A day is
   flagged when ``|z| > mad_threshold``.

2. **Isolation Forest** (sklearn).  A small, reproducible per-day feature set
   (deseasonalized value, robust z, rolling mean/std, lag delta, day-of-week) is
   fit *per metric* with a fixed ``random_state`` and ``n_estimators`` and a
   config-driven ``contamination``.  Predicted outliers are flagged.

Design choices (all parameters live in ``config.yaml`` -> ``anomaly_detection``):

* **Revenue metric = ``total_billed_cost``** (billing, source A), not
  ``recognized_revenue`` (ledger, source B).  Real demand/usage anomalies land in
  the billed series; ledger-vs-billing divergence is a *reconciliation* failure
  handled separately by the dbt recon rule + ``mart_daily_health.recon_breach``.
* **Reconciliation breaks are NOT scored here** - they are caught by the
  reconciliation logic and surfaced by the failure log; the series detectors are
  evaluated only against the labeled revenue/consumption metric anomalies.

``run_dq_monitoring()`` wires detection -> evaluation -> failure log -> figures and
is the entry point behind ``python -m finsentry.anomaly``.  Everything is
deterministic given the built warehouse (fixed seeds, no wall-clock in outputs).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from finsentry.config import Config, load_config

# Metrics scored by the series detectors and the mart column each reads.
# (revenue column is overridden from config.anomaly_detection.revenue_metric_column)
CONSUMPTION_COLUMN = "total_consumption_units"

# Per-day output schema produced by the detectors.
DETECT_COLUMNS = [
    "date",
    "metric",
    "value",
    "trend",
    "deseason",
    "robust_z",
    "mad_flag",
    "iforest_score",
    "iforest_flag",
    "any_flag",
]


# --------------------------------------------------------------------------- #
# Config access
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AnomalyParams:
    """Typed view over the ``anomaly_detection`` config block."""

    revenue_metric_column: str
    trend_window: int
    dow_deseasonalize: bool
    mad_threshold: float
    iforest_n_estimators: int
    iforest_contamination: float
    iforest_seed: int
    iforest_feature_window: int

    @classmethod
    def from_config(cls, cfg: Config) -> "AnomalyParams":
        a = cfg.anomaly_detection
        return cls(
            revenue_metric_column=str(a.get("revenue_metric_column", "total_billed_cost")),
            trend_window=int(a.get("trend_window", 31)),
            dow_deseasonalize=bool(a.get("dow_deseasonalize", True)),
            mad_threshold=float(a.get("mad_threshold", 3.5)),
            iforest_n_estimators=int(a.get("iforest_n_estimators", 200)),
            iforest_contamination=float(a.get("iforest_contamination", 0.03)),
            iforest_seed=int(a.get("iforest_seed", cfg.seed)),
            iforest_feature_window=int(a.get("iforest_feature_window", 7)),
        )


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def connect(cfg: Config, read_only: bool = True) -> duckdb.DuckDBPyConnection:
    """Open the FinSentry DuckDB warehouse."""
    return duckdb.connect(str(cfg.duckdb_path), read_only=read_only)


def load_metric_series(
    con: duckdb.DuckDBPyConnection, params: AnomalyParams
) -> dict[str, pd.DataFrame]:
    """Return ``{metric: DataFrame[date, value]}`` for revenue and consumption."""
    rev = con.execute(
        f"SELECT date, {params.revenue_metric_column} AS value "
        "FROM analytics.fct_daily_revenue ORDER BY date"
    ).fetchdf()
    cons = con.execute(
        f"SELECT date, {CONSUMPTION_COLUMN} AS value "
        "FROM analytics.fct_daily_consumption ORDER BY date"
    ).fetchdf()
    out = {}
    for metric, df in (("revenue", rev), ("consumption", cons)):
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df["value"] = df["value"].astype(float)
        out[metric] = df.sort_values("date").reset_index(drop=True)
    return out


def load_labels(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Ground-truth labels from ``stg_anomaly_labels`` (DATE normalized)."""
    df = con.execute(
        "SELECT anomaly_date, metric, anomaly_type, severity, magnitude, note "
        "FROM analytics.stg_anomaly_labels ORDER BY anomaly_date"
    ).fetchdf()
    df["anomaly_date"] = pd.to_datetime(df["anomaly_date"])
    return df


# --------------------------------------------------------------------------- #
# Detector 1: robust median / MAD
# --------------------------------------------------------------------------- #
def deseasonalize(df: pd.DataFrame, params: AnomalyParams) -> pd.DataFrame:
    """Add ``trend``, ``deseason`` and ``robust_z`` to a ``[date, value]`` frame.

    * ``trend``   = centered rolling median over ``trend_window`` days.
    * detrended   = value / trend (multiplicative; series is strictly positive).
    * day-of-week factor = robust (median) per-weekday level of the detrended
      series, normalized to median 1.0, divided out when ``dow_deseasonalize``.
    * ``deseason`` ~ 1.0 on normal days.
    * ``robust_z`` = 0.6745 * (deseason - median) / MAD  (Iglewicz-Hoaglin).
    """
    df = df.copy()
    x = df["value"].to_numpy(dtype=float)

    trend = (
        pd.Series(x)
        .rolling(params.trend_window, center=True, min_periods=1)
        .median()
        .to_numpy()
    )
    detrended = x / trend

    if params.dow_deseasonalize:
        dow = df["date"].dt.dayofweek.to_numpy()
        dow_factor = pd.Series(detrended).groupby(dow).transform("median").to_numpy()
        dow_factor = dow_factor / np.median(dow_factor)
    else:
        dow_factor = np.ones_like(detrended)

    deseason = detrended / dow_factor

    med = np.median(deseason)
    mad = np.median(np.abs(deseason - med))
    # MAD can be ~0 for a perfectly flat series; guard to keep z finite.
    scale = mad if mad > 1e-12 else (np.std(deseason) or 1.0)
    robust_z = 0.6745 * (deseason - med) / scale

    df["trend"] = trend
    df["deseason"] = deseason
    df["robust_z"] = robust_z
    return df


# --------------------------------------------------------------------------- #
# Detector 2: Isolation Forest
# --------------------------------------------------------------------------- #
def _build_features(df: pd.DataFrame, params: AnomalyParams) -> pd.DataFrame:
    """Reproducible per-day features for the Isolation Forest (needs `deseason`)."""
    df = df.copy()
    w = params.iforest_feature_window
    des = df["deseason"]
    df["roll_mean"] = des.rolling(w, center=True, min_periods=1).mean().to_numpy()
    df["roll_std"] = des.rolling(w, center=True, min_periods=1).std().fillna(0.0).to_numpy()
    df["lag_delta"] = np.concatenate([[0.0], np.diff(des.to_numpy())])
    df["dow"] = df["date"].dt.dayofweek.astype(float).to_numpy()
    return df


_IFOREST_FEATURES = ["deseason", "robust_z", "roll_mean", "roll_std", "lag_delta", "dow"]


def iforest_detect(df: pd.DataFrame, params: AnomalyParams) -> pd.DataFrame:
    """Add ``iforest_score`` (higher = more anomalous) and ``iforest_flag``."""
    df = _build_features(df, params)
    X = df[_IFOREST_FEATURES].to_numpy(dtype=float)
    clf = IsolationForest(
        n_estimators=params.iforest_n_estimators,
        contamination=params.iforest_contamination,
        random_state=params.iforest_seed,
    )
    pred = clf.fit_predict(X)  # -1 == outlier
    # score_samples: higher = more normal; negate so higher = more anomalous.
    df["iforest_score"] = -clf.score_samples(X)
    df["iforest_flag"] = pred == -1
    return df


# --------------------------------------------------------------------------- #
# Combined detection
# --------------------------------------------------------------------------- #
def detect_metric(
    df: pd.DataFrame, metric: str, params: AnomalyParams
) -> pd.DataFrame:
    """Run both detectors over one metric series; return the per-day frame."""
    df = deseasonalize(df, params)
    df = iforest_detect(df, params)
    df["metric"] = metric
    df["mad_flag"] = np.abs(df["robust_z"]) > params.mad_threshold
    df["any_flag"] = df["mad_flag"] | df["iforest_flag"]
    return df[DETECT_COLUMNS].reset_index(drop=True)


def detect_all(
    cfg: Config | None = None,
    con: duckdb.DuckDBPyConnection | None = None,
    params: AnomalyParams | None = None,
) -> pd.DataFrame:
    """Run detection over revenue + consumption; return one tidy per-day frame."""
    cfg = cfg or load_config()
    params = params or AnomalyParams.from_config(cfg)
    owned = con is None
    con = con or connect(cfg, read_only=True)
    try:
        series = load_metric_series(con, params)
    finally:
        if owned:
            con.close()
    frames = [detect_metric(series[m], m, params) for m in ("revenue", "consumption")]
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["metric", "date"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Orchestration entry point
# --------------------------------------------------------------------------- #
def run_dq_monitoring(cfg: Config | None = None, verbose: bool = True) -> dict[str, Any]:
    """Detection -> evaluation -> failure log -> figures, deterministically.

    Returns a summary dict (also the stage hook used by ``pipeline_runner``).
    """
    # Imported here to avoid a heavy import graph when only detectors are used.
    from finsentry import anomaly_eval, failure_log
    from finsentry import figures as figures_mod

    cfg = cfg or load_config()
    params = AnomalyParams.from_config(cfg)
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)

    def log(msg: str) -> None:
        if verbose:
            print(msg, flush=True)

    con = connect(cfg, read_only=False)
    try:
        detections = detect_all(cfg, con=con, params=params)
        labels = load_labels(con)
        health = con.execute(
            "SELECT date, total_billed_cost, recognized_revenue, recon_abs_diff, "
            "recon_pct_diff, total_consumption_units, recon_breach "
            "FROM analytics.mart_daily_health ORDER BY date"
        ).fetchdf()
        health["date"] = pd.to_datetime(health["date"])

        # B. Evaluation (per detector + union) vs metric ground truth.
        eval_result = anomaly_eval.evaluate(detections, labels, health, cfg, params)
        anomaly_eval.write_reports(eval_result, cfg)
        log("  eval written -> reports/anomaly_eval.json / .md")

        # C. DQ failure log -> DuckDB table + CSV.
        log_df = failure_log.build_failure_log(detections, labels, health, cfg, params)
        failure_log.write_failure_log(log_df, cfg, con=con)
        log(
            f"  failure log: {len(log_df)} rows -> analytics.dq_failure_log + "
            "reports/dq_failure_log.csv"
        )

        # E. Figures from the real run.
        tol = float(cfg.data_generation["recon_tolerance_pct"])
        written = figures_mod.generate_all(detections, labels, health, eval_result, tol, cfg)
        for p in written:
            log(f"  figure -> {p}")
    finally:
        con.close()

    stat = eval_result["detectors"]["mad"]
    log(
        "  MAD detector: precision={:.3f} recall={:.3f} f1={:.3f} "
        "(TP={} FP={} FN={})".format(
            stat["precision"], stat["recall"], stat["f1"],
            stat["tp"], stat["fp"], stat["fn"],
        )
    )
    return {
        "n_detections_rows": int(len(detections)),
        "n_failures": int(len(log_df)),
        "eval": eval_result,
        "figures": [str(p) for p in written],
    }


def main(argv: list[str] | None = None) -> int:
    run_dq_monitoring(load_config(), verbose=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
