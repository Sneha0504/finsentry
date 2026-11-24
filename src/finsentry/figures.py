"""FinSentry Unit 3 - figures, every one from a real detection run.

Deterministic matplotlib (Agg backend, fixed style, no wall-clock in any image):

* ``revenue_timeseries_anomalies.png`` - daily revenue + trend; labeled-truth vs
  detected days distinguished.
* ``consumption_timeseries_anomalies.png`` - same for consumption.
* ``reconciliation.png`` - daily recon_pct_diff vs the tolerance line, breaks
  highlighted.
* ``precision_recall.png`` - precision / recall / F1 per detector.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless, deterministic

import matplotlib.pyplot as plt
import pandas as pd

from finsentry.config import Config

_STYLE = {
    "figure.figsize": (12, 5),
    "figure.dpi": 120,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 10,
}

_C_SERIES = "#3b6ea5"
_C_TREND = "#222222"
_C_TRUTH = "#1a9641"
_C_MAD = "#d7191c"
_C_IF = "#fdae61"
_C_RECON = "#7b3294"


def _metric_labels(labels: pd.DataFrame, metric: str) -> pd.DataFrame:
    m = labels[(labels["metric"] == metric) & (labels["anomaly_type"] != "recon_break")]
    return m.copy()


def _timeseries_fig(
    det: pd.DataFrame,
    labels: pd.DataFrame,
    metric: str,
    title: str,
    ylabel: str,
    out_path: Path,
) -> Path:
    d = det[det["metric"] == metric].sort_values("date")
    truth = _metric_labels(labels, metric)
    truth_dates = set(pd.to_datetime(truth["anomaly_date"]).dt.normalize())

    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots()
        ax.plot(d["date"], d["value"], color=_C_SERIES, lw=1.0, label=f"daily {metric}")
        ax.plot(d["date"], d["trend"], color=_C_TREND, lw=1.2, ls="--",
                label="rolling-median trend")

        # Labeled ground truth (green triangles on the series).
        t = d[d["date"].dt.normalize().isin(truth_dates)]
        ax.scatter(t["date"], t["value"], marker="^", s=90, facecolors="none",
                   edgecolors=_C_TRUTH, linewidths=1.8, zorder=5,
                   label="labeled truth")

        # MAD-detected.
        mad = d[d["mad_flag"]]
        ax.scatter(mad["date"], mad["value"], marker="o", s=45, color=_C_MAD,
                   zorder=6, label="MAD flagged")
        # Isolation-Forest-detected (only-IF shown as separate marker).
        if_only = d[d["iforest_flag"] & ~d["mad_flag"]]
        ax.scatter(if_only["date"], if_only["value"], marker="x", s=55, color=_C_IF,
                   linewidths=1.8, zorder=6, label="IsolationForest-only flagged")

        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("date")
        ax.legend(loc="upper left", fontsize=8, ncol=2, framealpha=0.9)
        fig.tight_layout()
        fig.savefig(out_path, metadata={"Software": "FinSentry"})
        plt.close(fig)
    return out_path


def _reconciliation_fig(
    health: pd.DataFrame, labels: pd.DataFrame, tol: float, out_path: Path
) -> Path:
    h = health.sort_values("date").copy()
    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots()
        ax.plot(h["date"], h["recon_pct_diff"] * 100.0, color=_C_SERIES, lw=1.0,
                label="recon |pct diff|")
        ax.axhline(tol * 100.0, color=_C_MAD, ls="--", lw=1.3,
                   label=f"tolerance {tol * 100:.1f}%")
        breaches = h[h["recon_breach"] == True]  # noqa: E712
        ax.scatter(breaches["date"], breaches["recon_pct_diff"] * 100.0, marker="o",
                   s=70, color=_C_RECON, zorder=6, label="recon break")
        for r in breaches.itertuples():
            ax.annotate(
                pd.Timestamp(r.date).date().isoformat(),
                (r.date, r.recon_pct_diff * 100.0),
                textcoords="offset points", xytext=(0, 8), fontsize=8, ha="center",
            )
        ax.set_title("Daily revenue reconciliation (billing vs ledger)")
        ax.set_ylabel("|billed - recognized| / billed  (%)")
        ax.set_xlabel("date")
        ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
        fig.tight_layout()
        fig.savefig(out_path, metadata={"Software": "FinSentry"})
        plt.close(fig)
    return out_path


def _precision_recall_fig(eval_result: dict[str, Any], out_path: Path) -> Path:
    names = ["mad", "iforest", "union"]
    labels = [eval_result["detectors"][n]["description"].split(" (")[0] for n in names]
    metrics = ["precision", "recall", "f1"]
    colors = {"precision": "#2c7fb8", "recall": "#7fcdbb", "f1": "#253494"}

    import numpy as np

    x = np.arange(len(names))
    width = 0.26
    with plt.rc_context({**_STYLE, "figure.figsize": (9, 5)}):
        fig, ax = plt.subplots()
        for i, m in enumerate(metrics):
            vals = [eval_result["detectors"][n][m] for n in names]
            bars = ax.bar(x + (i - 1) * width, vals, width, label=m, color=colors[m])
            for b, v in zip(bars, vals):
                ax.annotate(f"{v:.2f}", (b.get_x() + b.get_width() / 2, v),
                            textcoords="offset points", xytext=(0, 3), ha="center",
                            fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("score")
        ax.set_title("Detector precision / recall / F1 (vs labeled metric anomalies)")
        ax.legend(loc="upper right", fontsize=9)
        fig.tight_layout()
        fig.savefig(out_path, metadata={"Software": "FinSentry"})
        plt.close(fig)
    return out_path


def generate_all(
    detections: pd.DataFrame,
    labels: pd.DataFrame,
    health: pd.DataFrame,
    eval_result: dict[str, Any],
    recon_tolerance_pct: float,
    cfg: Config,
) -> list[Path]:
    """Write all four figures; return the list of paths."""
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    rd = cfg.reports_dir
    out = []
    out.append(
        _timeseries_fig(
            detections, labels, "revenue",
            "Daily revenue with detected anomalies",
            "total billed cost (USD)",
            rd / "revenue_timeseries_anomalies.png",
        )
    )
    out.append(
        _timeseries_fig(
            detections, labels, "consumption",
            "Daily consumption with detected anomalies",
            "total consumption units",
            rd / "consumption_timeseries_anomalies.png",
        )
    )
    out.append(_reconciliation_fig(health, labels, recon_tolerance_pct,
                                   rd / "reconciliation.png"))
    out.append(_precision_recall_fig(eval_result, rd / "precision_recall.png"))
    return out
