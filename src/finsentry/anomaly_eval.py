"""FinSentry Unit 3 - evaluation of the anomaly detectors vs planted labels.

Scores each detector (robust MAD, Isolation Forest) and their UNION against the
labeled revenue/consumption metric anomalies (``stg_anomaly_labels`` excluding
``recon_break``).  Reconciliation breaks are reported separately - they are caught
by the dbt reconciliation rule / ``mart_daily_health.recon_breach``, not the series
detectors.

For every false positive we attribute a *cause*, mapping the flagged day to the
known, UNLABELED confounders from ``config.yaml`` (deep holiday dips, one-off
shocks) or to neighbor-day spillover, so the precision/recall story is explainable
rather than a bare number.  Results are written to ``reports/anomaly_eval.json``
and a readable ``reports/anomaly_eval.md``.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

import pandas as pd

from finsentry.config import Config

# Detectors evaluated, and the per-day flag column each uses.
_DETECTORS = {
    "mad": ("mad_flag", "Robust median/MAD (|z| > threshold)"),
    "iforest": ("iforest_flag", "Isolation Forest (sklearn)"),
    "union": ("any_flag", "Union (MAD OR Isolation Forest)"),
}


def _key(d: pd.Timestamp | dt.date, metric: str) -> str:
    return f"{pd.Timestamp(d).date().isoformat()}|{metric}"


# --------------------------------------------------------------------------- #
# Confounder / cause attribution
# --------------------------------------------------------------------------- #
def build_confounder_index(cfg: Config) -> dict[str, dict[str, Any]]:
    """Map ``date|metric`` (and ``date|*`` for holidays) -> confounder cause."""
    conf = cfg.confounders or {}
    idx: dict[str, dict[str, Any]] = {}

    for h in conf.get("holidays", []):
        d = pd.Timestamp(h["date"]).date().isoformat()
        # Holidays scale BOTH billing and consumption -> can FP on either metric.
        for metric in ("revenue", "consumption"):
            idx[f"{d}|{metric}"] = {
                "kind": "holiday",
                "cause": f"holiday dip ({h['note']}) - expected seasonal "
                "variation, not a true anomaly",
            }

    for s in conf.get("shock_days", []):
        d = pd.Timestamp(s["date"]).date().isoformat()
        idx[f"{d}|{s['metric']}"] = {
            "kind": "shock",
            "cause": f"one-off heavy-tail shock ({s['note']})",
        }
    return idx


def attribute_false_positive(
    date: pd.Timestamp,
    metric: str,
    conf_idx: dict[str, dict[str, Any]],
    label_dates: set[str],
) -> dict[str, str]:
    """Explain why a flagged day is a false positive (not a planted label)."""
    k = _key(date, metric)
    if k in conf_idx:
        return {"cause_kind": conf_idx[k]["kind"], "cause": conf_idx[k]["cause"]}

    # Neighbor-day spillover: a labeled event one/two days away can distort the
    # centered trend window and push an adjacent day over threshold.
    for off in (1, -1, 2, -2):
        nk = _key(date + pd.Timedelta(days=off), metric)
        if nk in label_dates:
            return {
                "cause_kind": "spillover",
                "cause": f"neighbor-day spillover from the labeled {metric} anomaly "
                f"on {(date + pd.Timedelta(days=off)).date().isoformat()} "
                "(centered detrending window straddles the event)",
            }
    return {
        "cause_kind": "residual_noise",
        "cause": "unexplained near-threshold series noise (no known calendar/shock "
        "cause) - residual false positive that keeps precision < 1",
    }


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def _metric_labels(labels: pd.DataFrame) -> pd.DataFrame:
    """Revenue/consumption labels only (recon_break handled separately)."""
    m = labels[labels["anomaly_type"] != "recon_break"].copy()
    return m.reset_index(drop=True)


def evaluate(
    detections: pd.DataFrame,
    labels: pd.DataFrame,
    health: pd.DataFrame,
    cfg: Config,
    params: Any,
) -> dict[str, Any]:
    """Score every detector + union and build the per-anomaly / FP breakdown.

    ``health`` is ``mart_daily_health`` and is used to MEASURE (not assume)
    reconciliation-break recall from the real ``recon_breach`` flag.
    """
    metric_labels = _metric_labels(labels)
    label_keys = {_key(r.anomaly_date, r.metric) for r in metric_labels.itertuples()}
    conf_idx = build_confounder_index(cfg)

    # date|metric -> label note for the caught/missed breakdown.
    label_note = {
        _key(r.anomaly_date, r.metric): {
            "date": pd.Timestamp(r.anomaly_date).date().isoformat(),
            "metric": r.metric,
            "anomaly_type": r.anomaly_type,
            "severity": r.severity,
            "magnitude": float(r.magnitude),
            "note": r.note,
        }
        for r in metric_labels.itertuples()
    }

    results: dict[str, Any] = {"detectors": {}}

    for name, (flag_col, desc) in _DETECTORS.items():
        flagged = detections[detections[flag_col]]
        flagged_keys = {_key(r.date, r.metric) for r in flagged.itertuples()}

        tp_keys = sorted(flagged_keys & label_keys)
        fp_keys = sorted(flagged_keys - label_keys)
        fn_keys = sorted(label_keys - flagged_keys)

        tp, fp, fn = len(tp_keys), len(fp_keys), len(fn_keys)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        # Per-anomaly caught/missed.
        caught = [label_note[k] for k in tp_keys]
        missed = [label_note[k] for k in fn_keys]

        # FP attribution.
        score_by_key = {
            _key(r.date, r.metric): {
                "robust_z": float(r.robust_z),
                "iforest_score": float(r.iforest_score),
            }
            for r in flagged.itertuples()
        }
        false_positives = []
        for k in fp_keys:
            d_str, m = k.split("|")
            d = pd.Timestamp(d_str)
            attr = attribute_false_positive(d, m, conf_idx, label_keys)
            false_positives.append(
                {
                    "date": d_str,
                    "metric": m,
                    "robust_z": round(score_by_key[k]["robust_z"], 3),
                    "iforest_score": round(score_by_key[k]["iforest_score"], 4),
                    **attr,
                }
            )

        results["detectors"][name] = {
            "description": desc,
            "flag_column": flag_col,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "caught": caught,
            "missed": missed,
            "false_positives": false_positives,
        }

    # Reconciliation breaks: caught by the recon rule / recon_breach flag. Recall
    # is MEASURED from the real mart_daily_health.recon_breach column, not assumed.
    recon_labels = labels[labels["anomaly_type"] == "recon_break"]
    labeled_break_dates = {
        pd.Timestamp(d).date().isoformat() for d in recon_labels["anomaly_date"]
    }
    breach_dates = {
        pd.Timestamp(d).date().isoformat()
        for d in health.loc[health["recon_breach"] == True, "date"]  # noqa: E712
    }
    detected = sorted(labeled_break_dates & breach_dates)   # measured hits
    missed = sorted(labeled_break_dates - breach_dates)     # labeled but not flagged
    unexpected = sorted(breach_dates - labeled_break_dates)  # flagged but unlabeled
    n_labeled = len(labeled_break_dates)
    recon_recall = len(detected) / n_labeled if n_labeled else 0.0
    results["reconciliation"] = {
        "description": "Reconciliation breaks are detected by the dbt reconciliation "
        "rule and mart_daily_health.recon_breach, NOT the series detectors. Recall is "
        "measured from the real recon_breach flag.",
        "labeled_breaks": int(n_labeled),
        "detected_breaks": int(len(detected)),
        "recall": round(recon_recall, 4),
        "detected_dates": detected,
        "missed_dates": missed,
        "unexpected_breaks": unexpected,  # breach days with NO recon_break label
    }

    results["meta"] = {
        "revenue_metric_column": params.revenue_metric_column,
        "trend_window": params.trend_window,
        "dow_deseasonalize": params.dow_deseasonalize,
        "mad_threshold": params.mad_threshold,
        "iforest_n_estimators": params.iforest_n_estimators,
        "iforest_contamination": params.iforest_contamination,
        "iforest_seed": params.iforest_seed,
        "n_metric_labels": int(len(metric_labels)),
        "n_days_scored": int(detections["date"].nunique()),
        "data_date_range": [
            cfg.start_date.isoformat(),
            cfg.end_date.isoformat(),
        ],
    }
    return results


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def write_reports(result: dict[str, Any], cfg: Config) -> dict[str, Path]:
    """Write ``anomaly_eval.json`` and a readable ``anomaly_eval.md``."""
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    json_path = cfg.reports_dir / "anomaly_eval.json"
    md_path = cfg.reports_dir / "anomaly_eval.md"

    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, sort_keys=False)
        fh.write("\n")

    md = _render_md(result, cfg)
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(md)

    return {"json": json_path, "md": md_path}


def _render_md(result: dict[str, Any], cfg: Config) -> str:
    meta = result["meta"]
    lines: list[str] = []
    lines.append("# FinSentry - Anomaly Detection Evaluation\n")
    lines.append(
        "Synthetic data; metrics come from a real detection run on the built "
        "DuckDB warehouse. Reconciliation breaks are scored separately (handled by "
        "the dbt recon rule).\n"
    )
    lines.append("## Configuration\n")
    lines.append(f"- Revenue metric column: `{meta['revenue_metric_column']}`")
    lines.append(
        f"- Trend window: {meta['trend_window']} days (centered median), "
        f"day-of-week deseasonalize: {meta['dow_deseasonalize']}"
    )
    lines.append(f"- MAD threshold: |z| > {meta['mad_threshold']}")
    lines.append(
        f"- Isolation Forest: n_estimators={meta['iforest_n_estimators']}, "
        f"contamination={meta['iforest_contamination']}, seed={meta['iforest_seed']}"
    )
    lines.append(
        f"- Labeled metric anomalies: {meta['n_metric_labels']} "
        f"(revenue + consumption), over {meta['n_days_scored']} days per metric\n"
    )

    lines.append("## Detector scores (vs labeled metric anomalies)\n")
    lines.append("| Detector | Precision | Recall | F1 | TP | FP | FN |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for name in ("mad", "iforest", "union"):
        d = result["detectors"][name]
        lines.append(
            f"| {d['description']} | {d['precision']:.3f} | {d['recall']:.3f} | "
            f"{d['f1']:.3f} | {d['tp']} | {d['fp']} | {d['fn']} |"
        )
    lines.append("")

    # Per-detector detail.
    for name in ("mad", "iforest", "union"):
        d = result["detectors"][name]
        lines.append(f"### {d['description']}\n")
        lines.append(f"**Caught ({len(d['caught'])}):**\n")
        if d["caught"]:
            for c in d["caught"]:
                lines.append(
                    f"- {c['date']} {c['metric']} {c['anomaly_type']} "
                    f"({c['severity']}, mag {c['magnitude']:+.2f}) - {c['note']}"
                )
        else:
            lines.append("- (none)")
        lines.append(f"\n**Missed ({len(d['missed'])}):**\n")
        if d["missed"]:
            for c in d["missed"]:
                lines.append(
                    f"- {c['date']} {c['metric']} {c['anomaly_type']} "
                    f"({c['severity']}, mag {c['magnitude']:+.2f}) - {c['note']}"
                )
        else:
            lines.append("- (none)")
        lines.append(f"\n**False positives ({len(d['false_positives'])}) with cause:**\n")
        if d["false_positives"]:
            for fp in d["false_positives"]:
                lines.append(
                    f"- {fp['date']} {fp['metric']} (robust_z={fp['robust_z']}, "
                    f"iforest_score={fp['iforest_score']}) -> "
                    f"[{fp['cause_kind']}] {fp['cause']}"
                )
        else:
            lines.append("- (none)")
        lines.append("")

    rec = result["reconciliation"]
    lines.append("## Reconciliation breaks (handled separately)\n")
    lines.append(rec["description"] + "\n")
    lines.append(
        f"- Detected {rec['detected_breaks']}/{rec['labeled_breaks']} labeled "
        f"recon breaks (measured recall {rec['recall']:.2f}): "
        f"{', '.join(rec['detected_dates']) or '(none)'}"
    )
    if rec["missed_dates"]:
        lines.append(f"- Missed labeled breaks: {', '.join(rec['missed_dates'])}")
    lines.append(
        "- Unexpected (unlabeled) breaches: "
        + (", ".join(rec["unexpected_breaks"]) if rec["unexpected_breaks"] else "none")
        + "\n"
    )
    return "\n".join(lines)
