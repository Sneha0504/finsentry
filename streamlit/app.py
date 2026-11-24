"""FinSentry Streamlit app - local, read-only, Snowflake-portable.

Run locally::

    streamlit run streamlit/app.py

It reads the DuckDB warehouse (``data/warehouse.duckdb``, schema ``analytics``)
and the committed reports, and presents three views:

1. **Data health**   - dbt test summary, reconciliation status, missing-day
   coverage, and KPI tiles (revenue, consumption, active accounts).
2. **Anomalies**     - the DQ failure log + detected anomalies, revenue /
   consumption time series with flagged points, precision/recall summary,
   filterable by metric and severity.
3. **Weekly review** - the latest weekly business review and its key numbers.

Everything is READ-ONLY (the DuckDB connection is opened ``read_only=True``).

------------------------------------------------------------------------------
SNOWFLAKE PORTABILITY
------------------------------------------------------------------------------
The same app deploys unchanged as a **Streamlit-in-Snowflake** app. The ONLY
change is the connection: instead of opening a local DuckDB file, call
``st.connection("snowflake")`` (or use the Snowpark ``get_active_session()`` that
Streamlit-in-Snowflake injects) and run the identical ``analytics.*`` SQL against
the Snowflake schema built by the same dbt project. See ``_get_connection`` and
``run_query`` below for the single swap point, and ``snowflake/DEPLOY.md`` for the
deploy steps. The project was built and validated on DuckDB; Snowflake is the
documented, equivalent deploy target.
"""

from __future__ import annotations

import datetime as dt
import re
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# --------------------------------------------------------------------------- #
# Paths (resolved relative to the repo root, independent of the launch cwd).
# --------------------------------------------------------------------------- #
REPO_ROOT = Path(__file__).resolve().parents[1]
DUCKDB_PATH = REPO_ROOT / "data" / "warehouse.duckdb"
REPORTS_DIR = REPO_ROOT / "reports"
ASSETS_DIR = REPO_ROOT / "assets"

# --------------------------------------------------------------------------- #
# Accessible palette (colorblind-safe; shared with finsentry/figures.py so the
# app and the committed PNGs read as one system). dataviz skill: one axis per
# chart, legend always present for >= 2 series, status colors reserved.
# --------------------------------------------------------------------------- #
C_SERIES = "#3b6ea5"   # primary series (blue)
C_TREND = "#222222"    # trend / neutral ink
C_TRUTH = "#1a9641"    # labeled ground truth (green)
C_MAD = "#d7191c"      # MAD-flagged / breach (red)
C_IF = "#fb9a29"       # IsolationForest-only (amber; darkened for contrast)
C_RECON = "#7b3294"    # reconciliation (purple)
C_GOOD = "#1a9641"
C_BAD = "#d7191c"

st.set_page_config(page_title="FinSentry", page_icon="chart", layout="wide")


# --------------------------------------------------------------------------- #
# Connection + query (THE Snowflake swap point).
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def _get_connection():
    """Open the warehouse READ-ONLY (local DuckDB).

    THE Snowflake swap point. To deploy as Streamlit-in-Snowflake, return the
    injected Snowpark session instead (``get_active_session()``), and change the
    ONE line in ``run_query`` noted there (``session.sql(sql).to_pandas()``).
    ``run_query`` already lowercases columns, so the rest of the app is unchanged.
    See ``snowflake/DEPLOY.md`` §4. The SQL is plain ``analytics.*`` and runs on
    both engines.
    """
    import duckdb

    # read_only=True: the app can never mutate the warehouse.
    return duckdb.connect(str(DUCKDB_PATH), read_only=True)


def run_query(sql: str) -> pd.DataFrame:
    """Run read-only SQL and return a DataFrame with lower-case column names.

    On DuckDB ``.execute().fetchdf()`` already yields lower-case columns; on
    Snowflake, swap the body for ``df = _get_connection().sql(sql).to_pandas()``
    (which returns UPPER-CASE columns). The ``str.lower()`` normalization below
    makes both engines look identical to the loaders, so nothing downstream
    changes. This is the only line the Snowflake port touches.
    """
    con = _get_connection()
    df = con.execute(sql).fetchdf()
    df.columns = df.columns.str.lower()
    return df


def warehouse_exists() -> bool:
    return DUCKDB_PATH.is_file() and DUCKDB_PATH.stat().st_size > 0


def _pipeline_hint() -> None:
    st.warning(
        "**No warehouse found yet.** Build it first with one command, then reload "
        "this page:\n\n"
        "```bash\nmake reproduce\n# or, directly:\n"
        '.venv/bin/python -m finsentry.pipeline_runner\n```'
    )
    st.caption(f"Expected warehouse at: `{DUCKDB_PATH}`")


# --------------------------------------------------------------------------- #
# Cached data loaders (read-only; each returns a DataFrame).
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def load_health() -> pd.DataFrame:
    df = run_query(
        "SELECT date, total_billed_cost, recognized_revenue, recon_abs_diff, "
        "recon_pct_diff, total_consumption_units, total_active_accounts, "
        "recon_breach FROM analytics.mart_daily_health ORDER BY date"
    )
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data(show_spinner=False)
def load_revenue() -> pd.DataFrame:
    df = run_query(
        "SELECT date, total_billed_cost, recognized_revenue, recon_pct_diff, "
        "billing_present, ledger_present FROM analytics.fct_daily_revenue ORDER BY date"
    )
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data(show_spinner=False)
def load_consumption() -> pd.DataFrame:
    df = run_query(
        "SELECT date, total_consumption_units, total_active_accounts "
        "FROM analytics.fct_daily_consumption ORDER BY date"
    )
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data(show_spinner=False)
def load_failure_log() -> pd.DataFrame:
    df = run_query(
        "SELECT date, failure_id, failure_type, metric, detector, severity, "
        "value, expected, deviation, cause, is_known_label "
        "FROM analytics.dq_failure_log ORDER BY date"
    )
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data(show_spinner=False)
def load_labels() -> pd.DataFrame:
    df = run_query(
        "SELECT anomaly_date, metric, anomaly_type, severity, magnitude, note "
        "FROM analytics.stg_anomaly_labels ORDER BY anomaly_date"
    )
    df["anomaly_date"] = pd.to_datetime(df["anomaly_date"])
    return df


@st.cache_data(show_spinner=False)
def load_anomaly_eval() -> dict | None:
    import json

    p = REPORTS_DIR / "anomaly_eval.json"
    if not p.is_file():
        return None
    return json.loads(p.read_text())


@st.cache_data(show_spinner=False)
def load_weekly_verification() -> dict | None:
    import json

    p = REPORTS_DIR / "weekly_review_verification.json"
    if not p.is_file():
        return None
    return json.loads(p.read_text())


def _money(x: float) -> str:
    return f"${x:,.2f}"


def _fmt_int(x: float) -> str:
    return f"{x:,.0f}"


# --------------------------------------------------------------------------- #
# View 1: Data health
# --------------------------------------------------------------------------- #
def view_data_health() -> None:
    st.header("Data health")
    st.caption(
        "Overview of the `analytics.mart_daily_health` mart, dbt data-test status, "
        "reconciliation, and calendar coverage. Data is SYNTHETIC (seed 42)."
    )

    health = load_health()
    rev = load_revenue()

    # --- KPI tiles (full-year) ---
    total_billed = float(health["total_billed_cost"].sum())
    total_recognized = float(health["recognized_revenue"].sum())
    total_consumption = float(health["total_consumption_units"].sum())
    avg_accounts = float(health["total_active_accounts"].mean())
    n_days = int(len(health))
    n_breaches = int(health["recon_breach"].sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Billed cost (year)", _money(total_billed))
    c2.metric("Recognized revenue (year)", _money(total_recognized))
    c3.metric("Consumption units (year)", _fmt_int(total_consumption))
    c4.metric("Avg active accounts / day", f"{avg_accounts:,.2f}")

    # --- Coverage + reconciliation status ---
    missing_side = rev[(~rev["billing_present"]) | (~rev["ledger_present"])]
    missing_both = health[
        health["total_billed_cost"].isna() | health["total_consumption_units"].isna()
    ]

    s1, s2, s3 = st.columns(3)
    with s1:
        st.metric("Calendar days covered", f"{n_days}")
        st.caption("Expected 365 (2025-01-01 .. 2025-12-31).")
    with s2:
        if n_breaches == 0:
            st.metric("Reconciliation breaks", "0")
        else:
            st.metric("Reconciliation breaks", f"{n_breaches}", delta="investigate",
                      delta_color="inverse")
        st.caption("Billed vs recognized beyond 1% tolerance.")
    with s3:
        n_missing = int(len(missing_side)) + int(len(missing_both))
        st.metric("Days missing a source", f"{n_missing}")
        st.caption("Billing or ledger absent for a calendar day.")

    # --- dbt test summary (from the last run_results.json) ---
    st.subheader("dbt data tests")
    dbt_summary = _dbt_test_summary()
    if dbt_summary is None:
        st.info(
            "No dbt `run_results.json` found yet - run `make test` (or the full "
            "pipeline) to populate the dbt test summary."
        )
    else:
        n_total, n_pass, n_fail = dbt_summary
        d1, d2, d3 = st.columns(3)
        d1.metric("Data tests", f"{n_total}")
        d2.metric("Passed", f"{n_pass}")
        d3.metric("Failed", f"{n_fail}",
                  delta=None if n_fail == 0 else "see dbt_test stage",
                  delta_color="inverse")
        if n_fail == 0:
            st.success(
                f"All {n_total} dbt tests passed (uniqueness, not-null, "
                "referential integrity, cross-source reconciliation, missing-source-day)."
            )
        else:
            st.error(f"{n_fail} dbt test(s) failing - see the runbook, section 4.")

    # --- Reconciliation detail table ---
    st.subheader("Reconciliation status")
    breaches = health[health["recon_breach"]].copy()
    if breaches.empty:
        st.success("No reconciliation breaks in the current warehouse.")
    else:
        show = breaches[["date", "total_billed_cost", "recognized_revenue",
                         "recon_abs_diff", "recon_pct_diff"]].copy()
        show["date"] = show["date"].dt.date
        show["recon_pct_diff"] = (show["recon_pct_diff"] * 100).round(2).astype(str) + "%"
        show = show.rename(columns={
            "total_billed_cost": "billed_cost",
            "recognized_revenue": "recognized_revenue",
            "recon_abs_diff": "abs_diff",
            "recon_pct_diff": "pct_diff",
        })
        st.dataframe(show, use_container_width=True, hide_index=True)
        st.caption(
            "Breaks are caught by the dbt `assert_revenue_reconciliation` singular "
            "test and `mart_daily_health.recon_breach`."
        )

    # --- Revenue vs recognized overview chart (two series, one axis) ---
    st.subheader("Daily billed cost vs recognized revenue")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=health["date"], y=health["total_billed_cost"], name="Billed cost",
        mode="lines", line=dict(color=C_SERIES, width=1.5),
        hovertemplate="%{x|%Y-%m-%d}<br>Billed $%{y:,.0f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=health["date"], y=health["recognized_revenue"], name="Recognized revenue",
        mode="lines", line=dict(color=C_RECON, width=1.5),
        hovertemplate="%{x|%Y-%m-%d}<br>Recognized $%{y:,.0f}<extra></extra>",
    ))
    if not breaches.empty:
        fig.add_trace(go.Scatter(
            x=breaches["date"], y=breaches["total_billed_cost"], name="Recon break",
            mode="markers", marker=dict(color=C_BAD, size=11, symbol="x",
                                        line=dict(width=1)),
            hovertemplate="%{x|%Y-%m-%d}<br>recon break<extra></extra>",
        ))
    _style(fig, "USD")
    st.plotly_chart(fig, use_container_width=True)


# --------------------------------------------------------------------------- #
# View 2: Anomalies
# --------------------------------------------------------------------------- #
def view_anomalies() -> None:
    st.header("Anomalies")
    st.caption(
        "Statistical detections (robust median/MAD + Isolation Forest) and the "
        "unified `analytics.dq_failure_log`, measured against the planted labels."
    )

    failures = load_failure_log()
    labels = load_labels()
    rev = load_revenue()
    cons = load_consumption()
    ev = load_anomaly_eval()

    # --- Precision / recall summary from the real eval ---
    if ev is not None:
        st.subheader("Detector precision / recall / F1 (vs labeled anomalies)")
        rows = []
        for key in ("mad", "iforest", "union"):
            d = ev["detectors"][key]
            rows.append({
                "detector": key,
                "TP": d["tp"], "FP": d["fp"], "FN": d["fn"],
                "precision": round(d["precision"], 4),
                "recall": round(d["recall"], 4),
                "F1": round(d["f1"], 4),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        recon = ev.get("reconciliation", {})
        st.caption(
            f"Reconciliation breaks (separate rule): "
            f"{recon.get('detected_breaks', '?')}/{recon.get('labeled_breaks', '?')} "
            f"detected (recall {recon.get('recall', 0):.0%}). "
            "The statistical MAD detector leads on F1; Isolation Forest underperforms "
            "on this univariate deseasonalized signal."
        )
        pr_png = ASSETS_DIR / "precision_recall.png"
        if pr_png.is_file():
            st.image(str(pr_png),
                     caption="Committed figure (assets/precision_recall.png)")
        else:
            st.info(
                "Committed figure `assets/precision_recall.png` not found - run "
                "`make figures` (or `make reproduce`) to generate it."
            )
    else:
        st.info("Run the pipeline to populate `reports/anomaly_eval.json`.")

    # --- Filters ---
    st.subheader("DQ failure log")
    f1, f2, f3 = st.columns(3)
    metric_opts = ["(all)"] + sorted(failures["metric"].dropna().unique().tolist())
    sev_opts = ["(all)"] + sorted(failures["severity"].dropna().unique().tolist())
    type_opts = ["(all)"] + sorted(failures["failure_type"].dropna().unique().tolist())
    sel_metric = f1.selectbox("Metric", metric_opts)
    sel_sev = f2.selectbox("Severity", sev_opts)
    sel_type = f3.selectbox("Failure type", type_opts)

    view = failures.copy()
    if sel_metric != "(all)":
        view = view[view["metric"] == sel_metric]
    if sel_sev != "(all)":
        view = view[view["severity"] == sel_sev]
    if sel_type != "(all)":
        view = view[view["failure_type"] == sel_type]

    k1, k2, k3 = st.columns(3)
    k1.metric("Rows (filtered)", f"{len(view)}")
    k2.metric("Known labels", f"{int(view['is_known_label'].sum())}")
    k3.metric("Suspected false positives", f"{int((~view['is_known_label']).sum())}")

    disp = view.copy()
    disp["date"] = disp["date"].dt.date
    disp["deviation"] = (disp["deviation"] * 100).round(2).astype(str) + "%"
    st.dataframe(
        disp[["date", "failure_type", "metric", "detector", "severity", "value",
              "expected", "deviation", "is_known_label", "cause"]],
        use_container_width=True, hide_index=True,
    )

    # --- Time series with flagged points ---
    st.subheader("Time series with flagged points")
    metric_choice = st.radio("Series", ["revenue", "consumption"], horizontal=True)
    if metric_choice == "revenue":
        series = rev.rename(columns={"total_billed_cost": "value"})[["date", "value"]]
        ylab = "total billed cost (USD)"
    else:
        series = cons.rename(columns={"total_consumption_units": "value"})[["date", "value"]]
        ylab = "total consumption units"

    flagged = failures[
        (failures["failure_type"] == f"{metric_choice}_anomaly")
    ][["date"]].drop_duplicates()
    flagged_pts = series.merge(flagged, on="date", how="inner")

    truth = labels[(labels["metric"] == metric_choice)
                   & (labels["anomaly_type"] != "recon_break")]
    truth_pts = series.merge(
        truth.rename(columns={"anomaly_date": "date"})[["date"]],
        on="date", how="inner",
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=series["date"], y=series["value"], name=f"daily {metric_choice}",
        mode="lines", line=dict(color=C_SERIES, width=1.2),
        hovertemplate="%{x|%Y-%m-%d}<br>%{y:,.0f}<extra></extra>",
    ))
    if not truth_pts.empty:
        fig.add_trace(go.Scatter(
            x=truth_pts["date"], y=truth_pts["value"], name="labeled truth",
            mode="markers",
            marker=dict(color="rgba(0,0,0,0)", size=14, symbol="triangle-up",
                        line=dict(color=C_TRUTH, width=2)),
            hovertemplate="%{x|%Y-%m-%d}<br>labeled<extra></extra>",
        ))
    if not flagged_pts.empty:
        fig.add_trace(go.Scatter(
            x=flagged_pts["date"], y=flagged_pts["value"], name="detector flagged",
            mode="markers", marker=dict(color=C_MAD, size=8, symbol="circle"),
            hovertemplate="%{x|%Y-%m-%d}<br>flagged<extra></extra>",
        ))
    _style(fig, ylab)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Green triangles = planted ground-truth labels; red dots = days written to "
        "the DQ failure log (detector-flagged). Precision < 1 is by design - the "
        "unlabeled confounders (holiday dips, shock days) keep the problem honest."
    )


# --------------------------------------------------------------------------- #
# View 3: Weekly revenue review
# --------------------------------------------------------------------------- #
def view_weekly_review() -> None:
    st.header("Weekly revenue review")
    st.caption(
        "The AI analyst's weekly business review. Every number is independently "
        "re-queried and checked by `verify_numbers.py`."
    )

    md_path = REPORTS_DIR / "weekly_review.md"
    verification = load_weekly_verification()

    if verification is not None:
        v1, v2, v3 = st.columns(3)
        v1.metric("Numbers checked", f"{verification.get('n_numbers_checked', '?')}")
        v2.metric("Matched", f"{verification.get('n_match', '?')}")
        n_mis = verification.get("n_mismatch", 0)
        v3.metric("Mismatched", f"{n_mis}",
                  delta=None if n_mis == 0 else "FAIL", delta_color="inverse")
        if verification.get("overall_pass"):
            st.success(
                f"ALL-PASS - every number verified ({verification.get('mode')}, "
                f"render {verification.get('render_mode')})."
            )
        else:
            st.error("Number verification did NOT pass - do not trust the review.")

        # Key numbers table from the verified claims.
        checks = verification.get("checks", [])
        if checks:
            with st.expander("Verified key numbers", expanded=True):
                kn = pd.DataFrame([
                    {"claim": str(c.get("label", c.get("id", ""))),
                     "kind": str(c.get("kind", "")),
                     "value in review": str(c.get("claim", "")),
                     "re-queried": str(c.get("expected", "")),
                     "tolerance": str(c.get("tolerance", "")),
                     "match": "yes" if c.get("match") else "no"}
                    for c in checks
                ])
                st.dataframe(kn, use_container_width=True, hide_index=True)
    else:
        st.info("Run the pipeline to populate `reports/weekly_review_verification.json`.")

    st.subheader("Review")
    if md_path.is_file():
        st.markdown(_render_markdown(md_path.read_text()), unsafe_allow_html=False)
    else:
        st.info(
            "`reports/weekly_review.md` not found yet - run `make reproduce` to "
            "generate the weekly review."
        )


_CLAIM_MARKER_RE = re.compile(r"<!--c:[^>]*-->")


def _render_markdown(text: str) -> str:
    """Make the on-disk review safe to display, without touching the file.

    1. Strip the hidden ``<!--c:ID-->`` claim markers so they do not show as
       literal text (the on-disk file keeps them - ``verify_numbers.py`` binds
       each number to its claim via those markers).
    2. Escape ``$`` as ``\\$`` so Streamlit's Markdown does not read ``$...$``
       dollar amounts as LaTeX math (e.g. the revenue paragraph).
    """
    cleaned = _CLAIM_MARKER_RE.sub("", text)
    cleaned = cleaned.replace("$", r"\$")
    return cleaned


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _style(fig: go.Figure, ylab: str) -> None:
    """Apply the shared, accessible chart style (one axis, clear labels)."""
    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0),
        hovermode="x unified",
        xaxis_title="date",
        yaxis_title=ylab,
        plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.08)")


def _dbt_test_summary() -> tuple[int, int, int] | None:
    """Return (total, pass, fail) from dbt's last ``run_results.json`` (tests only)."""
    import json

    p = REPO_ROOT / "dbt" / "finsentry" / "target" / "run_results.json"
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text())
    except Exception:
        return None
    tests = [r for r in data.get("results", [])
             if r.get("unique_id", "").startswith("test.")]
    if not tests:
        return None
    n_pass = sum(1 for r in tests if r.get("status") == "pass")
    n_fail = len(tests) - n_pass
    return len(tests), n_pass, n_fail


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    st.title("FinSentry - finance data-quality monitoring & AI analyst")
    st.caption(
        "SYNTHETIC data (documented assumptions + planted labeled anomalies). "
        "Local DuckDB; Snowflake-portable (Streamlit-in-Snowflake) - see "
        "`snowflake/DEPLOY.md`."
    )

    if not warehouse_exists():
        _pipeline_hint()
        return

    tab1, tab2, tab3 = st.tabs(["Data health", "Anomalies", "Weekly revenue review"])
    with tab1:
        try:
            view_data_health()
        except Exception as exc:  # friendly, never a raw traceback for the user
            st.error(f"Could not render Data health: {exc}")
            _pipeline_hint()
    with tab2:
        try:
            view_anomalies()
        except Exception as exc:
            st.error(f"Could not render Anomalies: {exc}")
            _pipeline_hint()
    with tab3:
        try:
            view_weekly_review()
        except Exception as exc:
            st.error(f"Could not render Weekly review: {exc}")
            _pipeline_hint()

    st.divider()
    st.caption(
        f"Read-only view of `{DUCKDB_PATH.name}` (schema `analytics`). "
        f"Rendered {dt.date.today().isoformat()}. No data is modified by this app."
    )


if __name__ == "__main__":
    main()
else:
    # `streamlit run` imports the module and executes top-level code.
    main()
