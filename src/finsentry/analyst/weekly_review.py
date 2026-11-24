"""Weekly business review - driven by the analyst agent, grounded in real tools.

Picks a deterministic review week from config (default ISO week 2025-12-01 ..
2025-12-07, which contains a +12% reconciliation break on 2025-12-03) and its
prior 7-day window, drives the agent through grounded sub-questions, and writes:

* ``reports/weekly_review.md``          - the narrative write-up, and
* ``reports/weekly_review_claims.json`` - a machine-readable manifest of every
  number in the review, each tied to the REAL tool result it came from (consumed
  by ``verify_numbers.py``).

Two honest render modes (``verify_numbers.py`` guards both):

* **offline provider (default, committed path):** the review is
  **template-assembled** from the real tool-result values and is deterministic /
  byte-reproducible. Every number carries a hidden ``<!--c:ID-->`` marker binding
  it to its claim id, so verification can confirm each prose number matches the
  re-queried value for THAT specific claim (not just "some" claim).
* **groq provider (opt-in):** the Groq model **writes the narrative prose itself**
  from the same grounded facts (the real "AI analyst writes the review"). No
  markers (free text); ``verify_numbers.py`` falls back to exact-string
  presence/coverage. Any shape/empty/error in the live path degrades cleanly to
  the offline template - it never crashes.

Grounded numbers always come from tool results; a malformed/empty tool result is
back-filled from a direct independent query (producer-side safety net) so the
manifest is always grounded. Tool calls are logged to
``reports/analyst/tool_calls.jsonl``.

Entry point::

    python -m finsentry.analyst.weekly_review [--provider {offline,groq}]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from typing import Any

import duckdb

from finsentry.analyst.agent import JsonlLogger, run_agent
from finsentry.analyst.llm import select_provider
from finsentry.analyst.planner import fmt_int, fmt_money, fmt_num
from finsentry.config import Config, load_config


# --------------------------------------------------------------------------- #
# Safe extraction of tool-result values
# --------------------------------------------------------------------------- #
def _scalar_safe(result: dict[str, Any] | None) -> float | None:
    if isinstance(result, dict) and result.get("ok") and result.get("rows"):
        try:
            return float(result["rows"][0][0])
        except (TypeError, ValueError, IndexError):
            return None
    return None


def _rows_safe(result: dict[str, Any] | None) -> list[list[Any]]:
    if isinstance(result, dict) and result.get("ok") and isinstance(result.get("rows"), list):
        return result["rows"]
    return []


def _first_result(agent_result: Any) -> dict[str, Any] | None:
    tr = getattr(agent_result, "tool_results", None)
    if tr:
        return tr[0]
    return None


def _direct_week_values(cfg: Config, wss: str, wes: str, pss: str, pes: str) -> dict[str, Any]:
    """Producer-side independent back-fill (direct SQL) for any metric a tool
    result failed to provide. Keeps the manifest grounded and never crashes."""
    con = duckdb.connect(str(cfg.duckdb_path), read_only=True, config={"enable_external_access": "false"})
    try:
        one = lambda sql: con.execute(sql).fetchone()
        return {
            "rev_week": float(one(f"SELECT sum(total_billed_cost) FROM analytics.fct_daily_revenue WHERE date BETWEEN '{wss}' AND '{wes}'")[0]),
            "rev_prior": float(one(f"SELECT sum(total_billed_cost) FROM analytics.fct_daily_revenue WHERE date BETWEEN '{pss}' AND '{pes}'")[0]),
            "recognized_week": float(one(f"SELECT sum(recognized_revenue) FROM analytics.fct_daily_revenue WHERE date BETWEEN '{wss}' AND '{wes}'")[0]),
            "cons_week": float(one(f"SELECT sum(total_consumption_units) FROM analytics.fct_daily_consumption WHERE date BETWEEN '{wss}' AND '{wes}'")[0]),
            "cons_prior": float(one(f"SELECT sum(total_consumption_units) FROM analytics.fct_daily_consumption WHERE date BETWEEN '{pss}' AND '{pes}'")[0]),
            "avg_active_week": float(one(f"SELECT avg(total_active_accounts) FROM analytics.fct_daily_consumption WHERE date BETWEEN '{wss}' AND '{wes}'")[0]),
        }
    finally:
        con.close()


# --------------------------------------------------------------------------- #
# Groq free-text narrative (opt-in; degrades cleanly)
# --------------------------------------------------------------------------- #
def _groq_narrative(provider: Any, facts: str) -> str | None:
    """Ask the Groq model to write the review prose from grounded facts.

    Returns the prose, or ``None`` on any error/empty response so the caller can
    fall back to the deterministic offline template.
    """
    try:
        resp = provider.client.chat.completions.create(
            model=provider.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a finance analyst writing a weekly business review. Use ONLY "
                        "the facts provided - do not invent any number. Write 2-4 short "
                        "paragraphs covering revenue vs prior week, consumption and active "
                        "accounts, reconciliation status, and anomalies. Keep every figure "
                        "EXACTLY as given (same digits, $ and % formatting). CRITICAL: "
                        "immediately after EACH number, copy its claim marker exactly as shown "
                        "in the facts (e.g. write `$809222.86<!--c:rev_week-->`). Do not include "
                        "any number that does not have a marker, and do not alter a marker."
                    ),
                },
                {"role": "user", "content": facts},
            ],
            temperature=0.0,
            seed=0,
            max_tokens=700,
        )
        text = resp.choices[0].message.content
        if not (isinstance(text, str) and text.strip()):
            return None
        # Fail closed: a groq review with no markers cannot be verified -> reject it.
        return text if "<!--c:" in text else None
    except Exception:
        return None


def build_review(
    cfg: Config | None = None,
    provider_name: str | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run the agent over the review week and write the markdown + claims manifest."""
    cfg = cfg or load_config()
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    acfg = cfg.raw.get("analyst", {})

    ws = dt.date.fromisoformat(str(acfg.get("review_week_start", "2025-12-01")))
    we = dt.date.fromisoformat(str(acfg.get("review_week_end", "2025-12-07")))
    window_days = (we - ws).days + 1
    ps = ws - dt.timedelta(days=window_days)
    pe = we - dt.timedelta(days=window_days)
    week_num = ws.isocalendar().week
    run_ts = str(acfg.get("run_ts", ""))
    max_iter = int(acfg.get("max_iterations", 6))

    default_provider = str(acfg.get("provider", "offline"))
    provider, resolved, notes = select_provider(cfg, provider_name or default_provider, verbose=verbose)

    log_path = cfg.reports_dir / "analyst" / "tool_calls.jsonl"
    logger = JsonlLogger(log_path, reset=True)

    wss, wes, pss, pes = ws.isoformat(), we.isoformat(), ps.isoformat(), pe.isoformat()

    def ask(q: str, rid: str):
        return run_agent(q, cfg, provider, run_id=rid, max_iterations=max_iter, logger=logger, run_ts=run_ts)

    # --- Grounded sub-questions (numbers below come from these tool results) ---
    r_rev = ask(f"What was the total billed cost between {wss} and {wes}?", "wr_rev_week")
    r_rev_p = ask(f"What was the total billed cost between {pss} and {pes}?", "wr_rev_prior")
    r_rec = ask(f"What was the total recognized revenue between {wss} and {wes}?", "wr_recognized_week")
    r_cons = ask(f"What were the total consumption units between {wss} and {wes}?", "wr_cons_week")
    r_cons_p = ask(f"What were the total consumption units between {pss} and {pes}?", "wr_cons_prior")
    r_acct = ask(f"What was the average daily active accounts between {wss} and {wes}?", "wr_active_week")
    r_recon = ask(f"List reconciliation breaks between {wss} and {wes}.", "wr_recon_list")
    r_anom = ask("How many anomalies did the anomaly detector flag?", "wr_anomaly")

    # Primary: real tool-result values. Safety net: direct query for any that are
    # missing/malformed (keeps the manifest grounded, never crashes on shape).
    vals = {
        "rev_week": _scalar_safe(_first_result(r_rev)),
        "rev_prior": _scalar_safe(_first_result(r_rev_p)),
        "recognized_week": _scalar_safe(_first_result(r_rec)),
        "cons_week": _scalar_safe(_first_result(r_cons)),
        "cons_prior": _scalar_safe(_first_result(r_cons_p)),
        "avg_active_week": _scalar_safe(_first_result(r_acct)),
    }
    if any(v is None for v in vals.values()):
        fallback = _direct_week_values(cfg, wss, wes, pss, pes)
        used_fallback = [k for k, v in vals.items() if v is None]
        for k, v in vals.items():
            if v is None:
                vals[k] = fallback[k]
        notes.append(f"tool-result back-fill used for: {', '.join(used_fallback)}")

    rev_week, rev_prior = vals["rev_week"], vals["rev_prior"]
    recognized_week, cons_week, cons_prior = vals["recognized_week"], vals["cons_week"], vals["cons_prior"]
    avg_active_week = vals["avg_active_week"]
    rev_wow = (rev_week - rev_prior) / rev_prior * 100.0 if rev_prior else 0.0
    cons_wow = (cons_week - cons_prior) / cons_prior * 100.0 if cons_prior else 0.0

    # Reconciliation breaks in the week (rows: date, abs, pct); robust to shape.
    recon_rows = _rows_safe(_first_result(r_recon))
    recon_breaks_week = len(recon_rows)
    recon_break_date = recon_break_abs = recon_break_pct = None
    if recon_rows:
        try:
            top = max(recon_rows, key=lambda r: abs(float(r[1])))
            recon_break_date = str(top[0])
            recon_break_abs = float(top[1])
            recon_break_pct = float(top[2]) * 100.0
        except (TypeError, ValueError, IndexError):
            recon_break_date = recon_break_abs = recon_break_pct = None

    # Anomaly-days flagged within the review week (from the anomaly-check result).
    anom_res = _first_result(r_anom) or {}
    flagged = anom_res.get("flagged", []) if isinstance(anom_res, dict) and anom_res.get("ok") else []
    week_flagged = sorted(f["date"] for f in flagged if isinstance(f, dict) and wss <= f.get("date", "") <= wes)
    anomalies_week = len(week_flagged)

    # --- Claims manifest (id -> label, kind, formatted text_value) ---
    claims: list[dict[str, Any]] = [
        {"id": "week_start", "label": "Review week start", "kind": "date", "text_value": wss},
        {"id": "week_end", "label": "Review week end", "kind": "date", "text_value": wes},
        {"id": "prior_start", "label": "Prior week start", "kind": "date", "text_value": pss},
        {"id": "prior_end", "label": "Prior week end", "kind": "date", "text_value": pes},
        {"id": "week_num", "label": "ISO week number", "kind": "count", "text_value": fmt_int(week_num)},
        {"id": "window_days", "label": "Window length (days)", "kind": "count", "text_value": fmt_int(window_days)},
        {"id": "rev_week", "label": "Billed cost (review week)", "kind": "money", "text_value": fmt_money(rev_week)},
        {"id": "rev_prior", "label": "Billed cost (prior week)", "kind": "money", "text_value": fmt_money(rev_prior)},
        {"id": "rev_wow_pct", "label": "Billed cost WoW %", "kind": "pct", "text_value": f"{rev_wow:+.2f}%"},
        {"id": "recognized_week", "label": "Recognized revenue (review week)", "kind": "money", "text_value": fmt_money(recognized_week)},
        {"id": "cons_week", "label": "Consumption units (review week)", "kind": "number", "text_value": fmt_num(cons_week)},
        {"id": "cons_prior", "label": "Consumption units (prior week)", "kind": "number", "text_value": fmt_num(cons_prior)},
        {"id": "cons_wow_pct", "label": "Consumption WoW %", "kind": "pct", "text_value": f"{cons_wow:+.2f}%"},
        {"id": "avg_active_week", "label": "Avg daily active accounts (review week)", "kind": "number", "text_value": fmt_num(avg_active_week)},
        {"id": "recon_breaks_week", "label": "Reconciliation breaks (review week)", "kind": "count", "text_value": fmt_int(recon_breaks_week)},
        {"id": "anomalies_week", "label": "Revenue/consumption anomaly-days (review week)", "kind": "count", "text_value": fmt_int(anomalies_week)},
    ]
    if recon_break_date is not None:
        claims += [
            {"id": "recon_break_date", "label": "Largest reconciliation break date", "kind": "date", "text_value": recon_break_date},
            {"id": "recon_break_abs", "label": "Largest reconciliation break (USD)", "kind": "money", "text_value": fmt_money(recon_break_abs)},
            {"id": "recon_break_pct", "label": "Largest reconciliation break (%)", "kind": "pct", "text_value": f"{recon_break_pct:.2f}%"},
        ]
    tv = {c["id"]: c["text_value"] for c in claims}

    # --- Decide render mode first ---
    render_mode = "offline-template"
    groq_prose: str | None = None
    if resolved == "groq":
        # The groq prompt REQUIRES the model to emit the same <!--c:ID--> markers,
        # so the free-text review is verified in the SAME marker-bound mode as
        # offline. A review with no markers is rejected -> offline fallback.
        groq_prose = _groq_narrative(provider, _facts_block(tv, recon_break_date))
        if groq_prose is not None:
            render_mode = "groq-markerbound"
        else:
            notes.append("groq narrative unavailable/unmarked -> fell back to the offline template")

    def m(cid: str) -> str:
        """Value text followed by its hidden claim marker."""
        return f"{tv[cid]}<!--c:{cid}-->"

    header = (
        f"# FinSentry Weekly Business Review\n\n"
        f"**Review week:** {m('week_start')} to {m('week_end')} "
        f"(ISO week {m('week_num')}, {m('window_days')} days)  \n"
        f"**Comparison window (prior week):** {m('prior_start')} to {m('prior_end')}  \n"
        f"**Provider:** `{resolved}` ({render_mode})  \n"
        f"**Generated by:** the FinSentry AI analyst agent - every figure below is drawn from a real tool query.\n\n"
        f"> SYNTHETIC data. All figures come from the analyst's read-only SQL / anomaly-check tools\n"
        f"> and are independently re-checked by `verify_numbers.py`. Numbers use plain decimals\n"
        f"> (money `$NNNN.NN`, no thousands separators; percentages `NN.NN%`; dates `YYYY-MM-DD`).\n"
        f"> Each number carries a hidden HTML-comment claim marker binding it to its claim.\n\n"
    )

    if render_mode == "groq-markerbound":
        body = groq_prose
    else:
        body = _offline_body(tv, m, rev_wow, cons_wow, recon_breaks_week, recon_break_date, week_flagged, anomalies_week)

    md = header + body
    md_path = cfg.reports_dir / "weekly_review.md"
    md_path.write_text(md, encoding="utf-8")

    # Record the deterministic claim-id sequence in DOCUMENT order (offline only,
    # where the producer controls the whole document) so verification can detect a
    # value-and-marker pair swap that keeps each pair internally consistent. The
    # groq narrative orders prose freely -> no order check (binding catches swaps).
    # Number-bound markers only (same pattern verify_numbers uses), so an unrelated
    # comment is never counted.
    _BOUND = r"(?:\$?[+-]?\d[\d,]*(?:\.\d+)?%?|\d{4}-\d{2}-\d{2})<!--c:(\w+)-->"
    if render_mode == "groq-markerbound":
        order_for_manifest: list[str] | None = None
    else:
        order_for_manifest = re.findall(_BOUND, md)

    manifest = {
        "review_week_start": wss,
        "review_week_end": wes,
        "prior_week_start": pss,
        "prior_week_end": pes,
        "provider": resolved,
        "render_mode": render_mode,
        "marker_order": order_for_manifest,
        "provider_notes": notes,
        "run_ts": run_ts,
        "number_format": "plain decimals; money=$NNNN.NN (no thousands separators); pct=NN.NN%; dates=YYYY-MM-DD",
        "claims": claims,
    }
    claims_path = cfg.reports_dir / "weekly_review_claims.json"
    with open(claims_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
        fh.write("\n")

    if verbose:
        print(f"  weekly review ({render_mode}) -> {md_path.relative_to(cfg.project_root)}", flush=True)
        print(f"  claims manifest ({len(claims)} numbers) -> {claims_path.relative_to(cfg.project_root)}", flush=True)
        print(f"  tool calls logged -> {log_path.relative_to(cfg.project_root)}", flush=True)

    return {"md_path": str(md_path), "claims_path": str(claims_path), "n_claims": len(claims),
            "provider": resolved, "render_mode": render_mode}


# Human labels for the facts block handed to the Groq model.
_FACT_LABELS = {
    "week_start": "Review week start", "week_end": "Review week end",
    "prior_start": "Prior week start", "prior_end": "Prior week end",
    "week_num": "ISO week number", "window_days": "Window length (days)",
    "rev_week": "Billed cost this week", "rev_prior": "Billed cost prior week",
    "rev_wow_pct": "Billed cost WoW %", "recognized_week": "Recognized revenue this week",
    "cons_week": "Consumption units this week", "cons_prior": "Consumption units prior week",
    "cons_wow_pct": "Consumption WoW %", "avg_active_week": "Avg daily active accounts",
    "recon_breaks_week": "Reconciliation breaks this week", "anomalies_week": "Anomaly-days this week",
    "recon_break_date": "Largest reconciliation break date",
    "recon_break_abs": "Largest reconciliation break (USD)",
    "recon_break_pct": "Largest reconciliation break (%)",
}


def _facts_block(tv: dict[str, str], recon_break_date: str | None) -> str:
    """Grounded facts + the exact marked form to copy, handed to the Groq model."""
    lines = [
        "Grounded facts for the weekly review. Each fact lists its label, its exact",
        "value, and the EXACT marked text you must copy verbatim wherever you state",
        "that number (value immediately followed by its <!--c:ID--> marker):",
        "",
    ]
    for cid, value in tv.items():
        label = _FACT_LABELS.get(cid, cid)
        lines.append(f"- {label}: value={value}  -> copy verbatim: {value}<!--c:{cid}-->")
    return "\n".join(lines)


def _offline_body(
    tv: dict[str, str],
    m,
    rev_wow: float,
    cons_wow: float,
    recon_breaks_week: int,
    recon_break_date: str | None,
    week_flagged: list[str],
    anomalies_week: int,
) -> str:
    """Deterministic, marker-bound narrative body (committed path)."""
    rev_dir = "up" if rev_wow >= 0 else "down"
    cons_dir = "up" if cons_wow >= 0 else "down"
    recon_status = (
        f"**{m('recon_breaks_week')} reconciliation break** this week"
        if recon_breaks_week == 1 else
        f"**{m('recon_breaks_week')} reconciliation breaks** this week"
    )
    if recon_break_date is not None:
        recon_detail = (
            f" The break on {m('recon_break_date')} was {m('recon_break_abs')} "
            f"({m('recon_break_pct')}): the general-ledger recognized revenue diverged from billed "
            f"cost beyond the reconciliation tolerance and should be investigated."
        )
    else:
        recon_detail = " Billing and the revenue ledger reconciled within tolerance on every day of the week."

    if anomalies_week == 0:
        anomaly_line = (
            f"The statistical anomaly detector flagged **{m('anomalies_week')}** revenue/consumption "
            f"anomaly-days during the review week."
        )
    else:
        anomaly_line = (
            f"The statistical anomaly detector flagged **{m('anomalies_week')}** revenue/consumption "
            f"anomaly-days during the review week (dates: {', '.join(week_flagged)})."
        )

    return (
        "## Revenue\n\n"
        f"Billed cost for the review week was **{m('rev_week')}**, {rev_dir} **{m('rev_wow_pct')}** "
        f"versus the prior week ({m('rev_prior')}). Recognized revenue (general ledger) for the week "
        f"was {m('recognized_week')}.\n\n"
        "## Consumption and accounts\n\n"
        f"Total consumption units were **{m('cons_week')}**, {cons_dir} **{m('cons_wow_pct')}** versus "
        f"the prior week ({m('cons_prior')}). The platform averaged **{m('avg_active_week')}** active "
        f"accounts per day over the week.\n\n"
        "## Data quality and reconciliation\n\n"
        f"There was {recon_status}.{recon_detail}\n\n"
        f"{anomaly_line}\n\n"
        "## Narrative\n\n"
        f"Week {m('week_num')} showed billed cost {rev_dir} {m('rev_wow_pct')} week-over-week to "
        f"{m('rev_week')}, with consumption moving {cons_dir} {m('cons_wow_pct')} to {m('cons_week')} "
        f"units - demand and billing moved together, a healthy sign. The one open data-quality item is "
        f"the reconciliation break noted above; aside from that the week's finance data is clean and the "
        f"revenue ledger tracks billed cost. No action is required on the revenue or consumption series.\n"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate the FinSentry weekly business review.")
    parser.add_argument("--provider", choices=["offline", "groq"], default=None)
    args = parser.parse_args(argv)
    try:
        build_review(provider_name=args.provider, verbose=True)
    except Exception as exc:
        print(f"ERROR: weekly review failed: {exc!r}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
