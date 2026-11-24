"""Number verification - the guardrail that grounds the weekly review in data.

Every number in ``reports/weekly_review.md`` is INDEPENDENTLY re-queried from the
warehouse (fresh read-only SQL on a new connection / the real detector) and
compared to the review. It never reuses the analyst's cached tool results and
never trusts the review's own text for the expected value.

Verification is **marker-bound and provider-independent**: both the offline
template AND the groq narrative must carry a ``<!--c:ID-->`` marker immediately
after every number (the groq prompt requires it; an unmarked groq review is
rejected and falls back to the offline template). Four things must all hold:

1. **Per-claim check** - for each claim in ``weekly_review_claims.json`` the
   value written in the review is compared to an independently recomputed
   expected value, tolerant by kind (exact for counts/dates, small relative
   tolerance for money, absolute for %).
2. **Binding check** - each number is tied to a SPECIFIC claim by its marker; the
   bound token must equal that claim's exact ``text_value`` string, every claim
   must be bound, and every marker id must be a real claim. A value swap (marker
   says ``rev_prior`` but the value is the current week's) FAILS here.
3. **Order check** - the sequence of marker ids must match the template's
   recorded ``marker_order``. This runs for the **offline** review, where the
   whole document is deterministic, so a value-and-marker pair swap that stays
   internally consistent still FAILS. It is **skipped for the groq narrative**,
   which orders prose freely.
4. **Coverage check** - after removing the bound numbers (and markers), no
   numeric token or ISO date may remain. A number with no valid marker (a
   fabricated/hallucinated figure) is UNCOVERED and FAILS.

What this does and does NOT guarantee (no over-claiming):
* The **offline template path** - which produces the committed, reproducible
  results - is fully guarded: per-claim recompute + exact binding + marker ORDER
  + coverage, all fail-closed. A wrong value, a value swap, a marker-order/pair
  swap, or an injected number all FAIL.
* The **groq free-text path** verifies that each marked number equals its claim's
  independently recomputed value, that every claim is present, and that no
  unmarked digit-form number appears. It does NOT verify the semantic placement
  of the surrounding prose (a pair swap where each number still equals its own
  swapped claim's value, with no fixed order, is not caught), and it does not
  scan spelled-out numbers. Treat the offline path as the authoritative,
  committed guardrail; the groq path is a strong but not exhaustive check.

Writes ``reports/weekly_review_verification.md`` + ``.json`` and returns overall
pass/fail. On the deterministic offline review this is ALL-PASS.

Number-format contract: plain decimals; money ``$NNNN.NN`` (no thousands
separators); percentages ``NN.NN%``; dates ``YYYY-MM-DD``.

Entry point::

    python -m finsentry.analyst.verify_numbers
"""

from __future__ import annotations

import datetime as dt
import json
import re
from pathlib import Path
from typing import Any

import duckdb

from finsentry.config import Config, load_config

_DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_NUM_RE = re.compile(r"-?\$?\d[\d,]*(?:\.\d+)?")
# A money/number/pct/date token immediately followed by its claim marker.
_BOUND_RE = re.compile(
    r"(\$?[+-]?\d[\d,]*(?:\.\d+)?%?|\d{4}-\d{2}-\d{2})<!--c:(\w+)-->"
)
_MARKER_RE = re.compile(r"<!--c:\w+-->")


# --------------------------------------------------------------------------- #
# Parsing helpers
# --------------------------------------------------------------------------- #
def parse_number(text_value: str) -> float:
    """Parse a formatted claim ('$809222.86', '+6.08%', '3445.29') to a float."""
    cleaned = str(text_value).replace("$", "").replace("%", "").replace(",", "").replace("+", "").strip()
    return float(cleaned)


def _tolerance_for(kind: str, cfg_tol: dict[str, Any]) -> tuple[str, float, float]:
    """Return (description, rel, abs) tolerance spec for a claim kind."""
    money_rel = float(cfg_tol.get("money_rel", 0.001))
    money_abs = float(cfg_tol.get("money_abs", 0.05))
    pct_abs = float(cfg_tol.get("pct_abs", 0.05))
    if kind in ("money", "number"):
        return (f"rel {money_rel} / abs {money_abs}", money_rel, money_abs)
    if kind == "pct":
        return (f"abs {pct_abs} pts", 0.0, pct_abs)
    if kind in ("count", "date"):
        return ("exact", 0.0, 0.0)
    return ("exact", 0.0, 0.0)


def _num_match(claim_val: float, expected: float, rel: float, abs_: float) -> bool:
    return abs(claim_val - expected) <= max(abs(expected) * rel, abs_)


# --------------------------------------------------------------------------- #
# Independent re-computation of each claim (fresh SQL / real detector)
# --------------------------------------------------------------------------- #
def _recompute_expected(cfg: Config) -> dict[str, Any]:
    """Recompute every claim id independently from the warehouse + config.

    Uses the review week from CONFIG (not from the review text) and a fresh
    read-only DuckDB connection. Returns ``{claim_id: expected_value}`` where
    values are floats, ints, or ISO date strings.
    """
    acfg = cfg.raw.get("analyst", {})
    ws = dt.date.fromisoformat(str(acfg.get("review_week_start", "2025-12-01")))
    we = dt.date.fromisoformat(str(acfg.get("review_week_end", "2025-12-07")))
    window_days = (we - ws).days + 1
    ps = ws - dt.timedelta(days=window_days)
    pe = we - dt.timedelta(days=window_days)
    wss, wes, pss, pes = ws.isoformat(), we.isoformat(), ps.isoformat(), pe.isoformat()

    con = duckdb.connect(str(cfg.duckdb_path), read_only=True)
    try:
        def q1(sql: str) -> Any:
            row = con.execute(sql).fetchone()
            return row[0] if row else None

        rev_week = float(q1(f"SELECT sum(total_billed_cost) FROM analytics.fct_daily_revenue WHERE date BETWEEN '{wss}' AND '{wes}'"))
        rev_prior = float(q1(f"SELECT sum(total_billed_cost) FROM analytics.fct_daily_revenue WHERE date BETWEEN '{pss}' AND '{pes}'"))
        recognized_week = float(q1(f"SELECT sum(recognized_revenue) FROM analytics.fct_daily_revenue WHERE date BETWEEN '{wss}' AND '{wes}'"))
        cons_week = float(q1(f"SELECT sum(total_consumption_units) FROM analytics.fct_daily_consumption WHERE date BETWEEN '{wss}' AND '{wes}'"))
        cons_prior = float(q1(f"SELECT sum(total_consumption_units) FROM analytics.fct_daily_consumption WHERE date BETWEEN '{pss}' AND '{pes}'"))
        avg_active_week = float(q1(f"SELECT avg(total_active_accounts) FROM analytics.fct_daily_consumption WHERE date BETWEEN '{wss}' AND '{wes}'"))
        recon_breaks_week = int(q1(f"SELECT count(*) FROM analytics.mart_daily_health WHERE recon_breach AND date BETWEEN '{wss}' AND '{wes}'"))

        recon_row = con.execute(
            f"SELECT date, recon_abs_diff, recon_pct_diff FROM analytics.mart_daily_health "
            f"WHERE recon_breach AND date BETWEEN '{wss}' AND '{wes}' ORDER BY recon_abs_diff DESC LIMIT 1"
        ).fetchone()
    finally:
        con.close()

    # anomaly-days flagged within the week - via the REAL detector (independent).
    from finsentry.anomaly import detect_all

    det = detect_all(cfg)
    det_week = det[det["any_flag"]]
    anomalies_week = int(
        sum(1 for d in det_week["date"] if wss <= (d.date().isoformat() if hasattr(d, "date") else str(d)) <= wes)
    )

    expected: dict[str, Any] = {
        "week_start": wss,
        "week_end": wes,
        "prior_start": pss,
        "prior_end": pes,
        "week_num": int(ws.isocalendar().week),
        "window_days": int(window_days),
        "rev_week": round(rev_week, 2),
        "rev_prior": round(rev_prior, 2),
        "rev_wow_pct": round((rev_week - rev_prior) / rev_prior * 100.0, 2) if rev_prior else 0.0,
        "recognized_week": round(recognized_week, 2),
        "cons_week": round(cons_week, 2),
        "cons_prior": round(cons_prior, 2),
        "cons_wow_pct": round((cons_week - cons_prior) / cons_prior * 100.0, 2) if cons_prior else 0.0,
        "avg_active_week": round(avg_active_week, 2),
        "recon_breaks_week": recon_breaks_week,
        "anomalies_week": anomalies_week,
    }
    if recon_row is not None:
        expected["recon_break_date"] = str(recon_row[0])
        expected["recon_break_abs"] = round(float(recon_row[1]), 2)
        expected["recon_break_pct"] = round(float(recon_row[2]) * 100.0, 2)
    return expected


# --------------------------------------------------------------------------- #
# Verification
# --------------------------------------------------------------------------- #
def verify(cfg: Config | None = None, verbose: bool = True) -> dict[str, Any]:
    """Verify every number in weekly_review.md; write reports; return summary."""
    cfg = cfg or load_config()
    md_path = cfg.reports_dir / "weekly_review.md"
    claims_path = cfg.reports_dir / "weekly_review_claims.json"
    if not md_path.is_file() or not claims_path.is_file():
        raise FileNotFoundError(
            "weekly_review.md / weekly_review_claims.json not found - run "
            "`python -m finsentry.analyst.weekly_review` first."
        )

    md_text = md_path.read_text(encoding="utf-8")
    manifest = json.loads(claims_path.read_text(encoding="utf-8"))
    claims = manifest["claims"]
    cfg_tol = cfg.raw.get("analyst", {}).get("tolerances", {})

    expected_map = _recompute_expected(cfg)

    # --- Pass 1: per-claim check (claim-as-written vs independent recompute) ---
    checks: list[dict[str, Any]] = []
    for c in claims:
        cid, kind, text_value = c["id"], c["kind"], c["text_value"]
        tol_desc, rel, abs_ = _tolerance_for(kind, cfg_tol)
        exp = expected_map.get(cid)
        if exp is None:
            checks.append({**c, "expected": None, "match": False, "tolerance": tol_desc,
                           "detail": "no independent recompute available for this claim id"})
            continue
        if kind == "date":
            match = str(text_value) == str(exp)
            exp_out: Any = exp
        else:
            claim_val = parse_number(text_value)
            if kind in ("count",):
                match = int(round(claim_val)) == int(exp)
            else:
                match = _num_match(claim_val, float(exp), rel, abs_)
            exp_out = exp
        checks.append({
            "id": cid, "label": c["label"], "kind": kind,
            "claim": text_value, "expected": exp_out, "match": bool(match),
            "tolerance": tol_desc,
        })

    claims_by_id = {c["id"]: c for c in claims}
    # Verification is ALWAYS marker-bound and provider-independent: both the
    # offline template and the groq narrative must carry <!--c:ID--> markers.
    render_mode = manifest.get("render_mode", "")
    mode = "marker-bound"

    binding_errors: list[str] = []
    order_errors: list[str] = []
    uncovered: list[dict[str, Any]] = []

    bound = _BOUND_RE.findall(md_text)
    if not bound:
        binding_errors.append("review contains no claim markers - cannot verify (fail closed)")

    # --- Binding: each marked token must equal its OWN claim's text_value ---
    bound_ids = set()
    for token, cid in bound:
        if cid not in claims_by_id:
            binding_errors.append(f"marker c:{cid} is not a known claim")
            continue
        bound_ids.add(cid)
        if token != claims_by_id[cid]["text_value"]:
            binding_errors.append(
                f"c:{cid}: prose '{token}' != claim value '{claims_by_id[cid]['text_value']}'"
            )
    for c in claims:
        if c["id"] not in bound_ids:
            binding_errors.append(f"claim {c['id']} ({c['text_value']}) is not bound anywhere in the review")

    # --- Order: the sequence of marker ids must match the template's (offline). ---
    expected_order = manifest.get("marker_order")
    actual_order = [cid for _tok, cid in bound]
    if expected_order is not None and actual_order != expected_order:
        order_errors.append(
            f"marker sequence {actual_order} != expected {expected_order} (a value/marker swap)"
        )

    # --- Coverage: remove bound tokens+markers; nothing numeric may remain ---
    stripped = _BOUND_RE.sub(" ", md_text)
    stripped = _MARKER_RE.sub(" ", stripped)
    for d in _DATE_RE.findall(stripped):
        uncovered.append({"token": d, "kind": "date", "note": "date not bound to a claim"})
    for tok in _NUM_RE.findall(_DATE_RE.sub(" ", stripped)):
        uncovered.append({"token": tok, "kind": "number", "note": "number not bound to a claim"})

    n_claims = len(checks)
    n_match = sum(1 for c in checks if c["match"])
    all_claims_ok = n_match == n_claims
    binding_ok = len(binding_errors) == 0
    order_ok = len(order_errors) == 0
    coverage_ok = len(uncovered) == 0
    overall_pass = all_claims_ok and binding_ok and order_ok and coverage_ok

    summary = {
        "weekly_review": str(md_path.relative_to(cfg.project_root)),
        "mode": mode,
        "render_mode": render_mode,
        "order_checked": expected_order is not None,
        "overall_pass": overall_pass,
        "n_numbers_checked": n_claims,
        "n_match": n_match,
        "n_mismatch": n_claims - n_match,
        "binding_ok": binding_ok,
        "binding_errors": binding_errors,
        "order_ok": order_ok,
        "order_errors": order_errors,
        "coverage_uncovered": uncovered,
        "tolerances": cfg_tol,
        "checks": checks,
    }
    _write_reports(summary, cfg)

    if verbose:
        status = "ALL-PASS" if overall_pass else "FAIL"
        print(f"  verification [{mode}, {render_mode or 'n/a'}]: {status} - "
              f"{n_match}/{n_claims} claims matched; {len(binding_errors)} binding error(s); "
              f"{len(order_errors)} order error(s); {len(uncovered)} uncovered token(s)", flush=True)
        for c in checks:
            if not c["match"]:
                print(f"    MISMATCH {c['id']}: claim={c['claim']} expected={c['expected']}", flush=True)
        for b in binding_errors:
            print(f"    BINDING {b}", flush=True)
        for o in order_errors:
            print(f"    ORDER {o}", flush=True)
        for u in uncovered:
            print(f"    UNCOVERED {u['kind']}: {u['token']}", flush=True)

    return summary


def _norm_num(text_value: str) -> float:
    """Normalise a formatted number to a float for exact-equality coverage."""
    return round(parse_number(text_value), 6)


def _write_reports(summary: dict[str, Any], cfg: Config) -> None:
    json_path = cfg.reports_dir / "weekly_review_verification.json"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)
        fh.write("\n")

    lines: list[str] = []
    lines.append("# Weekly review - number verification\n")
    status = "ALL-PASS ✅" if summary["overall_pass"] else "FAIL ❌"
    lines.append(f"**Result:** {status}  ")
    lines.append(f"**Mode:** `{summary['mode']}` (render: `{summary.get('render_mode') or 'n/a'}`, "
                 f"order check: {'on' if summary.get('order_checked') else 'off (free narrative)'})  ")
    lines.append(f"**Numbers checked:** {summary['n_numbers_checked']}  ")
    lines.append(f"**Matched:** {summary['n_match']}  |  **Mismatched:** {summary['n_mismatch']}  ")
    lines.append(f"**Binding errors:** {len(summary['binding_errors'])}  |  "
                 f"**Order errors:** {len(summary.get('order_errors', []))}  |  "
                 f"**Uncovered tokens:** {len(summary['coverage_uncovered'])}\n")
    lines.append(
        "Every number in `weekly_review.md` is independently re-queried from the warehouse "
        "(fresh read-only SQL / the real detector) and compared to the value written in the "
        "review. Each number is bound to a SPECIFIC claim via a `<!--c:ID-->` marker (required "
        "in both the offline and groq reviews): the bound token must equal that claim's exact "
        "value, every claim must be bound, the marker sequence must match the template order "
        "(offline), and no number may appear without a backing marker.\n"
    )
    lines.append("| Claim | Kind | In review | Re-queried | Match | Tolerance |")
    lines.append("|-------|------|-----------|-----------|-------|-----------|")
    for c in summary["checks"]:
        mark = "✅" if c["match"] else "❌"
        lines.append(
            f"| {c['label']} (`{c['id']}`) | {c['kind']} | {c['claim']} | "
            f"{c['expected']} | {mark} | {c['tolerance']} |"
        )
    if summary["binding_errors"]:
        lines.append("\n## Binding errors (number not tied to its own claim)\n")
        for b in summary["binding_errors"]:
            lines.append(f"- {b}")
    if summary.get("order_errors"):
        lines.append("\n## Order errors (marker sequence swapped)\n")
        for o in summary["order_errors"]:
            lines.append(f"- {o}")
    if summary["coverage_uncovered"]:
        lines.append("\n## Uncovered numbers (not bound to any claim)\n")
        for u in summary["coverage_uncovered"]:
            lines.append(f"- {u['kind']}: `{u['token']}` ({u.get('note', '')})")
    if not summary["binding_errors"] and not summary.get("order_errors") and not summary["coverage_uncovered"]:
        lines.append("\n_Every numeric token and date in the review is bound, in-order, to a verified claim._")

    md_path = cfg.reports_dir / "weekly_review_verification.md"
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    try:
        summary = verify(verbose=True)
    except Exception as exc:
        print(f"ERROR: verification failed to run: {exc!r}", flush=True)
        return 1
    return 0 if summary["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
