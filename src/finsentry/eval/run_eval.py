"""Run the 15-question analyst eval; report accuracy + failure cases.

Drives all 15 ``questions.yaml`` questions through the agent (offline by default,
deterministic), compares each answer to its INDEPENDENT known value, logs every
tool call to ``reports/eval_tool_calls.jsonl``, and writes
``reports/eval_results.json`` + ``reports/eval_results.md``.

Matching is tolerant where appropriate (relative tolerance for money/floats) and
exact for counts/dates; categorical answers use case-insensitive substring. The
offline provider is deterministic, so results are identical run-to-run.

Legitimately failing questions are reported as failure cases (they do NOT crash
the run). A non-zero exit is reserved for infrastructure errors (e.g. missing
warehouse), so the pipeline can tell "some answers wrong" from "eval broke".

Entry point::

    python -m finsentry.eval.run_eval [--provider {offline,groq}]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import yaml

from finsentry.analyst.agent import JsonlLogger, run_agent
from finsentry.analyst.llm import select_provider
from finsentry.analyst.planner import UNSUPPORTED_PREFIX
from finsentry.config import Config, load_config

_QUESTIONS_PATH = Path(__file__).resolve().parent / "questions.yaml"
_PROBES_PATH = Path(__file__).resolve().parent / "probes.yaml"
_NUM_RE = re.compile(r"-?\$?\d[\d,]*(?:\.\d+)?")
_DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")

# Anomaly questions whose ground truth must be recomputed via the real detector
# (detect_all), not SQL. Maps question id -> metric for the drift check.
_ANOMALY_GT = {"q08": "revenue", "q09": "consumption"}
_SERVICES = {"compute", "storage", "database", "network", "analytics"}


def load_questions() -> list[dict[str, Any]]:
    with open(_QUESTIONS_PATH, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    qs = data["questions"]
    if len(qs) != 15:
        raise ValueError(f"expected exactly 15 eval questions, found {len(qs)}")
    return qs


def load_probes() -> list[dict[str, Any]]:
    if not _PROBES_PATH.is_file():
        return []
    with open(_PROBES_PATH, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh).get("probes", [])


_MONEY_RE = re.compile(r"\$-?\d[\d,]*(?:\.\d+)?")


def extract_numbers(text: str) -> list[float]:
    """Pull every numeric token (handles $ and thousands commas) from text.

    ISO dates are stripped first so their components (2025, 03, 19) are not read
    as numbers.
    """
    out: list[float] = []
    for tok in _NUM_RE.findall(_DATE_RE.sub(" ", text or "")):
        cleaned = tok.replace("$", "").replace(",", "")
        try:
            out.append(float(cleaned))
        except ValueError:
            continue
    return out


def extract_money(text: str) -> list[float]:
    """Pull only $-prefixed money tokens, in order (so a bare year like '2025' in
    'Q3 2025' is not mistaken for the leading monetary value)."""
    out: list[float] = []
    for tok in _MONEY_RE.findall(text or ""):
        try:
            out.append(float(tok.replace("$", "").replace(",", "")))
        except ValueError:
            continue
    return out


def _num_matches(candidates: list[float], expected: float, rel_tol: float) -> float | None:
    tol = max(abs(expected) * rel_tol, 0.01)
    for c in candidates:
        if abs(c - expected) <= tol:
            return c
    return None


def grade(question: dict[str, Any], answer: str, tool_results: list[dict[str, Any]], cfg: Config) -> dict[str, Any]:
    """Return a grading record: passed + the matched 'got' value + reason."""
    mtype = question["match"]
    expected = question.get("expected")
    rel_tol = float(question.get("tolerance", 0.001))
    nums = extract_numbers(answer)
    passed = False
    got: Any = None
    reason = ""

    if mtype in ("money", "number"):
        # Tightened: the value must be the LEADING value (rejects padding like
        # "$1.00, $3247663.99, ..."). For money, use the leading $-token so a bare
        # year (e.g. "Q3 2025") is not mistaken for the amount.
        candidates = extract_money(answer) if mtype == "money" else nums
        if not candidates:  # money phrased without $ -> fall back to all numbers
            candidates = nums
        first = candidates[0] if candidates else None
        tol = max(abs(float(expected)) * rel_tol, 0.01)
        passed = first is not None and abs(first - float(expected)) <= tol
        got = first
        reason = "" if passed else f"expected {expected} as the leading value (tol {rel_tol}); saw {candidates}"

    elif mtype == "count":
        # Tightened: the count must be the SALIENT answer (first number stated) and
        # must not be negated ("There were not 1 but 2 breaks.").
        exp_i = int(expected)
        first = nums[0] if nums else None
        negated = re.search(rf"\bnot\s+{exp_i}\b", answer or "") is not None
        passed = (first is not None) and (abs(first - exp_i) < 1e-9) and not negated
        got = first
        if negated:
            reason = f"count {exp_i} is negated in the answer"
        elif not passed:
            reason = f"expected count {exp_i} as the leading number; saw {nums}"

    elif mtype == "date":
        # Tightened: the expected date must be the FIRST date in the answer (the
        # salient one), not merely a loose substring somewhere in the text.
        dates = _DATE_RE.findall(answer or "")
        first = dates[0] if dates else None
        passed = first == str(expected)
        got = first
        reason = "" if passed else f"expected date {expected} as the leading date; saw {dates}"

    elif mtype == "categorical":
        # Tightened: expected must be the LEADING value of its type (not buried in
        # a list) and must not be negated ("was not compute; it was storage").
        exp = str(expected)
        low = exp.lower()
        ans = answer or ""
        al = ans.lower()
        if re.search(r"\bnot\s+" + re.escape(low), al):
            passed, got, reason = False, None, f"expected '{exp}' is negated in the answer"
        else:
            if re.fullmatch(r"\d{4}-\d{2}", exp):
                cands = re.findall(r"\d{4}-\d{2}(?!-\d)", ans)  # YYYY-MM, not a full date
            elif low in _SERVICES:
                cands = [w for w in re.findall(r"[a-z]+", al) if w in _SERVICES]
            else:
                cands = None
            if cands is not None:
                passed = bool(cands) and cands[0].lower() == low
                reason = "" if passed else f"expected '{exp}' is not the leading value; saw {cands}"
            else:
                idx = al.find(low)
                passed = idx != -1 and al[:idx].count(",") < 1
                reason = "" if passed else f"expected '{exp}' not the sole/leading value"
            got = exp if passed else None

    elif mtype == "chart":
        chart_ok = False
        chart_path = None
        for r in tool_results:
            if r.get("ok") and "path" in r:
                p = Path(r["path"])
                full = p if p.is_absolute() else cfg.project_root / p
                if full.is_file() and full.stat().st_size > 0:
                    chart_ok = True
                    chart_path = r["path"]
        num_ok = _num_matches(nums, float(expected), rel_tol)
        passed = bool(chart_ok and num_ok is not None)
        got = {"chart": chart_path, "value": num_ok}
        if not chart_ok:
            reason = "no non-empty chart PNG was produced"
        elif num_ok is None:
            reason = f"chart produced but underlying value {expected} not stated (saw {nums})"
    else:
        reason = f"unknown match type '{mtype}'"

    return {"passed": passed, "got": got, "reason": reason}


def run_eval(
    cfg: Config | None = None,
    provider_name: str | None = None,
    verbose: bool = True,
    check_ground_truth: bool = False,
) -> dict[str, Any]:
    """Run all 15 questions; write reports; return a summary dict."""
    cfg = cfg or load_config()
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    questions = load_questions()

    default_provider = str(cfg.raw.get("analyst", {}).get("provider", "offline"))
    max_iter = int(cfg.raw.get("analyst", {}).get("max_iterations", 6))
    run_ts = str(cfg.raw.get("analyst", {}).get("run_ts", ""))
    provider, resolved, notes = select_provider(
        cfg, provider_name or default_provider, verbose=verbose
    )

    log_path = cfg.reports_dir / "eval_tool_calls.jsonl"
    logger = JsonlLogger(log_path, reset=True)

    # Optional: re-confirm every stored expected value against a fresh, independent
    # recomputation (ground-truth SQL for SQL facts; the real detector for the
    # anomaly questions). Off by default so the committed numbers are the source of
    # truth; when on, ANY drift makes `main` exit non-zero (regression guard).
    gt_drift: list[str] = []
    if check_ground_truth:
        from finsentry.analyst.tools import run_sql

        # detect_all anomaly counts (independent recompute for q08/q09).
        anomaly_counts: dict[str, int] = {}
        if any(q["id"] in _ANOMALY_GT for q in questions):
            from finsentry.anomaly import detect_all

            det = detect_all(cfg)
            anomaly_counts = (
                det[det["any_flag"]].groupby("metric").size().astype(int).to_dict()
            )

        for q in questions:
            qid = q["id"]
            if qid in _ANOMALY_GT:  # q08/q09 -> recompute via the detector
                metric = _ANOMALY_GT[qid]
                val = int(anomaly_counts.get(metric, -1))
                if val != int(q["expected"]):
                    gt_drift.append(f"{qid}: stored {q['expected']} != detector count {val} ({metric})")
                continue
            sql = q.get("ground_truth_sql")
            if not sql:
                gt_drift.append(f"{qid}: no ground_truth_sql and no recompute rule - cannot verify")
                continue
            res = run_sql(sql, cfg=cfg)
            if not (res.get("ok") and res.get("rows")):
                gt_drift.append(f"{qid}: ground-truth SQL returned no result ({res.get('error')})")
                continue
            val = res["rows"][0][0]
            if q["match"] in ("money", "number", "chart"):
                if abs(float(val) - float(q["expected"])) > max(abs(float(q["expected"])) * 1e-3, 0.01):
                    gt_drift.append(f"{qid}: stored {q['expected']} != ground-truth {val}")
            elif q["match"] == "count":
                if int(val) != int(q["expected"]):
                    gt_drift.append(f"{qid}: stored {q['expected']} != ground-truth {val}")
            else:
                if str(val) != str(q["expected"]):
                    gt_drift.append(f"{qid}: stored {q['expected']} != ground-truth {val}")

    records: list[dict[str, Any]] = []
    n_pass = 0
    for q in questions:
        result = run_agent(
            q["question"], cfg, provider,
            run_id=q["id"], max_iterations=max_iter, logger=logger, run_ts=run_ts,
        )
        g = grade(q, result.answer, result.tool_results, cfg)
        if g["passed"]:
            n_pass += 1
        records.append(
            {
                "id": q["id"],
                "question": q["question"],
                "match": q["match"],
                "expected": q.get("expected"),
                "got": g["got"],
                "answer": result.answer,
                "passed": g["passed"],
                "reason": g["reason"],
                "tools_used": result.tools_used,
                "iterations": result.iterations,
            }
        )
        if verbose:
            mark = "PASS" if g["passed"] else "FAIL"
            print(f"  [{mark}] {q['id']}: {result.answer}", flush=True)
            if not g["passed"]:
                print(f"         reason: {g['reason']}", flush=True)

    # --- Adversarial / held-out probes: prove the planner declines or correctly
    #     handles reworded / re-dimensioned questions (scope + failure cases). ---
    probe_records: list[dict[str, Any]] = []
    for p in load_probes():
        result = run_agent(
            p["question"], cfg, provider,
            run_id=p["id"], max_iterations=max_iter, logger=logger, run_ts=run_ts,
        )
        declined = result.answer.strip().startswith(UNSUPPORTED_PREFIX)
        if p["expect"] == "decline":
            passed = declined
            reason = "" if passed else "expected an explicit decline but got an answer"
            got = "declined" if declined else result.answer
        else:  # expect a supported value
            g = grade(p, result.answer, result.tool_results, cfg)
            passed, got, reason = g["passed"], g["got"], g["reason"]
            if declined:
                passed, reason = False, "declined a question it should support"
        probe_records.append({
            "id": p["id"], "question": p["question"], "expect": p["expect"],
            "expected": p.get("expected"), "got": got, "answer": result.answer,
            "passed": passed, "reason": reason, "note": p.get("note", ""),
            "tools_used": result.tools_used,
        })
        if verbose:
            mark = "PASS" if passed else "FAIL"
            print(f"  [probe {mark}] {p['id']}: {result.answer}", flush=True)

    accuracy = n_pass / len(questions)
    n_tool_calls = sum(len(r["tools_used"]) for r in records) + sum(len(r["tools_used"]) for r in probe_records)
    n_probe_pass = sum(1 for r in probe_records if r["passed"])
    summary = {
        "provider": resolved,
        "provider_notes": notes,
        "ground_truth_drift": gt_drift,
        "n_questions": len(questions),
        "n_pass": n_pass,
        "n_fail": len(questions) - n_pass,
        "accuracy": round(accuracy, 4),
        "n_tool_calls_logged": n_tool_calls,
        "tool_call_log": str(log_path.relative_to(cfg.project_root)),
        "results": records,
        "probes": probe_records,
        "n_probe_pass": n_probe_pass,
        "n_probes": len(probe_records),
    }

    _write_reports(summary, cfg)
    if verbose:
        print(
            f"\n  core accuracy: {n_pass}/{len(questions)} = {accuracy:.1%} "
            f"| probes: {n_probe_pass}/{len(probe_records)} behaved correctly "
            f"| tool calls logged: {n_tool_calls} -> {log_path.name}",
            flush=True,
        )
        failures = [r for r in records if not r["passed"]]
        if failures:
            print("  core failure cases:", flush=True)
            for r in failures:
                print(f"    - {r['id']}: {r['reason']}", flush=True)
        if gt_drift:
            print("  GROUND-TRUTH DRIFT:", flush=True)
            for d in gt_drift:
                print(f"    - {d}", flush=True)
    return summary


def _write_reports(summary: dict[str, Any], cfg: Config) -> None:
    json_path = cfg.reports_dir / "eval_results.json"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)
        fh.write("\n")

    lines: list[str] = []
    lines.append("# FinSentry analyst eval - 15 questions\n")
    lines.append(
        f"**Provider:** `{summary['provider']}` (deterministic offline = committed numbers)  "
    )
    lines.append(
        f"**Accuracy:** {summary['n_pass']}/{summary['n_questions']} "
        f"= {summary['accuracy']:.1%}  "
    )
    lines.append(f"**Tool calls logged:** {summary['n_tool_calls_logged']} "
                 f"-> `{summary['tool_call_log']}`\n")
    if summary["provider_notes"]:
        lines.append("> Provider notes: " + "; ".join(summary["provider_notes"]) + "\n")
    if summary["ground_truth_drift"]:
        lines.append("> **Ground-truth drift detected:** " + "; ".join(summary["ground_truth_drift"]) + "\n")

    lines.append("| # | Question | Expected | Got | Pass | Tools |")
    lines.append("|---|----------|----------|-----|------|-------|")
    for r in summary["results"]:
        got = r["got"]
        got_s = json.dumps(got) if isinstance(got, dict) else str(got)
        q = r["question"].replace("|", "\\|")
        mark = "✅" if r["passed"] else "❌"
        tools = ", ".join(r["tools_used"])
        lines.append(
            f"| {r['id']} | {q} | {r['expected']} | {got_s} | {mark} | {tools} |"
        )

    # --- Scope, limitations & failure cases (honest about the planner's design) ---
    lines.append("\n## Scope, limitations & failure cases\n")
    lines.append(
        "The committed numbers use the **deterministic offline planner**, which uses a "
        "**positive, fail-closed gate**: it answers only when the ENTIRE question is "
        "accounted for by recognized structure (a known metric/subject + a supported "
        "aggregate + a period/threshold/service it fully parses). It drops stopwords, "
        "then subtracts the tokens explained by the matched structure; if ANY content "
        "token or phrase is left unexplained, it DECLINES (`UNSUPPORTED: ...`) rather "
        "than guess. Periods that use ranges or relative wording it cannot parse exactly "
        "(before/after/since/through/YTD/half/H1-H2/from-to/non-ISO between) decline the "
        "whole question. Out-of-scope shapes - set operations (combined/only/excluding), "
        "per-X and region breakdowns, order statistics (median/percentile), weekday/"
        "quarter grouping, Nth-highest, growth/delta, dollar thresholds on a %-field, a "
        "period filter on the whole-series anomaly detector, yes/no questions, "
        "multi-superlative conjunctions ('highest revenue AND most active accounts'), a "
        "value/count conditioned on an extreme day ('X on the highest Y day'), and other "
        "cross-subject conditions - all decline. The core 15 fall inside the supported set, so they pass; each "
        "expected value is checked independently (`--check-ground-truth`). The open-"
        "vocabulary path is the live **Groq** provider (same tools, same loop), whose "
        "output is guarded number-by-number by `verify_numbers.py`.\n"
    )
    core_failures = [r for r in summary["results"] if not r["passed"]]
    lines.append("**Core failure cases:** " + ("none (all 15 passed)." if not core_failures else "") )
    for r in core_failures:
        lines.append(f"\n- **{r['id']}** - {r['question']}")
        lines.append(f"  - expected `{r['expected']}`, answer: {r['answer']}")
        lines.append(f"  - reason: {r['reason']}")
    if summary["ground_truth_drift"]:
        lines.append("\n**Ground-truth drift (independent recompute):**")
        for d in summary["ground_truth_drift"]:
            lines.append(f"- {d}")

    probes = summary.get("probes", [])
    if probes:
        lines.append(
            f"\n### Adversarial / held-out probes ({summary['n_probe_pass']}/"
            f"{summary['n_probes']} behaved correctly)\n"
        )
        lines.append(
            "Each probe reworded, re-filtered, or re-dimensioned a core question. "
            "'decline' = the analyst must refuse; 'value' = a shape it genuinely "
            "supports.\n"
        )
        lines.append("| # | Probe | Expect | Analyst answer | OK |")
        lines.append("|---|-------|--------|----------------|----|")
        for r in probes:
            qn = r["question"].replace("|", "\\|")
            ans = r["answer"].replace("|", "\\|")
            if len(ans) > 90:
                ans = ans[:90] + "..."
            mark = "✅" if r["passed"] else "❌"
            lines.append(f"| {r['id']} | {qn} | {r['expect']} | {ans} | {mark} |")
        lines.append(
            "\n_These probes document the regex-planner's scope: it correctly "
            "declines region/median/per-account questions (instead of returning a "
            "misleading total) and correctly handles supported reword/filter shapes "
            "(per-service totals, directional reconciliation thresholds, 'peak' = "
            "highest)._\n"
        )

    md_path = cfg.reports_dir / "eval_results.md"
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the FinSentry 15-question analyst eval.")
    parser.add_argument("--provider", choices=["offline", "groq"], default=None,
                        help="Override provider selection (default: config / GROQ_API_KEY).")
    parser.add_argument("--check-ground-truth", action="store_true",
                        help="Re-confirm stored expected values against fresh ground-truth SQL.")
    args = parser.parse_args(argv)
    try:
        summary = run_eval(
            provider_name=args.provider, verbose=True, check_ground_truth=args.check_ground_truth
        )
    except Exception as exc:  # infrastructure error -> non-zero exit
        print(f"ERROR: eval failed to run: {exc!r}", flush=True)
        return 1
    # Legitimately-failing eval questions are reported, not fatal. But a
    # ground-truth drift under --check-ground-truth is a real regression -> fail.
    if args.check_ground_truth and summary.get("ground_truth_drift"):
        print("FAIL: ground-truth drift detected (see above).", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
