"""Deterministic rule/intent planner - the OFFLINE analyst "brain".

Given a natural-language question it selects which real tool(s) to call and with
what arguments (building real SQL from the question's intent + extracted
parameters), then composes the final answer text *from the real tool results*.
It never hardcodes an answer value and uses no randomness and no wall-clock, so
the weekly review and the 15-question eval are byte-reproducible.

**Positive, fail-closed gate.** The planner answers only when the ENTIRE question
is accounted for by recognized structure: a known metric/subject, a supported
aggregate, and (optionally) a period/threshold/service it FULLY parses. It
computes the question's content tokens (dropping stopwords/punctuation), subtracts
the tokens explained by the matched structure, and if ANY salient token or phrase
remains unexplained it DECLINES (``UNSUPPORTED: ...``). Periods that use ranges or
relative phrasing it cannot parse exactly (before/after/since/through/YTD/half/
from-to/non-ISO between) also decline. It never answers on leftover/unknown
phrasing - a confident wrong total is never emitted.

The live Groq provider does not use this module - it lets the LLM choose tool
calls - but it calls the identical real tools.
"""

from __future__ import annotations

import calendar
import datetime as dt
import re
from dataclasses import dataclass, field
from typing import Any, Callable

_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11,
    "december": 12,
}
_MONTH_ABBR = {m[:3]: n for m, n in _MONTHS.items()}
_SERVICES = ("compute", "storage", "database", "network", "analytics")

UNSUPPORTED_PREFIX = "UNSUPPORTED:"


@dataclass
class Plan:
    intent: str
    calls: list[tuple[str, dict[str, Any]]]
    answer: Callable[[list[dict[str, Any]]], str]
    consumed: set[str] = field(default_factory=set)


# --------------------------------------------------------------------------- #
# Vocabulary: the ONLY tokens the planner recognizes. Anything else -> decline.
# --------------------------------------------------------------------------- #
_STOPWORDS = {
    "what", "whats", "was", "were", "is", "are", "be", "been", "being", "the", "a",
    "an", "of", "in", "on", "for", "to", "and", "or", "did", "do", "does", "done",
    "had", "has", "have", "how", "which", "who", "whom", "that", "this", "these",
    "those", "it", "its", "their", "there", "here", "we", "our", "us", "you",
    "your", "i", "my", "me", "show", "give", "tell", "get", "report", "please",
    "value", "overall", "across", "during", "within", "throughout", "by", "at",
    "as", "with", "occurred", "occur", "happen", "happened", "recorded", "record",
    "logged", "observed", "seen", "taken", "than", "no", "were", "about", "s",
    "data", "warehouse", "year's",
}

_METRIC = {
    "billed", "billing", "bill", "cost", "costs", "revenue", "revenues", "spend",
    "spending", "consumption", "usage", "units", "unit", "active", "accounts",
    "account", "recognized", "recognised", "ledger", "reconciliation",
    "reconciliations", "recon", "breaks", "break", "breaches", "breach",
    "anomaly", "anomalies", "service", "services", "detector", "detect",
    "detected", "flag", "flagged", "flags",
}

_AGG = {
    "total", "totals", "sum", "summed", "average", "avg", "mean", "highest",
    "lowest", "max", "maximum", "min", "minimum", "peak", "largest", "smallest",
    "top", "bottom", "most", "least", "greatest", "biggest", "count", "counts",
    "number", "many", "distinct", "daily", "monthly", "day", "days", "date",
    "dates", "month", "months", "when", "trough", "list",
}

_CHART = {
    "chart", "charts", "plot", "plots", "graph", "create", "draw", "visualize",
    "visualise", "visualization", "visualisation", "render",
}

_THRESHOLD = {
    "under", "over", "above", "below", "exceed", "exceeded", "exceeds",
    "exceeding", "least", "most", "more", "less", "greater", "fewer", "exactly",
    "equal", "equals",
}

_SERVICE_TOKENS = set(_SERVICES)

# Always-recognized vocabulary (tokens are "explained" even if a given handler
# ignores them; semantic correctness is enforced by the handler-consume check).
_VOCAB = _METRIC | _AGG | _CHART | _THRESHOLD | _SERVICE_TOKENS

# Period phrasing the parser will NOT handle -> the whole question declines
# (returning a wrong window is worse than declining).
_PERIOD_REDFLAGS = (
    "before", "after", "since", "through", "thru", "til", "till", "until",
    "ytd", "year to date", "year-to-date", "to date", "trailing", "rolling",
    "past ", "prior", "previous", "earlier", "later", "onward", "half", " h1",
    " h2", "ending", "up to", "as of", "so far", "first half", "second half",
)

_TOKEN_RE = re.compile(
    r"\d{4}-\d{2}-\d{2}|\$\d[\d,]*(?:\.\d+)?|\d+(?:\.\d+)?%|\d+(?:\.\d+)?|[a-z][a-z']*"
)
_THRESHOLD_OP_PHRASES = (
    "at least", "at most", "no more than", "no less than", "or more", "or greater",
    "or less", "or fewer", "more than", "less than", "greater than", "fewer than",
    "under", "over", "above", "below", "exceed", "exceeded", "exceeds",
    "exceeding", "exactly", ">=", "<=", ">", "<", "=",
)


def _content_tokens(ql: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(ql) if t not in _STOPWORDS]


def _has_threshold_op(ql: str) -> bool:
    return any(p in ql for p in _THRESHOLD_OP_PHRASES)


def _named_service(ql: str) -> str | None:
    """A service used as a filter: 'for [the] <svc>' or '<svc> service'."""
    m = re.search(r"\bfor\s+(?:the\s+)?(compute|storage|database|network|analytics)\b", ql)
    if m:
        return m.group(1)
    m = re.search(r"\b(compute|storage|database|network|analytics)\s+service\b", ql)
    if m:
        return m.group(1)
    return None


# --------------------------------------------------------------------------- #
# Strict period parsing: returns (status, period, matched_text).
#   status: "ok" | "none" | "decline"
# --------------------------------------------------------------------------- #
def _last_day(year: int, month: int) -> int:
    return calendar.monthrange(year, month)[1]


_ORDINALS = {"first": 1, "second": 2, "third": 3, "fourth": 4}


def parse_period_strict(q: str, default_year: int) -> tuple[str, tuple[str, str, str] | None, str]:
    ql = q.lower()
    if any(rf in ql for rf in _PERIOD_REDFLAGS):
        return "decline", None, ""
    if re.search(r"\bfrom\b.*\bto\b", ql):
        return "decline", None, ""

    m = re.search(r"between\s+(\d{4}-\d{2}-\d{2})\s+and\s+(\d{4}-\d{2}-\d{2})", ql)
    if m:
        s, e = m.group(1), m.group(2)
        return "ok", (s, e, f"{s} to {e}"), m.group(0)
    if re.search(r"\bbetween\b", ql):
        return "decline", None, ""  # a range we do not parse exactly

    m = re.search(r"\bon\s+(\d{4}-\d{2}-\d{2})\b", ql)
    if m:
        d = m.group(1)
        return "ok", (d, d, d), m.group(0)
    isos = re.findall(r"\b\d{4}-\d{2}-\d{2}\b", ql)
    if len(isos) > 1:
        return "decline", None, ""
    if len(isos) == 1:
        d = isos[0]
        return "ok", (d, d, d), d

    # Quarter (ordinal or Q#).
    moq = re.search(r"\b(first|second|third|fourth)\s+quarter(?:\s+(?:of\s+)?(\d{4}))?", ql)
    mq = re.search(r"\bq([1-4])\b(?:\s+(?:of\s+)?(\d{4}))?", ql)

    months_found = {mon for name, mon in _MONTHS.items() if re.search(rf"\b{name}\b", ql)}
    if len(months_found) > 1:
        return "decline", None, ""

    if moq:
        qn = _ORDINALS[moq.group(1)]
        yr = int(moq.group(2)) if moq.group(2) else default_year
        sm = (qn - 1) * 3 + 1
        em = sm + 2
        return "ok", (f"{yr}-{sm:02d}-01", f"{yr}-{em:02d}-{_last_day(yr, em):02d}", f"Q{qn} {yr}"), moq.group(0)
    if mq:
        qn = int(mq.group(1))
        yr = int(mq.group(2)) if mq.group(2) else default_year
        sm = (qn - 1) * 3 + 1
        em = sm + 2
        return "ok", (f"{yr}-{sm:02d}-01", f"{yr}-{em:02d}-{_last_day(yr, em):02d}", f"Q{qn} {yr}"), mq.group(0)

    if len(months_found) == 1:
        mon = next(iter(months_found))
        name = [n for n, v in _MONTHS.items() if v == mon][0]
        mm = re.search(rf"\b{name}\b(?:\s+(\d{{4}}))?", ql)
        yr = int(mm.group(1)) if mm.group(1) else default_year
        return "ok", (f"{yr}-{mon:02d}-01", f"{yr}-{mon:02d}-{_last_day(yr, mon):02d}", f"{name.capitalize()} {yr}"), mm.group(0)

    # Whole-year phrases / bare year.
    yphrase = re.search(r"\b(whole|full|entire)\s+year\b|\bannual\b|\bfor the year\b", ql)
    yr_m = re.search(r"\b(20\d{2})\b", ql)
    if yphrase:
        yr = int(yr_m.group(1)) if yr_m else default_year
        return "ok", (f"{yr}-01-01", f"{yr}-12-31", str(yr)), (yphrase.group(0) + (" " + yr_m.group(1) if yr_m else ""))
    if yr_m:
        yr = int(yr_m.group(1))
        return "ok", (f"{yr}-01-01", f"{yr}-12-31", str(yr)), yr_m.group(0)

    return "none", None, ""


def _where_range(period: tuple[str, str, str] | None) -> str:
    if period is None:
        return ""
    s, e, _ = period
    return f"WHERE date BETWEEN '{s}' AND '{e}'"


# --------------------------------------------------------------------------- #
# Qualifier detection (for the handler-consume check among recognized tokens).
# --------------------------------------------------------------------------- #
_SUPERLATIVE_RE = re.compile(
    r"\b(highest|lowest|max|maximum|min|minimum|peak|largest|smallest|biggest|"
    r"most|least|best|worst|trough|greatest|top|bottom)\b"
)
_SELECTOR_RE = re.compile(r"\b(which|what)\s+(day|date|service|month)\b")
_GROUPING = ("monthly", "per month", "per-month", "by month", "each month",
             "per day", "per-day", "by day", "each day", "weekly", "per week")
_COMPARISON_PHRASES = ("at least", "at most", "no less than", "no more than",
                       "less than", "more than", "greater than", "greater",
                       "fewer than", "or more", "or less", "or greater", "or fewer")


def detect_qualifiers(q: str, default_year: int) -> set[str]:
    ql = q.lower()
    tags: set[str] = set()
    if re.search(r"\d+(?:\.\d+)?\s*%", ql) and _has_threshold_op(ql):
        tags.add("threshold")
    scrub = ql
    for ph in _COMPARISON_PHRASES:
        scrub = scrub.replace(ph, " ")
    if _SUPERLATIVE_RE.search(scrub):
        tags.add("superlative")
    if _SELECTOR_RE.search(ql) or "date of" in ql or "on which" in ql:
        tags.add("selector")
    if any(w in ql for w in _GROUPING):
        tags.add("grouping")
    if any(w in ql for w in ("average", "avg", "mean")):
        tags.add("avg")
    if parse_period_strict(q, default_year)[0] == "ok":
        tags.add("period")
    return tags


# --------------------------------------------------------------------------- #
# Formatting helpers
# --------------------------------------------------------------------------- #
def fmt_money(v: float) -> str:
    return f"${float(v):.2f}"


def fmt_num(v: float, decimals: int = 2) -> str:
    return f"{float(v):.{decimals}f}"


def fmt_int(v: Any) -> str:
    return str(int(round(float(v))))


def _scalar(result: dict[str, Any]) -> Any:
    if not result.get("ok") or not result.get("rows"):
        return None
    return result["rows"][0][0]


def _row(result: dict[str, Any]) -> list[Any]:
    if not result.get("ok") or not result.get("rows"):
        return []
    return result["rows"][0]


def _pct_threshold(q: str) -> tuple[str, float, str] | None:
    m = re.search(r"(\d+(?:\.\d+)?)\s*%", q)
    if not m:
        return None
    pct_str = m.group(1)
    frac = float(pct_str) / 100.0
    ql = q.lower()
    if any(w in ql for w in ("at least", "or more", "or greater", "no less than", ">=")):
        op = ">="
    elif any(w in ql for w in ("at most", "or less", "or fewer", "no more than", "<=")):
        op = "<="
    elif "exactly" in ql or "equal" in ql:
        op = "=="
    elif any(w in ql for w in ("under", "below", "less than", "fewer than", "smaller")):
        op = "<"
    elif any(w in ql for w in ("over", "above", "exceed", "exceeded", "exceeds", "exceeding", "more than", "greater than", "larger")):
        op = ">"
    else:
        return None
    return op, frac, pct_str


_OP_WORD = {">=": "of at least", ">": "exceeding", "<": "below", "<=": "of at most", "==": "of exactly"}


def _wrong_subject_for_total(ql: str) -> bool:
    return ("anomal" in ql) or ("recon" in ql)


def _is_count_intent(ql: str) -> bool:
    return any(w in ql for w in ("how many", "number of", "count of"))


# Cross-condition / form guards applied globally in route(): no supported question
# asks for a value/count CONDITIONED on an extreme day ("X on the highest Y day"),
# and none is a yes/no question.
_ON_THE_EXTREME_RE = re.compile(
    r"\bon\b[^.]*\b(highest|lowest|max|maximum|min|minimum|peak|largest|smallest|"
    r"greatest|biggest|top|bottom|most|least|fewest|worst|best)\b"
)
_YES_NO_RE = re.compile(r"^\s*(was|were|is|are|does|do|did|has|have|had)\b")
_REV_TERMS = ("revenue", "billed", "cost", "spend")
_CONS_TERMS = ("consumption", "usage", "units")


def _on_the_extreme(ql: str) -> bool:
    return bool(_ON_THE_EXTREME_RE.search(ql))


def _is_yes_no(ql: str) -> bool:
    return bool(_YES_NO_RE.match(ql))


_SUPERLATIVE_COUNT_RE = re.compile(
    r"\b(highest|lowest|max|maximum|min|minimum|most|least|fewest|top|bottom|peak|"
    r"largest|smallest|greatest|biggest|worst|best|trough)\b"
)


def _count_superlatives(ql: str) -> int:
    """Count superlative constructs (comparison phrases like 'at least' scrubbed).

    Two or more means a conjunction of extremes ('highest revenue AND most active
    accounts') that no single row/day can be verified to satisfy -> decline."""
    scrub = ql
    for ph in ("at least", "at most", "no less than", "no more than",
               "or more", "or less", "or greater", "or fewer"):
        scrub = scrub.replace(ph, " ")
    return len(_SUPERLATIVE_COUNT_RE.findall(scrub))


# --------------------------------------------------------------------------- #
# Handlers. Each sets `consumed` = qualifier tags it accounts for.
# --------------------------------------------------------------------------- #
def _h_chart(q: str, yr: int) -> Plan | None:
    ql = q.lower()
    if not any(w in ql for w in ("chart", "plot", "graph", "visualiz", "visualise", "draw", "render")):
        return None
    consumption = any(w in ql for w in ("consumption", "usage", "units"))
    col = "total_consumption_units" if consumption else "total_billed_cost"
    table = "fct_daily_consumption" if consumption else "fct_daily_revenue"
    metric = "consumption" if consumption else "revenue"
    period = parse_period_strict(q, yr)[1]
    where = _where_range(period)
    label = period[2] if period else str(yr)
    sql = f"SELECT date, {col} FROM analytics.{table} {where} ORDER BY date"
    title = f"Daily {metric} {label}"

    def answer(results: list[dict[str, Any]]) -> str:
        r = results[0]
        if not r.get("ok"):
            return f"Could not create the chart: {r.get('error')}"
        tot = fmt_money(r["y_sum"]) if not consumption else fmt_num(r["y_sum"])
        peak = fmt_money(r["y_max"]) if not consumption else fmt_num(r["y_max"])
        return (f"Chart created and saved to {r['path']} ({r['n_points']} days). "
                f"Total {metric} over {label} was {tot}; peak day value {peak}.")

    return Plan("chart", [("create_chart", {"title": title, "sql": sql, "x": "date", "y": col, "kind": "line"})],
                answer, consumed={"period"})


def _h_anomaly_count(q: str, yr: int) -> Plan | None:
    ql = q.lower()
    if "anomal" not in ql:
        return None
    if not any(w in ql for w in ("how many", "number", "count", "flag")):
        return None
    if "recon" in ql or "breach" in ql or "break" in ql:  # cross-subject -> decline
        return None
    metric = None
    if "revenue" in ql or "billed" in ql:
        metric = "revenue"
    elif "consumption" in ql or "usage" in ql or "units" in ql:
        metric = "consumption"
    args = {"metric": metric} if metric else {}

    def answer(results: list[dict[str, Any]]) -> str:
        r = results[0]
        if not r.get("ok"):
            return f"Anomaly check failed: {r.get('error')}"
        label = metric or "total"
        return (f"The anomaly detector flagged {fmt_int(r['n_flagged'])} {label} "
                f"anomaly-days (counts by metric: {r.get('counts_by_metric')}).")

    return Plan("anomaly_count", [("run_anomaly_check", args)], answer, consumed=set())


def _h_recon_largest(q: str, yr: int) -> Plan | None:
    ql = q.lower()
    if not (("recon" in ql) and any(w in ql for w in ("break", "breach"))):
        return None
    if "anomal" in ql:  # cross-subject -> decline
        return None
    if not any(w in ql for w in ("largest", "biggest", "date of", "which day", "when", "worst", "max", "greatest")):
        return None
    sql = ("SELECT date, recon_abs_diff, recon_pct_diff FROM analytics.fct_daily_revenue "
           "ORDER BY recon_abs_diff DESC LIMIT 1")

    def answer(results: list[dict[str, Any]]) -> str:
        row = _row(results[0])
        if not row:
            return "No reconciliation data found."
        d, absd, pct = row[0], float(row[1]), float(row[2])
        return (f"The largest reconciliation break was on {d}, an absolute difference of "
                f"{fmt_money(absd)} ({fmt_num(pct * 100)}%).")

    return Plan("recon_largest", [("run_sql", {"query": sql})], answer, consumed={"superlative", "selector"})


def _h_recon_list(q: str, yr: int) -> Plan | None:
    ql = q.lower()
    if not (("recon" in ql) and any(w in ql for w in ("break", "breach"))):
        return None
    if "anomal" in ql:  # cross-subject -> decline
        return None
    if not any(w in ql for w in ("list", "which", "show", "what dates", "detail")):
        return None
    period = parse_period_strict(q, yr)[1]
    where = "WHERE recon_breach"
    if period:
        where += f" AND date BETWEEN '{period[0]}' AND '{period[1]}'"
    sql = (f"SELECT date, recon_abs_diff, recon_pct_diff FROM analytics.mart_daily_health "
           f"{where} ORDER BY date")

    def answer(results: list[dict[str, Any]]) -> str:
        r = results[0]
        if not r.get("ok"):
            return f"Query failed: {r.get('error')}"
        if not r["rows"]:
            return "No reconciliation breaks in the requested window."
        parts = [f"{row[0]} ({fmt_money(float(row[1]))}, {fmt_num(float(row[2]) * 100)}%)" for row in r["rows"]]
        return f"{len(parts)} reconciliation break(s): " + "; ".join(parts) + "."

    return Plan("recon_list", [("run_sql", {"query": sql})], answer, consumed={"period", "selector"})


def _h_recon_count(q: str, yr: int) -> Plan | None:
    ql = q.lower()
    if not (("recon" in ql) and any(w in ql for w in ("break", "breach"))):
        return None
    if "anomal" in ql:  # cross-subject -> decline
        return None
    if not any(w in ql for w in ("how many", "number", "count")):
        return None
    period = parse_period_strict(q, yr)[1]
    thr = _pct_threshold(q)
    where = "WHERE recon_breach"
    if period:
        where += f" AND date BETWEEN '{period[0]}' AND '{period[1]}'"
    if thr is not None:
        op, frac, pct_str = thr
        if op == "==":
            where += f" AND round(abs(recon_pct_diff) * 100, 2) = {float(pct_str)}"
        else:
            where += f" AND abs(recon_pct_diff) {op} {frac}"
    sql = f"SELECT count(*) FROM analytics.mart_daily_health {where}"
    consumed = {"period"}
    if thr is not None:
        consumed.add("threshold")

    def answer(results: list[dict[str, Any]]) -> str:
        n = _scalar(results[0])
        if thr is not None:
            op, _, pct_str = thr
            extra = f" {_OP_WORD[op]} {pct_str}%"
        else:
            extra = ""
        return f"There were {fmt_int(n)} reconciliation breaks{extra} in the data."

    return Plan("recon_count", [("run_sql", {"query": sql})], answer, consumed=consumed)


def _h_metric_extreme_day(q: str, yr: int) -> Plan | None:
    ql = q.lower()
    if "day" not in ql and "date" not in ql:
        return None
    is_cons = any(w in ql for w in _CONS_TERMS)
    is_rev = any(w in ql for w in _REV_TERMS)
    if not (is_cons or is_rev):
        return None
    # Fail closed on a second subject / wrong target / count intent: this handler
    # only answers a clean "which <metric> day" and returns a DATE for ONE metric.
    if any(w in ql for w in ("anomal", "recon", "breach", "break", "service", "account",
                             "recognized", "recognised", "ledger")):
        return None
    if any(re.search(rf"\b{s}\b", ql) for s in _SERVICES):  # bare service name -> not a plain-metric day
        return None
    if _is_count_intent(ql):
        return None
    if is_cons and is_rev:  # ambiguous: the superlative metric and the asked metric differ
        return None
    low = any(w in ql for w in ("lowest", "smallest", "min", "minimum", "least", "worst", "trough"))
    high = any(w in ql for w in ("highest", "largest", "biggest", "max", "maximum", "top", "most", "best", "peak", "greatest"))
    if not (low or high):
        return None
    order = "ASC" if (low and not high) else "DESC"
    direction = "lowest" if order == "ASC" else "highest"
    period = parse_period_strict(q, yr)[1]
    where = _where_range(period)
    if is_cons:
        col, table, noun, money = "total_consumption_units", "fct_daily_consumption", "consumption", False
    else:
        col, table, noun, money = "total_billed_cost", "fct_daily_revenue", "revenue", True
    sql = f"SELECT date, {col} FROM analytics.{table} {where} ORDER BY {col} {order} LIMIT 1"

    def answer(results: list[dict[str, Any]]) -> str:
        row = _row(results[0])
        if not row:
            return "No data found."
        val = fmt_money(float(row[1])) if money else fmt_num(float(row[1]))
        return f"The {direction}-{noun} day was {row[0]} with {noun} value {val}."

    return Plan("metric_extreme_day", [("run_sql", {"query": sql})], answer,
                consumed={"superlative", "selector", "period"})


def _h_service_extreme(q: str, yr: int) -> Plan | None:
    ql = q.lower()
    if "service" not in ql:
        return None
    # This handler ranks services and returns a SERVICE. Decline if the question
    # actually asks for a day/date, names a specific service, or mixes subjects.
    if "day" in ql or "date" in ql:
        return None
    if _named_service(ql) or "anomal" in ql or "recon" in ql:
        return None
    low = any(w in ql for w in ("lowest", "smallest", "min", "minimum", "least", "worst"))
    high = any(w in ql for w in ("top", "highest", "largest", "most", "biggest", "maximum", "best", "greatest"))
    if not (low or high):
        return None
    if not any(w in ql for w in ("billed", "cost", "revenue", "spend")):
        return None
    order = "ASC" if (low and not high) else "DESC"
    direction = "lowest" if order == "ASC" else "highest"
    period = parse_period_strict(q, yr)[1]
    where = _where_range(period)
    sql = (f"SELECT service, sum(billed_cost) AS total_billed FROM analytics.fct_service_daily "
           f"{where} GROUP BY service ORDER BY total_billed {order} LIMIT 1")

    def answer(results: list[dict[str, Any]]) -> str:
        row = _row(results[0])
        if not row:
            return "No service data found."
        return f"The service with the {direction} total billed cost was '{row[0]}' at {fmt_money(float(row[1]))}."

    return Plan("service_extreme", [("run_sql", {"query": sql})], answer,
                consumed={"superlative", "selector", "period"})


def _h_month_extreme(q: str, yr: int) -> Plan | None:
    ql = q.lower()
    if "month" not in ql:
        return None
    low = any(w in ql for w in ("lowest", "smallest", "min", "minimum", "least", "worst"))
    high = any(w in ql for w in ("highest", "most", "largest", "top", "biggest", "maximum", "best", "greatest"))
    if not (low or high):
        return None
    if not any(w in ql for w in ("billed", "cost", "revenue", "spend")):
        return None
    order = "ASC" if (low and not high) else "DESC"
    direction = "lowest" if order == "ASC" else "highest"
    sql = ("SELECT strftime(date, '%Y-%m') AS month, sum(total_billed_cost) AS total_billed "
           f"FROM analytics.fct_daily_revenue GROUP BY month ORDER BY total_billed {order} LIMIT 1")

    def answer(results: list[dict[str, Any]]) -> str:
        row = _row(results[0])
        if not row:
            return "No revenue data found."
        return f"The month with the {direction} total billed cost was {row[0]} at {fmt_money(float(row[1]))}."

    return Plan("month_extreme", [("run_sql", {"query": sql})], answer,
                consumed={"grouping", "superlative", "selector"})


def _h_distinct_services(q: str, yr: int) -> Plan | None:
    ql = q.lower()
    if "service" not in ql:
        return None
    if not any(w in ql for w in ("how many", "number of", "count of", "distinct")):
        return None
    if any(w in ql for w in ("billed", "cost", "top", "revenue")):
        return None
    sql = "SELECT count(DISTINCT service) FROM analytics.fct_service_daily"

    def answer(results: list[dict[str, Any]]) -> str:
        return f"There are {fmt_int(_scalar(results[0]))} distinct services."

    return Plan("distinct_services", [("run_sql", {"query": sql})], answer, consumed=set())


def _h_service_total(q: str, yr: int) -> Plan | None:
    ql = q.lower()
    svc = _named_service(ql)
    if svc is None:
        return None
    if not any(w in ql for w in ("billed", "cost", "revenue", "spend", "consumption", "usage", "units")):
        return None
    if any(w in ql for w in ("average", "avg", "mean")):
        return None  # no per-service average handler
    if any(w in ql for w in ("consumption", "usage", "units")):
        col, verb, money = "consumption_units", "consumption units", False
    else:
        col, verb, money = "billed_cost", "billed cost", True
    period = parse_period_strict(q, yr)[1]
    where = f"WHERE service = '{svc}'"
    if period:
        where += f" AND date BETWEEN '{period[0]}' AND '{period[1]}'"
    sql = f"SELECT sum({col}) FROM analytics.fct_service_daily {where}"
    label = f" in {period[2]}" if period else ""

    def answer(results: list[dict[str, Any]]) -> str:
        v = _scalar(results[0])
        if v is None:
            return f"No data found for the '{svc}' service."
        shown = fmt_money(v) if money else fmt_num(v)
        return f"Total {verb} for the '{svc}' service{label} was {shown}."

    return Plan("service_total", [("run_sql", {"query": sql})], answer, consumed={"period"})


def _h_avg_active_accounts(q: str, yr: int) -> Plan | None:
    ql = q.lower()
    if "active account" not in ql:
        return None
    if not any(w in ql for w in ("average", "avg", "mean", "typical", "daily")):
        return None
    if _wrong_subject_for_total(ql) or _named_service(ql):
        return None
    period = parse_period_strict(q, yr)[1]
    where = _where_range(period)
    sql = f"SELECT avg(total_active_accounts) FROM analytics.fct_daily_consumption {where}"
    label = f" between {period[0]} and {period[1]}" if period else ""

    def answer(results: list[dict[str, Any]]) -> str:
        return f"The average daily active accounts{label} was {fmt_num(_scalar(results[0]))}."

    return Plan("avg_active_accounts", [("run_sql", {"query": sql})], answer, consumed={"avg", "period"})


def _h_avg_daily_billed(q: str, yr: int) -> Plan | None:
    ql = q.lower()
    if not any(w in ql for w in ("average", "avg", "mean")):
        return None
    if not any(w in ql for w in ("billed", "revenue", "cost")):
        return None
    if not any(w in ql for w in ("daily", "per day", "day")):
        return None
    if _wrong_subject_for_total(ql) or _named_service(ql):
        return None
    period = parse_period_strict(q, yr)[1]
    where = _where_range(period)
    sql = f"SELECT avg(total_billed_cost) FROM analytics.fct_daily_revenue {where}"
    label = f" between {period[0]} and {period[1]}" if period else ""

    def answer(results: list[dict[str, Any]]) -> str:
        return f"The average daily billed cost{label} was {fmt_money(_scalar(results[0]))}."

    return Plan("avg_daily_billed", [("run_sql", {"query": sql})], answer, consumed={"avg", "period"})


def _h_consumption_total(q: str, yr: int) -> Plan | None:
    ql = q.lower()
    if not any(w in ql for w in ("consumption", "usage", "units")):
        return None
    if "account" in ql:
        return None
    if _wrong_subject_for_total(ql) or _is_count_intent(ql) or _named_service(ql):
        return None
    period = parse_period_strict(q, yr)[1]
    where = _where_range(period)
    sql = f"SELECT sum(total_consumption_units) FROM analytics.fct_daily_consumption {where}"
    label = f" {period[2]}" if period else ""
    single_day = bool(period and period[0] == period[1])

    def answer(results: list[dict[str, Any]]) -> str:
        v = _scalar(results[0])
        verb = "Total consumption units" if not single_day else "Consumption units"
        return f"{verb}{(' on ' + period[0]) if single_day else label} was {fmt_num(v)}."

    return Plan("consumption_total", [("run_sql", {"query": sql})], answer, consumed={"period"})


def _h_recognized_revenue_total(q: str, yr: int) -> Plan | None:
    ql = q.lower()
    if not any(w in ql for w in ("recognized revenue", "recognised revenue", "ledger", "recognized", "recognised")):
        return None
    if _wrong_subject_for_total(ql) or _is_count_intent(ql) or _named_service(ql):
        return None
    period = parse_period_strict(q, yr)[1]
    where = _where_range(period)
    sql = f"SELECT sum(recognized_revenue) FROM analytics.fct_daily_revenue {where}"
    label = f" in {period[2]}" if period else ""

    def answer(results: list[dict[str, Any]]) -> str:
        return f"Total recognized revenue{label} was {fmt_money(_scalar(results[0]))}."

    return Plan("recognized_revenue_total", [("run_sql", {"query": sql})], answer, consumed={"period"})


def _h_billed_total(q: str, yr: int) -> Plan | None:
    ql = q.lower()
    if not any(w in ql for w in ("billed", "revenue", "cost", "spend")):
        return None
    if _wrong_subject_for_total(ql) or _is_count_intent(ql) or _named_service(ql):
        return None
    period = parse_period_strict(q, yr)[1]
    where = _where_range(period)
    sql = f"SELECT sum(total_billed_cost) FROM analytics.fct_daily_revenue {where}"
    label = f" in {period[2]}" if period else ""

    def answer(results: list[dict[str, Any]]) -> str:
        return f"Total billed cost{label} was {fmt_money(_scalar(results[0]))}."

    return Plan("billed_total", [("run_sql", {"query": sql})], answer, consumed={"period"})


_HANDLERS: list[Callable[[str, int], Plan | None]] = [
    _h_chart, _h_anomaly_count, _h_recon_largest, _h_recon_list, _h_recon_count,
    _h_metric_extreme_day, _h_service_extreme, _h_month_extreme, _h_distinct_services,
    _h_service_total, _h_avg_active_accounts, _h_avg_daily_billed, _h_consumption_total,
    _h_recognized_revenue_total, _h_billed_total,
]


def _decline(reason: str) -> Plan:
    def answer(_results: list[dict[str, Any]]) -> str:
        return f"{UNSUPPORTED_PREFIX} {reason}. I did not run a query, to avoid returning a misleading answer."

    return Plan("unsupported", [], answer)


def route(question: str, default_year: int = 2025) -> Plan:
    """Positive, fail-closed routing: decline unless the whole question is
    explained by recognized structure and maps to a supported handler."""
    ql = question.lower()

    # 0. Unsupported question FORMS: yes/no, and "X on the <extreme> day" cross-
    #    conditions (a value/count conditioned on an extreme day).
    if _is_yes_no(ql):
        return _decline("yes/no questions are not supported by the offline planner")
    if _on_the_extreme(ql):
        return _decline("this question conditions a value on an extreme day, which the planner does not support")
    if _count_superlatives(ql) >= 2:
        return _decline("this question asks for two or more extremes at once, a conjunction the planner cannot verify")

    # 1. Strict period: decline on any range/relative phrasing we can't parse.
    pstatus, _period, matched_period = parse_period_strict(question, default_year)
    if pstatus == "decline":
        return _decline("the question uses a time range/relative period the planner cannot parse exactly")

    # 2. Positive token accounting: every content token must be recognized.
    content = set(_content_tokens(ql))
    explained = content & _VOCAB
    if _has_threshold_op(ql):  # explain the percent value of a threshold
        explained |= {t for t in content if t.endswith("%")}
    if pstatus == "ok":  # explain exactly the tokens the period parser consumed
        explained |= set(_content_tokens(matched_period))
    leftover = content - explained
    if leftover:
        return _decline("the question contains terms the planner does not understand ("
                        + ", ".join(sorted(leftover)) + ")")

    # 3. Route to the first handler that consumes all salient qualifiers.
    present = detect_qualifiers(question, default_year)
    leftover_q: set[str] | None = None
    for handler in _HANDLERS:
        plan = handler(question, default_year)
        if plan is None:
            continue
        missing = present - plan.consumed
        if not missing:
            return plan
        if leftover_q is None:
            leftover_q = missing

    if leftover_q:
        return _decline("this question combines an aggregate/filter the planner does not support ("
                        + ", ".join(sorted(leftover_q)) + ")")
    return _decline("I could not map this question to a supported intent (the offline planner answers "
                    "revenue/consumption/reconciliation/anomaly/service/chart questions)")
