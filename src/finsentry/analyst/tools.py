"""FinSentry analyst tools - the three REAL tools the agent can call.

Each tool ships as (1) a plain-Python implementation that hits the real DuckDB
warehouse / the real anomaly detector / matplotlib, and (2) an OpenAI-/Groq-
compatible JSON tool-schema (``{"type": "function", "function": {...}}``) so the
offline planner and the live Groq loop call exactly the same tools.

Tools
-----
* ``run_sql(query)`` - READ-ONLY SQL on the warehouse. Read-only is enforced two
  ways: the DuckDB connection is opened with ``read_only=True`` *and* the
  statement is validated to be a single ``SELECT``/``WITH`` with no
  INSERT/UPDATE/DELETE/CREATE/DROP/ATTACH/PRAGMA/COPY/... Row count is capped
  (auto-``LIMIT``). Errors are returned as data (never raised) so the loop can
  react.
* ``run_anomaly_check(metric=None)`` - reuses :func:`finsentry.anomaly.detect_all`
  (never reimplements detection) and returns the flagged days / counts, optionally
  filtered to one metric.
* ``create_chart(title, sql, x, y, kind)`` - runs read-only SQL and renders a
  deterministic matplotlib PNG under ``reports/analyst/<slug>.png`` (Agg backend,
  fixed style, no wall-clock), returning the path plus summary stats of the series.

All results are plain JSON-serialisable dicts. Everything is deterministic given
the built warehouse (no randomness, no clock), so the offline paths are
byte-reproducible.
"""

from __future__ import annotations

import datetime as dt
import json
import re
from decimal import Decimal
from pathlib import Path
from typing import Any

import duckdb

from finsentry.config import Config, load_config

# --------------------------------------------------------------------------- #
# Read-only SQL enforcement
# --------------------------------------------------------------------------- #
# Statement-level keywords that must never appear in an analyst query. Matched on
# word boundaries (case-insensitive). The read_only=True connection is the hard
# backstop; this gives a clear, early error string the loop can report.
_FORBIDDEN = (
    "insert", "update", "delete", "merge", "upsert",
    "create", "drop", "alter", "truncate", "replace",
    "attach", "detach", "copy", "export", "import",
    "install", "load", "pragma", "set", "reset",
    "call", "vacuum", "analyze", "grant", "revoke",
    "begin", "commit", "rollback",
)
_FORBIDDEN_RE = re.compile(
    r"(?<![A-Za-z0-9_])(" + "|".join(_FORBIDDEN) + r")(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_LIMIT_RE = re.compile(r"(?<![A-Za-z0-9_])limit(?![A-Za-z0-9_])", re.IGNORECASE)

# File/URL-access functions and schemes. Rejected at the validation layer so the
# read-only warehouse is protected even if the connection cannot set
# enable_external_access (e.g. when another connection to the file is already
# open with a different config). Defense in depth alongside that connection flag.
_EXTERNAL_FUNCS = (
    "read_csv", "read_csv_auto", "read_parquet", "parquet_scan", "read_json",
    "read_json_auto", "read_ndjson", "read_ndjson_auto", "read_text", "read_blob",
    "sniff_csv", "glob", "parquet_metadata", "parquet_schema", "iceberg_scan",
    "delta_scan", "postgres_scan", "sqlite_scan", "mysql_scan",
)
_EXTERNAL_FUNC_RE = re.compile(
    r"(?<![A-Za-z0-9_])(" + "|".join(_EXTERNAL_FUNCS) + r")\s*\(", re.IGNORECASE
)
_EXTERNAL_URL_RE = re.compile(r"(https?|s3|gcs|gs|azure|abfss?|file|hf|r2)://", re.IGNORECASE)
# DuckDB can scan a file by path used as a table source: `FROM 'file.csv'` (single
# quotes = file scan) or `FROM "…/file.parquet"`. Block any quoted source after
# FROM/JOIN so a path/URL cannot be read even when the connection cannot disable
# external access (checked on the raw code, with string literals intact).
_QUOTED_FROM_SQUOTE_RE = re.compile(r"(?<![A-Za-z0-9_])(from|join)\s+'", re.IGNORECASE)
_QUOTED_FROM_DQUOTE_PATH_RE = re.compile(
    r'(?<![A-Za-z0-9_])(from|join)\s+"[^"]*(?:/|\\|://|\.csv|\.parquet|\.json|\.jsonl|\.ndjson|\.txt|\.tsv|\.xlsx?)',
    re.IGNORECASE,
)

# Hard cap on rows returned to the loop (keeps tool output small + deterministic).
DEFAULT_MAX_ROWS = 1000


def _strip_sql_comments(sql: str) -> str:
    """Remove ``-- line`` and ``/* block */`` comments so validation sees code."""
    sql = re.sub(r"/\*.*?\*/", " ", sql, flags=re.DOTALL)
    sql = re.sub(r"--[^\n]*", " ", sql)
    return sql


def _strip_string_literals(sql: str) -> str:
    """Blank out single-quoted string literals so keyword/`;` scans ignore their
    contents (e.g. ``SELECT 'reset' AS s`` must not trip the 'reset' keyword)."""
    return re.sub(r"'(?:[^']|'')*'", "''", sql)


def validate_read_only_sql(query: str) -> tuple[bool, str]:
    """Return ``(ok, reason)``. ``ok=False`` means the statement is rejected.

    Enforces: non-empty; a single statement (no stacked ``;`` queries); begins
    with ``SELECT`` or ``WITH``; contains none of the forbidden write/DDL/PRAGMA
    keywords. String-literal contents are ignored so harmless literals like
    ``'reset'`` are not false-rejected, while real writes / stacked statements
    outside literals are still blocked.
    """
    if not query or not query.strip():
        return False, "empty query"
    code = _strip_sql_comments(query).strip()
    # Allow a single trailing semicolon only.
    code = code.rstrip(";").strip()
    # Scan for stacked statements and forbidden keywords with literals blanked out.
    scan = _strip_string_literals(code)
    if ";" in scan:
        return False, "multiple statements are not allowed (single SELECT only)"
    first = scan.split(None, 1)[0].lower() if scan.split() else ""
    if first not in ("select", "with"):
        return False, f"only SELECT/WITH queries are allowed (got '{first.upper() or '?'}')"
    m = _FORBIDDEN_RE.search(scan)
    if m:
        return False, f"forbidden keyword '{m.group(1).upper()}' is not allowed in a read-only query"
    mf = _EXTERNAL_FUNC_RE.search(scan)
    if mf:
        return False, f"external-data function '{mf.group(1)}()' is not allowed (warehouse tables only)"
    # URL and quoted-file-source checks run on `code` (string literals INTACT), so a
    # path/URL hidden inside quotes (e.g. FROM '…/x.csv' or FROM 'https://…') is caught.
    if _EXTERNAL_URL_RE.search(code):
        return False, "external URLs are not allowed in a read-only query"
    if _QUOTED_FROM_SQUOTE_RE.search(code) or _QUOTED_FROM_DQUOTE_PATH_RE.search(code):
        return False, "reading a file/path as a table source is not allowed (warehouse tables only)"
    # A quoted string anywhere in the FROM table-list (including comma-separated
    # positions, e.g. `FROM a, 'path' b`) is a file scan -> reject. We scan the
    # comma-separated table list between FROM and the first JOIN/clause keyword, so
    # legit string literals in SELECT/WHERE and in JOIN ON conditions are allowed.
    if _from_table_list_has_quote(code):
        return False, "a quoted/path value in the FROM table list is not allowed (warehouse tables only)"
    return True, ""


_FROM_CLAUSE_RE = re.compile(
    r"\bfrom\b(.*?)(?:\bwhere\b|\bgroup\b|\bhaving\b|\bqualify\b|\bwindow\b|\border\b|"
    r"\blimit\b|\bunion\b|\bexcept\b|\bintersect\b|$)",
    re.IGNORECASE | re.DOTALL,
)


def _from_table_list_has_quote(code: str) -> bool:
    """True if a single-quoted string appears in a FROM table-list slot.

    Scans each FROM clause, takes the comma-separated table list up to the first
    JOIN (so JOIN ON literals are excluded), and flags a single quote there.
    """
    for m in _FROM_CLAUSE_RE.finditer(code):
        from_clause = m.group(1)
        table_list = re.split(r"\bjoin\b", from_clause, flags=re.IGNORECASE)[0]
        if "'" in table_list:
            return True
    return False


def _auto_limit(query: str, max_rows: int) -> str:
    """Append ``LIMIT max_rows+1`` when the query has no LIMIT (for truncation detection)."""
    code = _strip_sql_comments(query).rstrip().rstrip(";").rstrip()
    if _LIMIT_RE.search(code):
        return code
    return f"{code}\nLIMIT {max_rows + 1}"


def _jsonify(value: Any) -> Any:
    """Coerce a DuckDB cell to a JSON-serialisable, deterministic Python value."""
    if value is None:
        return None
    if isinstance(value, (dt.datetime,)):
        return value.date().isoformat() if value.time() == dt.time(0, 0) else value.isoformat()
    if isinstance(value, dt.date):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, float):
        # Round floats to a stable precision so the committed logs are byte-stable.
        return round(value, 6)
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", "replace")
    return value


# --------------------------------------------------------------------------- #
# Tool 1: run_sql
# --------------------------------------------------------------------------- #
def run_sql(query: str, cfg: Config | None = None, max_rows: int = DEFAULT_MAX_ROWS) -> dict[str, Any]:
    """Execute a READ-ONLY SQL query against the warehouse; return rows as data.

    Returns ``{"ok": True, "columns": [...], "rows": [[...], ...], "row_count": n,
    "truncated": bool}`` on success, or ``{"ok": False, "error": "..."}`` on any
    rejection / DuckDB error (never raises).
    """
    cfg = cfg or load_config()
    ok, reason = validate_read_only_sql(query)
    if not ok:
        return {"ok": False, "error": f"rejected: {reason}", "query": query}

    limited = _auto_limit(query, max_rows)
    # read_only + disable external access so queries cannot read local files / URLs.
    # If another connection to the file is already open with a different config,
    # DuckDB refuses the config override; we then open a plain read-only connection
    # and disable external access at SESSION level instead. Either way, the
    # connection run_sql uses has external access OFF; if we cannot guarantee that,
    # we FAIL CLOSED. (Validation above is an independent second layer.)
    con = None
    try:
        con = duckdb.connect(
            str(cfg.duckdb_path),
            read_only=True,
            config={"enable_external_access": "false"},
        )
    except Exception:
        try:
            con = duckdb.connect(str(cfg.duckdb_path), read_only=True)
        except Exception as exc:  # pragma: no cover - connection failure
            return {"ok": False, "error": f"could not open warehouse read-only: {exc}"}
    # Enforce external access OFF on whatever connection we got (session level).
    try:
        con.execute("SET enable_external_access=false")
    except Exception:
        try:
            if con is not None:
                con.close()
        except Exception:
            pass
        return {"ok": False, "error": "could not guarantee external access is disabled; refusing to run (fail closed)"}
    try:
        cur = con.execute(limited)
        columns = [d[0] for d in cur.description] if cur.description else []
        raw_rows = cur.fetchall()
    except Exception as exc:
        con.close()
        return {"ok": False, "error": f"SQL error: {exc}", "query": query}
    con.close()

    truncated = len(raw_rows) > max_rows
    raw_rows = raw_rows[:max_rows]
    rows = [[_jsonify(c) for c in row] for row in raw_rows]
    return {
        "ok": True,
        "columns": columns,
        "rows": rows,
        "row_count": len(rows),
        "truncated": truncated,
    }


RUN_SQL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "run_sql",
        "description": (
            "Run a single READ-ONLY SQL SELECT query against the FinSentry DuckDB "
            "warehouse (schema `analytics`) and return the result rows. Only "
            "SELECT/WITH statements are allowed; writes/DDL are rejected. Key "
            "tables: analytics.fct_daily_revenue(date, total_billed_cost, "
            "recognized_revenue, recon_abs_diff, recon_pct_diff), "
            "analytics.fct_daily_consumption(date, total_consumption_units, "
            "total_active_accounts), analytics.fct_service_daily(date, service, "
            "billed_cost, consumption_units, active_accounts), "
            "analytics.mart_daily_health(date, ..., recon_breach), "
            "analytics.dq_failure_log(date, failure_type, metric, detector, severity, "
            "value, expected, deviation, cause)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A single read-only SQL SELECT/WITH statement.",
                }
            },
            "required": ["query"],
        },
    },
}


# --------------------------------------------------------------------------- #
# Tool 2: run_anomaly_check
# --------------------------------------------------------------------------- #
def run_anomaly_check(metric: str | None = None, cfg: Config | None = None) -> dict[str, Any]:
    """Run the existing anomaly detector and return flagged days / counts.

    Reuses :func:`finsentry.anomaly.detect_all` (never reimplements detection).
    ``metric`` optionally restricts to ``"revenue"`` or ``"consumption"``.
    A day is "flagged" when either detector fired (``any_flag``).
    """
    cfg = cfg or load_config()
    from finsentry.anomaly import detect_all  # local import: heavy graph

    det = detect_all(cfg)
    if metric:
        metric = metric.strip().lower()
        if metric not in ("revenue", "consumption"):
            return {"ok": False, "error": f"unknown metric '{metric}' (use 'revenue' or 'consumption')"}
        det = det[det["metric"] == metric]

    flagged = det[det["any_flag"]].sort_values(["metric", "date"])
    flagged_list = [
        {
            "date": d.date().isoformat() if hasattr(d, "date") else str(d),
            "metric": str(m),
            "value": round(float(v), 2),
            "robust_z": round(float(z), 3),
            "mad_flag": bool(mf),
            "iforest_flag": bool(iff),
        }
        for d, m, v, z, mf, iff in zip(
            flagged["date"], flagged["metric"], flagged["value"],
            flagged["robust_z"], flagged["mad_flag"], flagged["iforest_flag"],
        )
    ]
    counts = (
        det[det["any_flag"]].groupby("metric").size().astype(int).to_dict()
        if not det.empty else {}
    )
    return {
        "ok": True,
        "metric": metric or "all",
        "n_flagged": int(len(flagged_list)),
        "counts_by_metric": {str(k): int(v) for k, v in counts.items()},
        "flagged": flagged_list,
    }


RUN_ANOMALY_CHECK_SCHEMA = {
    "type": "function",
    "function": {
        "name": "run_anomaly_check",
        "description": (
            "Run FinSentry's statistical anomaly detector (robust median/MAD + "
            "Isolation Forest) over the daily revenue and consumption series and "
            "return the flagged days and per-metric counts. Optionally filter to a "
            "single metric."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "metric": {
                    "type": "string",
                    "enum": ["revenue", "consumption"],
                    "description": "Optional. Restrict the check to one metric.",
                }
            },
            "required": [],
        },
    },
}


# --------------------------------------------------------------------------- #
# Tool 3: create_chart
# --------------------------------------------------------------------------- #
def _slugify(title: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", title.strip().lower()).strip("-")
    return slug or "chart"


def create_chart(
    title: str,
    sql: str,
    x: str,
    y: str,
    kind: str = "line",
    cfg: Config | None = None,
) -> dict[str, Any]:
    """Render a deterministic PNG chart from a READ-ONLY SQL query.

    Writes ``reports/analyst/<slug(title)>.png`` with a fixed matplotlib style
    (Agg, no wall-clock metadata). Returns the path plus series summary stats so
    the caller can state the underlying value without re-querying.
    """
    cfg = cfg or load_config()
    res = run_sql(sql, cfg=cfg)
    if not res.get("ok"):
        return {"ok": False, "error": f"chart SQL failed: {res.get('error')}"}
    cols = res["columns"]
    if x not in cols or y not in cols:
        return {"ok": False, "error": f"columns {x!r}/{y!r} not in result columns {cols}"}
    xi, yi = cols.index(x), cols.index(y)
    rows = res["rows"]
    if not rows:
        return {"ok": False, "error": "chart SQL returned no rows"}
    xs = [r[xi] for r in rows]
    ys = [float(r[yi]) for r in rows]

    import matplotlib

    matplotlib.use("Agg")  # headless, deterministic
    import matplotlib.pyplot as plt

    with plt.rc_context(
        {
            "figure.figsize": (12, 5),
            "figure.dpi": 120,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 10,
        }
    ):
        fig, ax = plt.subplots()
        positions = list(range(len(xs)))
        if kind == "bar":
            ax.bar(positions, ys, color="#3b6ea5")
        else:
            ax.plot(positions, ys, color="#3b6ea5", marker="o", markersize=3, linewidth=1.5)
        # Deterministic, readable x ticks (cap to ~12 labels).
        step = max(1, len(xs) // 12)
        ax.set_xticks(positions[::step])
        ax.set_xticklabels([str(xs[i]) for i in positions[::step]], rotation=45, ha="right")
        ax.set_title(title)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        fig.tight_layout()
        out_dir = cfg.reports_dir / "analyst"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{_slugify(title)}.png"
        # metadata fixed -> no creation-time chunk -> byte-stable PNG.
        fig.savefig(out_path, metadata={"Software": "FinSentry"})
        plt.close(fig)

    return {
        "ok": True,
        "path": str(out_path.relative_to(cfg.project_root)) if out_path.is_relative_to(cfg.project_root) else str(out_path),
        "title": title,
        "kind": kind,
        "n_points": len(ys),
        "y_sum": round(sum(ys), 2),
        "y_max": round(max(ys), 2),
        "y_min": round(min(ys), 2),
        "x_first": str(xs[0]),
        "x_last": str(xs[-1]),
    }


CREATE_CHART_SCHEMA = {
    "type": "function",
    "function": {
        "name": "create_chart",
        "description": (
            "Render a PNG chart (line or bar) from a READ-ONLY SQL query and save it "
            "under reports/analyst/. The SQL must return the x and y columns named by "
            "the `x` and `y` arguments. Returns the saved file path and summary stats "
            "(sum/max/min) of the y series."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Chart title (also the file name)."},
                "sql": {"type": "string", "description": "Read-only SQL returning the x and y columns."},
                "x": {"type": "string", "description": "Name of the column to plot on the x axis."},
                "y": {"type": "string", "description": "Name of the column to plot on the y axis."},
                "kind": {
                    "type": "string",
                    "enum": ["line", "bar"],
                    "description": "Chart kind (default 'line').",
                },
            },
            "required": ["title", "sql", "x", "y"],
        },
    },
}


# --------------------------------------------------------------------------- #
# Registry / dispatch
# --------------------------------------------------------------------------- #
TOOL_SCHEMAS: list[dict[str, Any]] = [
    RUN_SQL_SCHEMA,
    RUN_ANOMALY_CHECK_SCHEMA,
    CREATE_CHART_SCHEMA,
]

_DISPATCH = {
    "run_sql": run_sql,
    "run_anomaly_check": run_anomaly_check,
    "create_chart": create_chart,
}


def execute_tool(name: str, arguments: dict[str, Any], cfg: Config | None = None) -> dict[str, Any]:
    """Execute a tool by name with a dict of arguments; return its JSON result."""
    cfg = cfg or load_config()
    fn = _DISPATCH.get(name)
    if fn is None:
        return {"ok": False, "error": f"unknown tool '{name}'"}
    try:
        return fn(cfg=cfg, **arguments)
    except TypeError as exc:
        return {"ok": False, "error": f"bad arguments for '{name}': {exc}"}
    except Exception as exc:  # pragma: no cover - defensive
        return {"ok": False, "error": f"tool '{name}' failed: {exc}"}


def result_summary(result: dict[str, Any], limit: int = 240) -> str:
    """A short, deterministic one-line summary of a tool result (for logs)."""
    if not result.get("ok", False):
        return f"error: {result.get('error', 'unknown')}"[:limit]
    if "rows" in result:
        return f"{result['row_count']} row(s), columns={result.get('columns')}"[:limit]
    if "n_flagged" in result:
        return f"{result['n_flagged']} flagged; counts={result.get('counts_by_metric')}"[:limit]
    if "path" in result:
        return f"chart -> {result['path']} (y_sum={result.get('y_sum')})"[:limit]
    return json.dumps(result, sort_keys=True)[:limit]


def result_detail(result: dict[str, Any], max_rows: int = 5) -> dict[str, Any]:
    """The actual result VALUE(S) for the tool-call log (full, or first N rows).

    Deterministic and size-capped so a log reader can check the answer the tool
    actually returned - not just a shape summary.
    """
    if not result.get("ok", False):
        return {"error": result.get("error", "unknown")}
    if "rows" in result:
        rows = result["rows"]
        return {
            "columns": result.get("columns"),
            "rows": rows[:max_rows],
            "row_count": result.get("row_count"),
            "rows_truncated": bool(result.get("truncated") or len(rows) > max_rows),
        }
    if "n_flagged" in result:
        return {
            "n_flagged": result.get("n_flagged"),
            "counts_by_metric": result.get("counts_by_metric"),
            "flagged_sample": result.get("flagged", [])[:max_rows],
            "flagged_truncated": len(result.get("flagged", [])) > max_rows,
        }
    if "path" in result:
        return {k: result[k] for k in ("path", "n_points", "y_sum", "y_max", "y_min") if k in result}
    return result
