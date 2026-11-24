"""Local end-to-end pipeline runner for FinSentry (no Docker, no Airflow).

Runs the identical stages the Airflow DAG mirrors, so ``make reproduce`` works on
any machine with the pinned venv and no external credentials:

    1. generate_data   -> write SYNTHETIC raw CSVs (fixed seed)
    2. load_raw        -> load CSVs into DuckDB `raw` schema
    3. dbt run         -> build staging + mart models in `analytics`
    4. dbt test        -> run all data quality tests (uniqueness, nulls, FK, recon)
    5. dq_monitoring   -> anomaly detection + eval + failure log + figures (Unit 3)

Later units (AI analyst, evaluation) append their stages to
``STAGES`` below; the ordering and hook points are already in place.

Each stage is timed and logged with a clear PASS/FAIL marker. The process exits
non-zero if any *required* stage fails, so CI / make can detect breakage. dbt is
invoked via subprocess from the repo root with the project's own profiles-dir.

Usage::

    python -m finsentry.pipeline_runner
    python -m finsentry.pipeline_runner --skip-generate      # reuse existing CSVs
    python -m finsentry.pipeline_runner --select fct_daily_revenue   # dbt --select passthrough
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from finsentry.config import find_project_root

# --- Markers (ASCII-safe fallbacks are fine on any terminal) ---
_OK = "✓"  # ✓
_FAIL = "✗"  # ✗

REPO_ROOT = find_project_root()
DBT_PROJECT_DIR = REPO_ROOT / "dbt" / "finsentry"
DUCKDB_PATH = REPO_ROOT / "data" / "warehouse.duckdb"


def _log(msg: str) -> None:
    print(msg, flush=True)


def _dbt_executable() -> str:
    """Prefer the dbt shipped in the active venv; fall back to PATH."""
    candidate = Path(sys.executable).parent / "dbt"
    return str(candidate) if candidate.exists() else "dbt"


def _dbt_base_cmd() -> list[str]:
    return [
        _dbt_executable(),
        "--no-use-colors",
        # subcommand is appended by each stage
    ]


def _dbt_env() -> dict[str, str]:
    """Env for dbt: pin the DuckDB path to an absolute location so the run is
    independent of the current working directory."""
    env = dict(os.environ)
    env["FINSENTRY_DUCKDB"] = str(DUCKDB_PATH)
    return env


# --------------------------------------------------------------------------- #
# Stage implementations
# --------------------------------------------------------------------------- #
def _stage_generate_data() -> bool:
    from finsentry.generate_data import main as generate_main

    generate_main()
    return True


def _stage_load_raw() -> bool:
    from finsentry.load_raw import main as load_main

    load_main()
    return True


def _stage_dq_monitoring() -> bool:
    """Unit 3: anomaly detection -> eval -> failure log -> figures (deterministic).

    Success is contingent on the stage actually producing its outputs: the eval
    JSON, a populated ``analytics.dq_failure_log`` table, and all four non-empty
    figures. A silent miss therefore surfaces as a non-zero pipeline exit.
    """
    import duckdb

    from finsentry.anomaly import run_dq_monitoring
    from finsentry.config import load_config

    cfg = load_config()
    result = run_dq_monitoring(cfg, verbose=True)

    # 1. eval JSON written.
    eval_json = cfg.reports_dir / "anomaly_eval.json"
    if not eval_json.is_file() or eval_json.stat().st_size == 0:
        _log(f"    ! missing/empty eval report: {eval_json}")
        return False

    # 2. failure-log table exists and is populated; CSV written.
    csv_path = cfg.reports_dir / "dq_failure_log.csv"
    if not csv_path.is_file():
        _log(f"    ! missing failure-log CSV: {csv_path}")
        return False
    con = duckdb.connect(str(cfg.duckdb_path), read_only=True)
    try:
        n_rows = con.execute(
            "SELECT COUNT(*) FROM analytics.dq_failure_log"
        ).fetchone()[0]
    except Exception as exc:
        _log(f"    ! analytics.dq_failure_log not readable: {exc!r}")
        return False
    finally:
        con.close()
    if n_rows <= 0:
        _log("    ! analytics.dq_failure_log is empty")
        return False

    # 3. all expected figures exist and are non-empty.
    figures = result.get("figures", [])
    if len(figures) != 4:
        _log(f"    ! expected 4 figures, got {len(figures)}")
        return False
    for p in figures:
        fp = Path(p)
        if not fp.is_file() or fp.stat().st_size == 0:
            _log(f"    ! missing/empty figure: {fp}")
            return False

    return True


# Captured so the final summary can surface the eval accuracy.
_ANALYST_SUMMARY: dict[str, object] = {}


def _stage_weekly_review() -> bool:
    """Unit 4: generate the deterministic weekly business review (offline provider)."""
    from finsentry.analyst.weekly_review import build_review
    from finsentry.config import load_config

    out = build_review(load_config(), provider_name=_ANALYST_SUMMARY.get("provider"), verbose=True)
    md = Path(out["md_path"])
    return md.is_file() and md.stat().st_size > 0


def _stage_verify_numbers() -> bool:
    """Unit 4: verify EVERY number in the weekly review against the warehouse.

    This is the grounding guardrail: a mismatch or an uncovered (hallucinated)
    number fails the stage (non-zero pipeline exit).
    """
    from finsentry.analyst.verify_numbers import verify
    from finsentry.config import load_config

    summary = verify(load_config(), verbose=True)
    return bool(summary["overall_pass"])


def _stage_eval() -> bool:
    """Unit 4: run the 15-question eval. Questions legitimately failing are reported
    but do NOT fail the stage; only an infrastructure error does."""
    from finsentry.config import load_config
    from finsentry.eval.run_eval import run_eval

    summary = run_eval(load_config(), provider_name=_ANALYST_SUMMARY.get("provider"), verbose=True)
    _ANALYST_SUMMARY["eval_accuracy"] = summary["accuracy"]
    _ANALYST_SUMMARY["eval_pass"] = summary["n_pass"]
    _ANALYST_SUMMARY["eval_total"] = summary["n_questions"]
    return True  # run completed; per-question failures are reported, not fatal


def _run_dbt(subcommand: str, extra: list[str]) -> bool:
    """Invoke a dbt subcommand from the repo root; stream output. Returns success."""
    cmd = [
        *_dbt_base_cmd(),
        subcommand,
        "--project-dir",
        str(DBT_PROJECT_DIR),
        "--profiles-dir",
        str(DBT_PROJECT_DIR),
        *extra,
    ]
    _log(f"    $ {' '.join(cmd)}")
    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        env=_dbt_env(),
        capture_output=True,
        text=True,
    )
    # dbt summary lines are the most useful; show stdout (and stderr on failure).
    if proc.stdout:
        for line in proc.stdout.rstrip().splitlines():
            _log(f"    | {line}")
    if proc.returncode != 0 and proc.stderr:
        for line in proc.stderr.rstrip().splitlines():
            _log(f"    ! {line}")
    return proc.returncode == 0


@dataclass
class Stage:
    name: str
    func: Callable[[], bool]
    required: bool = True
    # Populated at runtime.
    ok: bool | None = field(default=None, init=False)
    seconds: float = field(default=0.0, init=False)


def build_stages(args: argparse.Namespace) -> list[Stage]:
    """Assemble the ordered stage list. Later units append here."""
    select = ["--select", *args.select] if args.select else []

    stages: list[Stage] = []
    if not args.skip_generate:
        stages.append(Stage("generate_data", _stage_generate_data))
    stages.append(Stage("load_raw", _stage_load_raw))
    stages.append(Stage("dbt_run", lambda: _run_dbt("run", select)))
    stages.append(Stage("dbt_test", lambda: _run_dbt("test", select)))
    # Unit 3: data-quality monitoring (detection -> eval -> failure log -> figures).
    if not args.skip_dq:
        stages.append(Stage("dq_monitoring", _stage_dq_monitoring))

    # Unit 4: AI analyst -> weekly review -> number verification -> 15-question eval.
    # All run offline/deterministic by default. verify_numbers failing fails the
    # pipeline (grounding guardrail); eval questions failing are reported only.
    if not args.skip_analyst:
        _ANALYST_SUMMARY["provider"] = args.provider
        stages.append(Stage("weekly_review", _stage_weekly_review))
        stages.append(Stage("verify_numbers", _stage_verify_numbers))
        stages.append(Stage("eval", _stage_eval))
    return stages


def run(stages: list[Stage]) -> int:
    """Execute stages in order. Returns a process exit code (0 == success)."""
    _log("=" * 70)
    _log("FinSentry pipeline runner (local, no Docker/Airflow)")
    _log(f"repo root : {REPO_ROOT}")
    _log(f"duckdb    : {DUCKDB_PATH}")
    _log(f"dbt proj  : {DBT_PROJECT_DIR}")
    _log("=" * 70)

    overall_start = time.perf_counter()
    failed_required = False

    for i, stage in enumerate(stages, start=1):
        _log(f"\n[{i}/{len(stages)}] {stage.name} ...")
        start = time.perf_counter()
        try:
            stage.ok = bool(stage.func())
        except Exception as exc:  # a stage must never crash the whole runner
            stage.ok = False
            _log(f"    ! exception: {exc!r}")
        stage.seconds = time.perf_counter() - start
        marker = _OK if stage.ok else _FAIL
        tag = "" if stage.required else " (optional)"
        _log(f"    {marker} {stage.name}{tag} in {stage.seconds:.2f}s")
        if not stage.ok and stage.required:
            failed_required = True
            # Stop at the first required failure: downstream stages depend on it.
            break

    total = time.perf_counter() - overall_start
    _log("\n" + "=" * 70)
    _log("Summary")
    for stage in stages:
        if stage.ok is None:
            marker, status = "-", "SKIPPED"
        elif stage.ok:
            marker, status = _OK, "PASS"
        else:
            marker, status = _FAIL, "FAIL"
        _log(f"  {marker} {stage.name:<16} {status:<8} {stage.seconds:>6.2f}s")
    _log(f"  total {total:>40.2f}s")
    if "eval_accuracy" in _ANALYST_SUMMARY:
        _log(
            f"  analyst eval accuracy: {_ANALYST_SUMMARY['eval_pass']}/"
            f"{_ANALYST_SUMMARY['eval_total']} "
            f"= {float(_ANALYST_SUMMARY['eval_accuracy']):.1%}"
        )
    _log("=" * 70)

    if failed_required:
        _log(f"{_FAIL} pipeline FAILED (a required stage did not pass)")
        return 1
    _log(f"{_OK} pipeline completed successfully")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the FinSentry pipeline end-to-end.")
    parser.add_argument(
        "--skip-generate",
        action="store_true",
        help="Reuse existing raw CSVs instead of regenerating them.",
    )
    parser.add_argument(
        "--select",
        nargs="+",
        default=None,
        help="Passthrough to dbt --select for the run/test stages.",
    )
    parser.add_argument(
        "--skip-dq",
        action="store_true",
        help="Skip the Unit 3 data-quality monitoring stage.",
    )
    parser.add_argument(
        "--skip-analyst",
        action="store_true",
        help="Skip the Unit 4 analyst stages (weekly review, verify, eval).",
    )
    parser.add_argument(
        "--provider",
        choices=["offline", "groq"],
        default=None,
        help="Analyst provider override (default: offline/config; GROQ_API_KEY=>groq).",
    )
    args = parser.parse_args(argv)
    return run(build_stages(args))


if __name__ == "__main__":
    raise SystemExit(main())
