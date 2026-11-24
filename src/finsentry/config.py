"""Central configuration loader for FinSentry.

Reads ``config.yaml`` from the project root into a typed :class:`Config` object.
Every downstream module (data generation, loading, anomaly detection, app)
imports :func:`load_config` so there are no magic numbers scattered around.

Paths in the config are stored relative to the project root; the loader resolves
them to absolute paths against the detected project root.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


def find_project_root(start: Path | None = None) -> Path:
    """Return the project root (the directory containing ``config.yaml``).

    Walks upward from ``start`` (or this file) until ``config.yaml`` is found,
    so the loader works regardless of the current working directory.
    """
    start = start or Path(__file__).resolve()
    for candidate in [start, *start.parents]:
        if (candidate / "config.yaml").is_file():
            return candidate
    raise FileNotFoundError(
        "Could not locate config.yaml walking up from "
        f"{start!s}; is the project root intact?"
    )


@dataclass(frozen=True)
class Config:
    """Typed view over ``config.yaml`` plus resolved absolute paths."""

    seed: int
    start_date: dt.date
    end_date: dt.date

    # Absolute, resolved paths.
    project_root: Path
    raw_dir: Path
    duckdb_path: Path
    reports_dir: Path

    # Nested sections kept as plain dicts/lists for flexibility downstream.
    data_generation: dict[str, Any]
    anomalies: list[dict[str, Any]]
    confounders: dict[str, Any]
    anomaly_detection: dict[str, Any]

    # The full raw mapping, for anything not surfaced as a typed field.
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def dates(self) -> list[dt.date]:
        """Inclusive list of daily dates from ``start_date`` to ``end_date``."""
        n = (self.end_date - self.start_date).days + 1
        return [self.start_date + dt.timedelta(days=i) for i in range(n)]

    @property
    def n_days(self) -> int:
        """Number of inclusive days in the configured date range."""
        return (self.end_date - self.start_date).days + 1


def _as_date(value: Any) -> dt.date:
    if isinstance(value, dt.date):
        return value
    if isinstance(value, str):
        return dt.date.fromisoformat(value)
    raise TypeError(f"Expected ISO date string or date, got {type(value)!r}")


def load_config(config_path: str | Path | None = None) -> Config:
    """Load and validate configuration.

    Parameters
    ----------
    config_path:
        Optional explicit path to ``config.yaml``. When omitted, the project
        root is auto-detected.
    """
    if config_path is None:
        root = find_project_root()
        config_path = root / "config.yaml"
    else:
        config_path = Path(config_path).resolve()
        root = config_path.parent

    with open(config_path, "r", encoding="utf-8") as fh:
        raw: dict[str, Any] = yaml.safe_load(fh)

    paths = raw["paths"]
    start_date = _as_date(raw["start_date"])
    end_date = _as_date(raw["end_date"])
    if end_date < start_date:
        raise ValueError("end_date must be on or after start_date")

    # Light validation of revenue shares (must sum ~1.0).
    shares = raw["data_generation"]["service_revenue_shares"]
    if abs(sum(shares) - 1.0) > 1e-6:
        raise ValueError(f"service_revenue_shares must sum to 1.0, got {sum(shares)}")

    return Config(
        seed=int(raw["seed"]),
        start_date=start_date,
        end_date=end_date,
        project_root=root,
        raw_dir=(root / paths["raw_dir"]).resolve(),
        duckdb_path=(root / paths["duckdb_path"]).resolve(),
        reports_dir=(root / paths["reports_dir"]).resolve(),
        data_generation=raw["data_generation"],
        anomalies=raw["anomalies"],
        confounders=raw.get("confounders", {}),
        anomaly_detection=raw.get("anomaly_detection", {}),
        raw=raw,
    )


if __name__ == "__main__":
    cfg = load_config()
    print(f"Project root : {cfg.project_root}")
    print(f"Seed         : {cfg.seed}")
    print(f"Date range   : {cfg.start_date} -> {cfg.end_date} ({cfg.n_days} days)")
    print(f"Raw dir      : {cfg.raw_dir}")
    print(f"DuckDB path  : {cfg.duckdb_path}")
    print(f"Reports dir  : {cfg.reports_dir}")
    print(f"Planted anomalies: {len(cfg.anomalies)}")
