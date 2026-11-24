"""Generate a clearly SYNTHETIC daily finance dataset for FinSentry.

All data produced here is synthetic and generated from a single fixed seed
(``config.seed``) via :class:`numpy.random.Generator`. There is no wall-clock or
OS-entropy input anywhere, so running this module twice produces **bit-for-bit
identical** CSV files.

Assumptions & model
-------------------
The business is modeled as a cloud provider that bills granular line items, rolls
them up into a GL-style revenue ledger, and separately meters consumption.

* **Daily revenue baseline** = ``base_daily_revenue`` scaled by:
    - a gentle linear growth trend across the year (``annual_trend_growth``),
    - weekday seasonality (``weekday_multipliers``; weekends lighter),
    - small multiplicative **Student-t** noise with a HIGH df (``revenue_noise_df``
      ~60), which is near-gaussian / light-tailed so the noise floor is predictable
      (``revenue_noise_sigma`` sets the band). The heavy tails in the series come
      from the ``shock_days`` confounders below, NOT from this noise's df,
    - UNLABELED confounders (``confounders``): holiday dips and a month-end bump,
    - and, on planted *revenue* anomaly dates, a multiplicative ``factor``.
* The daily total is split across services (``service_revenue_shares``), then
  across regions, then into individual billing line items. Each line item gets a
  ``unit_price`` (service-typical, lightly jittered) and a derived
  ``usage_quantity`` so that ``billed_cost == round(usage_quantity * unit_price, 2)``.

Why the problem is non-degenerate
---------------------------------
Planted anomalies come in two severities. **Large** ones (>=~45% deviation) are
unambiguous; **subtle** ones (~7-16%) sit near a detector's threshold so a
sensible detector misses some (recall < 1). Separately, UNLABELED confounders
(holiday dips of ~20-34%, a month-end bump, and the one-off ``shock_days`` — which
are where the series' heavy tails come from, not the near-gaussian high-df noise)
look anomalous to a calendar-unaware detector, so it raises false alarms
(precision < 1). The net
effect is a realistic detection problem rather than trivially separable spikes.

Three RAW source tables plus a ground-truth label table are written to
``data/raw/``:

``billing_line_items.csv`` (source A, granular)
    billing_date, account_id, service, region, resource_id,
    usage_quantity, unit_price, billed_cost, currency

``revenue_ledger.csv`` (source B, GL summary)
    ledger_date, product_line, recognized_revenue, currency
    One row per (date, product_line). product_line == service. On NORMAL days
    ``recognized_revenue`` equals the daily SUM of ``billed_cost`` for that
    service plus tiny GL jitter (kept *within* ``recon_tolerance_pct``). On
    planted ``recon_break`` dates the ledger is multiplied by the anomaly
    ``factor`` so it diverges from billing BEYOND tolerance. Confounders scale
    billing, and the ledger follows it, so they never create recon breaks.

``consumption.csv``
    usage_date, service, consumption_units, active_accounts
    An independent metered series with its own trend/seasonality/noise, the same
    holiday/month-end confounders, plus the planted *consumption* anomalies.

``anomaly_labels.csv`` (EXACT ground truth)
    date, metric, anomaly_type, severity, magnitude, note
    Contains every planted anomaly and ONLY planted anomalies. Confounders are
    deliberately absent.

Reconciliation contract
-----------------------
For any date, ``SUM(billed_cost)`` over billing_line_items reconciles with
``SUM(recognized_revenue)`` over revenue_ledger to within ``recon_tolerance_pct``
on normal days, and diverges beyond it on the ``recon_break`` dates. This
cross-source check is exercised by the reconciliation query in Unit 1 and will be
enforced by dbt tests in Unit 2.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from finsentry.config import Config, load_config

# Fixed CSV float formatting for bit-for-bit reproducibility.
_FLOAT_FORMAT = "%.6f"


def _trend_factor(n_days: int, growth: float) -> np.ndarray:
    """Linear growth multiplier from 1.0 (day 0) to 1.0 + growth (final day)."""
    if n_days <= 1:
        return np.ones(n_days)
    return 1.0 + growth * (np.arange(n_days) / (n_days - 1))


def _weekday_factor(dates: list[dt.date], multipliers: list[float]) -> np.ndarray:
    """Per-day weekday seasonality multiplier (Mon=0 .. Sun=6)."""
    return np.array([multipliers[d.weekday()] for d in dates], dtype=float)


def _student_t_noise(
    rng: np.random.Generator, n: int, sigma: float, df: float, clip: list[float]
) -> np.ndarray:
    """Multiplicative noise ~ 1 + sigma * t(df), clipped to ``clip`` bounds.

    At the configured HIGH df (~60) the Student-t is effectively gaussian
    (light-tailed), giving a predictable noise floor. The series' heavy tails are
    injected explicitly by the ``shock_days`` confounders, not by this noise's df.
    """
    draws = 1.0 + sigma * rng.standard_t(df, size=n)
    return np.clip(draws, clip[0], clip[1])


def _anomaly_lookup(
    anomalies: list[dict[str, Any]], metric: str
) -> dict[dt.date, dict[str, Any]]:
    """Map date -> anomaly record for a given metric."""
    out: dict[dt.date, dict[str, Any]] = {}
    for a in anomalies:
        if a["metric"] == metric:
            out[dt.date.fromisoformat(a["date"])] = a
    return out


def _holiday_lookup(cfg: Config) -> dict[dt.date, float]:
    """Map date -> holiday multiplier (UNLABELED confounder)."""
    out: dict[dt.date, float] = {}
    for h in cfg.confounders.get("holidays", []):
        out[dt.date.fromisoformat(h["date"])] = float(h["factor"])
    return out


def _shock_lookup(cfg: Config, metric: str) -> dict[dt.date, float]:
    """Map date -> one-off shock multiplier for a metric (UNLABELED confounder)."""
    out: dict[dt.date, float] = {}
    for s in cfg.confounders.get("shock_days", []):
        if s["metric"] == metric:
            out[dt.date.fromisoformat(s["date"])] = float(s["factor"])
    return out


def _is_month_end(day: dt.date) -> bool:
    return (day + dt.timedelta(days=1)).month != day.month


def _confounder_factor(
    day: dt.date, holidays: dict[dt.date, float], month_end_bump: float
) -> float:
    """Combined UNLABELED confounder multiplier for a given day."""
    factor = 1.0
    if day in holidays:
        factor *= holidays[day]
    if _is_month_end(day):
        factor *= month_end_bump
    return factor


def _build_billing_and_ledger(
    cfg: Config, rng: np.random.Generator
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate billing_line_items (source A) and revenue_ledger (source B)."""
    dg = cfg.data_generation
    dates = cfg.dates
    n = cfg.n_days

    services: list[str] = dg["services"]
    regions: list[str] = dg["regions"]
    shares = np.asarray(dg["service_revenue_shares"], dtype=float)
    unit_prices = np.asarray(dg["service_unit_price"], dtype=float)
    n_items = int(dg["line_items_per_service_region"])
    n_accounts = int(dg["n_accounts"])
    currency = dg["currency"]

    # Tuning knobs (config-driven; no magic numbers here).
    noise_clip = dg["noise_clip"]
    share_jitter_sigma = float(dg["service_share_jitter_sigma"])
    share_jitter_clip = dg["service_share_jitter_clip"]
    region_conc = float(dg["region_dirichlet_concentration"])
    item_conc = float(dg["line_item_dirichlet_concentration"])
    price_jitter_sigma = float(dg["unit_price_jitter_sigma"])
    min_unit_price = float(dg["min_unit_price"])

    accounts = [f"ACCT-{i:04d}" for i in range(1, n_accounts + 1)]

    trend = _trend_factor(n, float(dg["annual_trend_growth"]))
    weekday = _weekday_factor(dates, dg["weekday_multipliers"])
    noise = _student_t_noise(
        rng, n, float(dg["revenue_noise_sigma"]), float(dg["revenue_noise_df"]), noise_clip
    )

    revenue_anoms = _anomaly_lookup(cfg.anomalies, "revenue")
    recon_anoms = _anomaly_lookup(cfg.anomalies, "reconciliation")
    holidays = _holiday_lookup(cfg)
    month_end_bump = float(cfg.confounders.get("month_end_bump_factor", 1.0))
    shocks = _shock_lookup(cfg, "revenue")

    base = float(dg["base_daily_revenue"])

    billing_rows: list[dict[str, Any]] = []
    ledger_rows: list[dict[str, Any]] = []

    for di, day in enumerate(dates):
        daily_total = base * trend[di] * weekday[di] * noise[di]
        # UNLABELED confounders (holidays, month-end, one-off shocks) scale billing.
        daily_total *= _confounder_factor(day, holidays, month_end_bump)
        if day in shocks:
            daily_total *= shocks[day]
        # Planted (labeled) revenue anomaly.
        if day in revenue_anoms:
            daily_total *= float(revenue_anoms[day]["factor"])

        # Split daily total across services (fixed shares, tiny jitter).
        svc_jitter = rng.normal(1.0, share_jitter_sigma, size=len(services))
        svc_weights = shares * np.clip(svc_jitter, share_jitter_clip[0], share_jitter_clip[1])
        svc_weights = svc_weights / svc_weights.sum()
        svc_totals = daily_total * svc_weights

        for si, service in enumerate(services):
            # Split service total across regions (Dirichlet -> deterministic given rng).
            region_weights = rng.dirichlet(np.full(len(regions), region_conc))
            region_totals = svc_totals[si] * region_weights
            service_billed_sum = 0.0

            for ri, region in enumerate(regions):
                # Split region total across line items.
                item_weights = rng.dirichlet(np.full(n_items, item_conc))
                item_totals = region_totals[ri] * item_weights

                for ii in range(n_items):
                    unit_price = round(
                        float(unit_prices[si]) * float(rng.normal(1.0, price_jitter_sigma)), 4
                    )
                    unit_price = max(unit_price, min_unit_price)
                    target_cost = max(float(item_totals[ii]), 0.0)
                    usage_quantity = round(target_cost / unit_price, 4)
                    billed_cost = round(usage_quantity * unit_price, 2)
                    service_billed_sum += billed_cost

                    acct = accounts[int(rng.integers(0, n_accounts))]
                    billing_rows.append(
                        {
                            "billing_date": day.isoformat(),
                            "account_id": acct,
                            "service": service,
                            "region": region,
                            "resource_id": f"{service}-{region}-r{ii + 1}",
                            "usage_quantity": usage_quantity,
                            "unit_price": unit_price,
                            "billed_cost": billed_cost,
                            "currency": currency,
                        }
                    )

            # Ledger (source B): equals the service's billed sum + tiny GL jitter,
            # kept within tolerance on normal days; scaled by factor on recon breaks.
            jitter = float(rng.normal(1.0, float(dg["recon_jitter_sigma"])))
            recognized = service_billed_sum * jitter
            if day in recon_anoms:
                recognized = service_billed_sum * float(recon_anoms[day]["factor"])

            ledger_rows.append(
                {
                    "ledger_date": day.isoformat(),
                    "product_line": service,
                    "recognized_revenue": round(recognized, 2),
                    "currency": currency,
                }
            )

    billing = pd.DataFrame(billing_rows)
    ledger = pd.DataFrame(ledger_rows)
    return billing, ledger


def _build_consumption(cfg: Config, rng: np.random.Generator) -> pd.DataFrame:
    """Generate the daily consumption / usage metrics table."""
    dg = cfg.data_generation
    dates = cfg.dates
    n = cfg.n_days
    services: list[str] = dg["services"]
    shares = np.asarray(dg["service_revenue_shares"], dtype=float)
    noise_clip = dg["noise_clip"]
    acct_noise_sigma = float(dg["active_accounts_noise_sigma"])

    trend = _trend_factor(n, float(dg["consumption_annual_trend_growth"]))
    weekday = _weekday_factor(dates, dg["weekday_multipliers"])
    acct_trend = _trend_factor(n, float(dg["active_accounts_annual_trend_growth"]))

    consumption_anoms = _anomaly_lookup(cfg.anomalies, "consumption")
    holidays = _holiday_lookup(cfg)
    month_end_bump = float(cfg.confounders.get("month_end_bump_factor", 1.0))
    shocks = _shock_lookup(cfg, "consumption")
    base_cons = float(dg["base_daily_consumption"])
    base_accounts = int(dg["base_active_accounts"])
    cons_sigma = float(dg["consumption_noise_sigma"])
    cons_df = float(dg["consumption_noise_df"])

    rows: list[dict[str, Any]] = []
    for di, day in enumerate(dates):
        # Same UNLABELED confounders as billing; labeled anomaly on top.
        day_factor = _confounder_factor(day, holidays, month_end_bump)
        if day in shocks:
            day_factor *= shocks[day]
        if day in consumption_anoms:
            day_factor *= float(consumption_anoms[day]["factor"])

        for si, service in enumerate(services):
            svc_noise = float(
                np.clip(1.0 + cons_sigma * rng.standard_t(cons_df), noise_clip[0], noise_clip[1])
            )
            units = base_cons * shares[si] * trend[di] * weekday[di] * svc_noise * day_factor

            acct_noise = float(rng.normal(1.0, acct_noise_sigma))
            active = base_accounts * shares[si] / shares.max() * acct_trend[di] * acct_noise
            active_accounts = int(round(max(active, 1.0)))

            rows.append(
                {
                    "usage_date": day.isoformat(),
                    "service": service,
                    "consumption_units": round(units, 2),
                    "active_accounts": active_accounts,
                }
            )

    return pd.DataFrame(rows)


def _build_labels(cfg: Config) -> pd.DataFrame:
    """Materialize the ground-truth anomaly labels from config."""
    rows: list[dict[str, Any]] = []
    for a in cfg.anomalies:
        factor = float(a["factor"])
        # magnitude: signed fractional deviation from baseline (e.g. +0.75, -0.55).
        magnitude = round(factor - 1.0, 4)
        rows.append(
            {
                "date": a["date"],
                "metric": a["metric"],
                "anomaly_type": a["anomaly_type"],
                "severity": a.get("severity", ""),
                "magnitude": magnitude,
                "note": a["note"],
            }
        )
    df = pd.DataFrame(
        rows, columns=["date", "metric", "anomaly_type", "severity", "magnitude", "note"]
    )
    return df.sort_values(["date", "metric"]).reset_index(drop=True)


def generate(cfg: Config | None = None) -> dict[str, Path]:
    """Generate all raw CSVs. Returns a mapping of table name -> written path."""
    cfg = cfg or load_config()
    rng = np.random.default_rng(cfg.seed)

    cfg.raw_dir.mkdir(parents=True, exist_ok=True)

    # Order matters for reproducibility: billing/ledger consume the rng first,
    # then consumption. Labels are deterministic from config.
    billing, ledger = _build_billing_and_ledger(cfg, rng)
    consumption = _build_consumption(cfg, rng)
    labels = _build_labels(cfg)

    outputs = {
        "billing_line_items": cfg.raw_dir / "billing_line_items.csv",
        "revenue_ledger": cfg.raw_dir / "revenue_ledger.csv",
        "consumption": cfg.raw_dir / "consumption.csv",
        "anomaly_labels": cfg.raw_dir / "anomaly_labels.csv",
    }

    billing.to_csv(outputs["billing_line_items"], index=False, float_format=_FLOAT_FORMAT)
    ledger.to_csv(outputs["revenue_ledger"], index=False, float_format=_FLOAT_FORMAT)
    consumption.to_csv(outputs["consumption"], index=False, float_format=_FLOAT_FORMAT)
    labels.to_csv(outputs["anomaly_labels"], index=False)

    return outputs


def main() -> None:
    cfg = load_config()
    outputs = generate(cfg)
    print("SYNTHETIC data generated (seed = %d):" % cfg.seed)
    for name, path in outputs.items():
        df = pd.read_csv(path)
        print(f"  {name:<20} {len(df):>7,} rows  -> {path}")
    n_large = sum(1 for a in cfg.anomalies if a.get("severity") == "large")
    n_subtle = sum(1 for a in cfg.anomalies if a.get("severity") == "subtle")
    n_holiday = len(cfg.confounders.get("holidays", []))
    print(
        f"Planted anomalies: {len(cfg.anomalies)} "
        f"({n_large} large, {n_subtle} subtle) | "
        f"unlabeled confounders: {n_holiday} holidays + month-end bump"
    )


if __name__ == "__main__":
    main()
