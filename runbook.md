# FinSentry Data-Quality Runbook

Practical triage for every data-quality failure FinSentry raises. Data is
**synthetic** with planted, labeled anomalies (ground truth in
`analytics.stg_anomaly_labels`); the same steps apply to a real Snowflake deploy.

## Where failures live

| Source | Location |
|---|---|
| Unified failure log | DuckDB `analytics.dq_failure_log` and `reports/dq_failure_log.csv` |
| Detector detail / scores | `reports/anomaly_eval.json`, `reports/anomaly_eval.md` |
| Figures (from the real run) | `reports/revenue_timeseries_anomalies.png`, `reports/consumption_timeseries_anomalies.png`, `reports/reconciliation.png`, `reports/precision_recall.png` |
| dbt test results | `dbt/finsentry/target/run_results.json` + console |

Rebuild everything with one command:

```bash
.venv/bin/python -m finsentry.pipeline_runner
# stages: generate_data -> load_raw -> dbt run -> dbt test -> dq_monitoring
```

Re-run only the monitoring step (warehouse already built):

```bash
.venv/bin/python -m finsentry.anomaly
```

Inspect the failure log:

```sql
-- in DuckDB (data/warehouse.duckdb)
SELECT date, failure_type, metric, detector, severity, deviation, is_known_label, cause
FROM analytics.dq_failure_log
ORDER BY date;
```

---

## 1. Revenue anomaly (`failure_type = 'revenue_anomaly'`)

- **Symptom:** a day flagged by the MAD and/or Isolation Forest detector on
  `total_billed_cost`; `deviation` is the fractional gap from the deseasonalized
  norm.
- **Triage:**
  1. `SELECT * FROM analytics.dq_failure_log WHERE date = '<day>';` — read `cause`
     and `is_known_label`.
  2. Cross-check the series: `SELECT date, total_billed_cost FROM
     analytics.fct_daily_revenue ORDER BY date;` and the figure
     `reports/revenue_timeseries_anomalies.png`.
  3. Drill to service level: `SELECT service, billed_cost FROM
     analytics.fct_service_daily WHERE date = '<day>' ORDER BY billed_cost DESC;`
- **Likely cause:** real billing event (price-list migration double-bill, refund
  batch, enterprise prepay) **or** a known confounder — a deep holiday dip
  (Christmas 12-25, Thanksgiving 11-27) or a one-off demand shock. The log's
  `cause` names which; confounders are expected seasonal variation, not true
  anomalies.
- **Fix:** if a real billing defect, correct the upstream billing feed / rate card
  and reload; if an expected confounder, annotate and suppress (add to the
  calendar so the detector deseasonalizes it next cycle).
- **Confirm resolved:** re-run `python -m finsentry.anomaly`; the day should drop
  out of `dq_failure_log` (or flip to a documented, suppressed confounder).

## 2. Consumption anomaly (`failure_type = 'consumption_anomaly'`)

- **Symptom:** a day flagged on `total_consumption_units`.
- **Triage:** same as revenue but against `analytics.fct_daily_consumption` and
  `reports/consumption_timeseries_anomalies.png`; service breakdown via
  `fct_service_daily.consumption_units`.
- **Likely cause:** metering-pipeline issues (collector crash, agent outage,
  backfill job, load-test traffic counted as real) or a metering wobble
  confounder. `cause` distinguishes them.
- **Fix:** replay/repair the metering pipeline for the day; if load-test or
  synthetic traffic, exclude it at source.
- **Confirm resolved:** re-run monitoring; day clears or is documented.

## 3. Reconciliation break (`failure_type = 'reconciliation_break'`)

- **Symptom:** `mart_daily_health.recon_breach = true` — billed cost (source A) and
  recognized revenue (source B) diverge beyond `recon_tolerance_pct` (1.0%).
  `deviation` is signed `(recognized - billed)/billed`.
- **Triage:**
  1. `SELECT date, total_billed_cost, recognized_revenue, recon_pct_diff FROM
     analytics.fct_daily_revenue WHERE recon_pct_diff > 0.01 ORDER BY date;`
  2. See `reports/reconciliation.png` (breaks highlighted vs the tolerance line).
  3. Check whether it is a known labeled break:
     `SELECT * FROM analytics.stg_anomaly_labels WHERE anomaly_type='recon_break';`
- **Likely cause:** GL posted revenue the billing feed never recorded, a late
  billing feed leaving GL short, or a manual GL adjustment.
- **Fix:** reconcile the ledger to billing (post/reverse the missing entry); if a
  timing gap, reload the late feed.
- **Confirm resolved:** re-run `dbt test` — `assert_revenue_reconciliation` passes
  (zero unexplained breaches); `recon_breach` clears in `mart_daily_health`.

## 4. dbt test failure

- **Symptom:** `dbt test` exits non-zero; `pipeline_runner` stops at the
  `dbt_test` stage.
- **Triage:**
  1. Read the failing test name from the console / `target/run_results.json`.
  2. `assert_revenue_reconciliation` → an **unexpected** recon break (not in the
     labels) — go to section 3.
  3. `assert_no_missing_source_day` → a day is missing from billing or the ledger
     — go to section 5.
  4. `not_null` / `unique` / `relationships` → inspect the named model and column.
- **Likely cause:** upstream schema/row-count drift, a new unexplained break, or a
  missing source day.
- **Fix:** repair the upstream data, then `dbt run --select <model>` and re-test.
- **Confirm resolved:** `.venv/bin/dbt test --project-dir dbt/finsentry
  --profiles-dir dbt/finsentry` is all green.

## 5. Missing day / pipeline break (`assert_no_missing_source_day`)

- **Symptom:** a calendar day is present in only one of billing / ledger (or
  neither). `fct_daily_revenue` now FULL OUTER joins the two sources, so a day in
  only one source still appears with `billing_present` or `ledger_present = false`
  (the missing side coalesced to 0). Reconciliation effect depends on which side
  is gone: **ledger missing** -> recognized=0, billed>0 -> recon_pct_diff=100% ->
  recon_breach=true (also shows as a reconciliation break); **billing missing** ->
  billed=0 -> recon_pct_diff is NULL -> recon_breach=false, so the
  `assert_no_missing_source_day` test is what catches it. A day missing from
  **both** sources has no `fct_daily_revenue` row, so `mart_daily_health` carries
  it from `dim_date` with NULL metrics -> caught by the `not_null` test on
  `mart_daily_health.total_billed_cost` / `total_consumption_units` and listed as a
  `missing_day` row in the failure log.
- **Triage:**
  1. `SELECT date, billing_present, ledger_present FROM
     analytics.fct_daily_revenue WHERE NOT billing_present OR NOT ledger_present;`
  2. For a day missing from BOTH sources: `SELECT date FROM
     analytics.mart_daily_health WHERE total_billed_cost IS NULL OR
     total_consumption_units IS NULL;` (also appears as `missing_day` in
     `analytics.dq_failure_log`).
  3. Confirm the raw load: `SELECT COUNT(*) FROM raw.billing_line_items;` /
     `raw.revenue_ledger;` and that the CSVs exist in `data/raw/`.
- **Likely cause:** an upstream extract dropped a day, or `generate_data` /
  `load_raw` did not complete.
- **Fix:** re-extract/reload the missing day (or re-run
  `python -m finsentry.pipeline_runner`).
- **Confirm resolved:** `assert_no_missing_source_day` returns zero rows, the
  `not_null` tests on `mart_daily_health.total_billed_cost` /
  `total_consumption_units` pass, and both `*_present` flags are true for every day.

---

### Severity & known-label semantics

- `is_known_label = true`: the failure matches a planted ground-truth label — in
  production this maps to a confirmed/triaged incident.
- `is_known_label = false`: a **suspected false positive**; the `cause` attributes
  it to a known confounder (holiday dip, one-off shock), neighbor-day spillover, or
  residual near-threshold noise. Residual-noise flags are the expected cost of
  keeping recall high (precision < 1 by design).
