-- Reconciliation test between two source tables (billing A vs revenue ledger B).
--
-- Business rule: for every day, SUM(billed_cost) and SUM(recognized_revenue) must
-- reconcile within `recon_tolerance_pct` (default 0.01) -- EXCEPT on days with a
-- planted, labeled recon_break. The allowlist is DATA-DRIVEN from the ground-truth
-- labels (stg_anomaly_labels where anomaly_type = 'recon_break'), not hardcoded.
--
-- A dbt test passes when it returns ZERO rows. This query returns a row for any day
-- that breaches tolerance and is NOT a known, labeled break -> so it:
--   * passes today (the only breaches are the 3 labeled recon_break days), AND
--   * would FAIL the moment a NEW, unexplained reconciliation break appears on any
--     other day, or if a labeled break's magnitude drifted onto a different day.
with known_breaks as (
    select anomaly_date as break_date
    from {{ ref('stg_anomaly_labels') }}
    where anomaly_type = 'recon_break'
)

select
    r.date,
    r.total_billed_cost,
    r.recognized_revenue,
    r.recon_abs_diff,
    r.recon_pct_diff
from {{ ref('fct_daily_revenue') }} r
where r.recon_pct_diff > {{ var('recon_tolerance_pct') }}
  and r.date not in (select break_date from known_breaks)
