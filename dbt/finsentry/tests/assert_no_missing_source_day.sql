-- Missing-source day guard (DQ test).
--
-- fct_daily_revenue is now a FULL OUTER join of billing (A) and the revenue
-- ledger (B), so a day present in only one source still produces a row with a
-- `billing_present` / `ledger_present` flag set to false. A dbt test passes when
-- it returns ZERO rows: this returns a row for any day missing from either
-- source, so it:
--   * passes today (every calendar day exists in both billing and ledger), AND
--   * would FAIL the moment an upstream feed drops a day, turning a silently
--     missing day into a loud, actionable data-quality failure.
select
    date,
    billing_present,
    ledger_present
from {{ ref('fct_daily_revenue') }}
where not billing_present
   or not ledger_present
