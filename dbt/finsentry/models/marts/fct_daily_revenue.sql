-- Daily revenue reconciliation fact: one row per day.
-- Cross-source check between billing (source A) and the revenue ledger (source B).
--
-- FULL OUTER join on date (not inner): a day present in only ONE source still
-- produces a row (the missing side coalesced to 0) instead of being silently
-- dropped. `billing_present` / `ledger_present` flag the gap so a missing-source
-- day surfaces as a DQ failure, caught by tests/assert_no_missing_source_day.sql.
-- Effect on reconciliation depends on WHICH side is missing:
--   * ledger missing  -> recognized_revenue=0, billed>0 -> recon_pct_diff=100% ->
--     recon_breach=true (also surfaces as a reconciliation break).
--   * billing missing -> total_billed_cost=0 -> recon_pct_diff = x/nullif(0,.) =
--     NULL -> recon_breach=false; here ONLY the missing-source-day test flags it.
-- On the current clean data every day exists in both sources, so all flags are
-- true and output is identical to the inner-join version -> existing tests stay green.
with billing as (
    select billing_date as date, sum(billed_cost) as total_billed_cost
    from {{ ref('stg_billing_line_items') }}
    group by 1
),

ledger as (
    select ledger_date as date, sum(recognized_revenue) as recognized_revenue
    from {{ ref('stg_revenue_ledger') }}
    group by 1
),

joined as (
    select
        coalesce(b.date, l.date)                 as date,
        coalesce(b.total_billed_cost, 0)         as total_billed_cost,
        coalesce(l.recognized_revenue, 0)        as recognized_revenue,
        (b.date is not null)                     as billing_present,
        (l.date is not null)                     as ledger_present
    from billing b
    full outer join ledger l using (date)
)

select
    date,
    round(total_billed_cost, 2)                                  as total_billed_cost,
    round(recognized_revenue, 2)                                 as recognized_revenue,
    round(abs(total_billed_cost - recognized_revenue), 2)        as recon_abs_diff,
    abs(total_billed_cost - recognized_revenue) / nullif(total_billed_cost, 0)
                                                                 as recon_pct_diff,
    billing_present,
    ledger_present
from joined
