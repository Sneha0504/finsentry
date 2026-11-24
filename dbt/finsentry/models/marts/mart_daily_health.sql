-- Convenience daily health table: revenue + consumption + a reconciliation flag.
-- recon_breach is driven by the configurable var `recon_tolerance_pct` (default 0.01).
-- Built off dim_date so every calendar day is present even if a source has a gap.
select
    d.date,
    r.total_billed_cost,
    r.recognized_revenue,
    r.recon_abs_diff,
    r.recon_pct_diff,
    c.total_consumption_units,
    c.total_active_accounts,
    coalesce(r.recon_pct_diff > {{ var('recon_tolerance_pct') }}, false) as recon_breach
from {{ ref('dim_date') }} d
left join {{ ref('fct_daily_revenue') }} r on d.date = r.date
left join {{ ref('fct_daily_consumption') }} c on d.date = c.date
