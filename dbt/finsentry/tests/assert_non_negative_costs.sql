-- Range/sanity test: monetary, quantity and account columns must never be negative.
-- Returns offending rows -> fails if any appear.
select 'billing' as source_table, billing_date as event_date, billed_cost as bad_value
from {{ ref('stg_billing_line_items') }}
where billed_cost < 0 or unit_price < 0 or usage_quantity < 0

union all

select 'ledger', ledger_date, recognized_revenue
from {{ ref('stg_revenue_ledger') }}
where recognized_revenue < 0

union all

select 'consumption', usage_date, consumption_units
from {{ ref('stg_consumption') }}
where consumption_units < 0 or active_accounts < 0
