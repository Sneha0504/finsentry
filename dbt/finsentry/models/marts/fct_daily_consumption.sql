-- Daily consumption rollup: one row per day across all services.
with daily as (
    select
        usage_date                 as date,
        sum(consumption_units)     as total_consumption_units,
        sum(active_accounts)       as total_active_accounts
    from {{ ref('stg_consumption') }}
    group by 1
)

select
    date,
    round(total_consumption_units, 2) as total_consumption_units,
    total_active_accounts
from daily
