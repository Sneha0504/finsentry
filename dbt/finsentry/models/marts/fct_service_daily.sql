-- Service-level daily fact: billing + consumption at (date, service) grain.
with billing as (
    select billing_date as date, service, sum(billed_cost) as billed_cost
    from {{ ref('stg_billing_line_items') }}
    group by 1, 2
),

consumption as (
    select
        usage_date as date,
        service,
        sum(consumption_units) as consumption_units,
        sum(active_accounts)   as active_accounts
    from {{ ref('stg_consumption') }}
    group by 1, 2
)

select
    b.date,
    b.service,
    cast(b.date as varchar) || '|' || b.service as service_day_key,
    round(b.billed_cost, 2)        as billed_cost,
    round(c.consumption_units, 2)  as consumption_units,
    c.active_accounts
from billing b
join consumption c using (date, service)
