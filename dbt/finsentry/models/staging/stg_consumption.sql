-- Metered consumption. One-to-one with source.
-- Natural grain: (usage_date, service) -> consumption_key.
with source as (
    select * from {{ source('raw', 'consumption') }}
)

select
    cast(usage_date as date)           as usage_date,
    cast(service as varchar)           as service,
    cast(consumption_units as double)  as consumption_units,
    cast(active_accounts as bigint)    as active_accounts,
    cast(usage_date as varchar) || '|' || service as consumption_key
from source
