-- Source A: granular billing line items. Light typing/renaming, one-to-one with source.
-- Natural grain: (billing_date, service, region, resource_id) -> line_item_key.
with source as (
    select * from {{ source('raw', 'billing_line_items') }}
)

select
    cast(billing_date as date)        as billing_date,
    cast(account_id as varchar)       as account_id,
    cast(service as varchar)          as service,
    cast(region as varchar)           as region,
    cast(resource_id as varchar)      as resource_id,
    cast(usage_quantity as double)    as usage_quantity,
    cast(unit_price as double)        as unit_price,
    cast(billed_cost as double)       as billed_cost,
    cast(currency as varchar)         as currency,
    cast(billing_date as varchar) || '|' || service || '|' || region || '|' || resource_id
                                      as line_item_key
from source
