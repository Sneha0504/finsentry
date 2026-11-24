-- Source B: daily GL revenue ledger. One-to-one with source.
-- Natural grain: (ledger_date, product_line) -> ledger_key. product_line == service.
with source as (
    select * from {{ source('raw', 'revenue_ledger') }}
)

select
    cast(ledger_date as date)          as ledger_date,
    cast(product_line as varchar)      as product_line,
    cast(recognized_revenue as double) as recognized_revenue,
    cast(currency as varchar)          as currency,
    cast(ledger_date as varchar) || '|' || product_line as ledger_key
from source
