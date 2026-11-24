-- Date spine covering the full range present across all source tables.
--
-- Cross-dialect WITHOUT extra package deps: the date spine is the only
-- engine-specific piece, branched on `target.type`. DuckDB uses
-- `generate_series(... , interval '1 day')`; Snowflake has no `generate_series`,
-- so it uses `TABLE(GENERATOR(...))` + `DATEADD` and filters to the end date
-- (GENERATOR's rowcount must be a constant, so we over-generate and clip). The
-- DuckDB branch is unchanged, so its output is byte-identical to before.
with bounds as (
    select
        least(
            (select min(billing_date) from {{ ref('stg_billing_line_items') }}),
            (select min(ledger_date)  from {{ ref('stg_revenue_ledger') }}),
            (select min(usage_date)   from {{ ref('stg_consumption') }})
        ) as start_date,
        greatest(
            (select max(billing_date) from {{ ref('stg_billing_line_items') }}),
            (select max(ledger_date)  from {{ ref('stg_revenue_ledger') }}),
            (select max(usage_date)   from {{ ref('stg_consumption') }})
        ) as end_date
),

spine as (
{%- if target.type == 'snowflake' %}
    -- Snowflake: GENERATOR rowcount is a constant; over-generate ~55 years of
    -- days from start_date and clip to end_date in the final WHERE below.
    -- seq4() is not guaranteed gap-free, so derive a gap-free 0-based offset with
    -- row_number() over (order by seq4()).
    select
        dateadd(
            day,
            row_number() over (order by seq4()) - 1,
            (select start_date from bounds)
        ) as date_ts
    from table(generator(rowcount => 20000))
{%- else %}
    -- DuckDB (default / reproduce path): native date-series generator.
    select unnest(
        generate_series(
            (select start_date from bounds),
            (select end_date from bounds),
            interval '1 day'
        )
    ) as date_ts
{%- endif %}
)

select
    cast(date_ts as date)                                   as date,
    extract(year  from date_ts)                             as year,
    extract(month from date_ts)                             as month,
    extract(day   from date_ts)                             as day_of_month,
{%- if target.type == 'snowflake' %}
    dayofweek(date_ts)                                      as day_of_week,  -- 0=Sun .. 6=Sat (WEEK_START=0)
    (dayofweek(date_ts) in (0, 6))                          as is_weekend
{%- else %}
    extract(dow   from date_ts)                             as day_of_week,  -- 0=Sun .. 6=Sat
    (extract(dow from date_ts) in (0, 6))                   as is_weekend
{%- endif %}
from spine
{%- if target.type == 'snowflake' %}
where date_ts <= (select end_date from bounds)
{%- endif %}
