-- Ground-truth planted-anomaly labels. `date` renamed to `anomaly_date` (reserved-word safe).
with source as (
    select * from {{ source('raw', 'anomaly_labels') }}
)

select
    cast(date as date)          as anomaly_date,
    cast(metric as varchar)     as metric,
    cast(anomaly_type as varchar) as anomaly_type,
    cast(severity as varchar)   as severity,
    cast(magnitude as double)   as magnitude,
    cast(note as varchar)       as note
from source
