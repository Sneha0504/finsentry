-- ===========================================================================
-- FinSentry - Snowflake DDL for the RAW landing tables
-- ===========================================================================
-- Parallel to the DuckDB `raw` schema that `python -m finsentry.load_raw` builds
-- from data/raw/*.csv. The dbt staging/mart models read `source('raw', …)` on top
-- of these tables; all are portable ANSI SQL except `dim_date`, the one
-- dialect-specific model, which branches on `target.type` (see DEPLOY.md §3), so
-- selecting the snowflake target is enough — no manual model edits.
--
-- Data is SYNTHETIC (fixed seed 42). The project was built and validated on
-- DuckDB; this file is the Snowflake-equivalent landing layer. See DEPLOY.md.
--
-- Column names/order mirror data/raw/*.csv exactly so the same COPY INTO works.
-- Run with:  snowsql -f snowflake/ddl.sql   (or paste into a Snowsight worksheet)
-- ===========================================================================

-- --- Database / warehouse / schema -----------------------------------------
-- Adjust names/sizes to taste; these match DEPLOY.md.
CREATE DATABASE IF NOT EXISTS FINSENTRY;
CREATE WAREHOUSE IF NOT EXISTS FINSENTRY_WH
    WAREHOUSE_SIZE = 'XSMALL'
    AUTO_SUSPEND = 60
    AUTO_RESUME = TRUE
    INITIALLY_SUSPENDED = TRUE;

USE DATABASE FINSENTRY;
USE WAREHOUSE FINSENTRY_WH;

-- RAW landing schema (parallel to DuckDB `raw`). dbt builds staging + marts into
-- a separate ANALYTICS schema (see profiles.yml / DEPLOY.md).
CREATE SCHEMA IF NOT EXISTS RAW;
USE SCHEMA RAW;

-- ===========================================================================
-- Source A: granular cloud billing line items
--   one row per (date, service, region, resource); ~21,900 rows / year
-- ===========================================================================
CREATE OR REPLACE TABLE RAW.BILLING_LINE_ITEMS (
    billing_date    DATE            NOT NULL,
    account_id      VARCHAR         NOT NULL,
    service         VARCHAR         NOT NULL,
    region          VARCHAR         NOT NULL,
    resource_id     VARCHAR         NOT NULL,
    usage_quantity  NUMBER(20, 6)   NOT NULL,
    unit_price      NUMBER(12, 6)   NOT NULL,
    billed_cost     NUMBER(18, 6)   NOT NULL,
    currency        VARCHAR         NOT NULL
);

-- ===========================================================================
-- Source B: daily GL revenue ledger
--   one row per (date, product_line); ~1,825 rows / year
--   Reconciles to billing within tolerance except on planted recon_break days.
-- ===========================================================================
CREATE OR REPLACE TABLE RAW.REVENUE_LEDGER (
    ledger_date         DATE            NOT NULL,
    product_line        VARCHAR         NOT NULL,
    recognized_revenue  NUMBER(18, 6)   NOT NULL,
    currency            VARCHAR         NOT NULL
);

-- ===========================================================================
-- Daily metered consumption
--   one row per (date, service); ~1,825 rows / year
-- ===========================================================================
CREATE OR REPLACE TABLE RAW.CONSUMPTION (
    usage_date          DATE            NOT NULL,
    service             VARCHAR         NOT NULL,
    consumption_units   NUMBER(20, 6)   NOT NULL,
    active_accounts     NUMBER(12, 0)   NOT NULL
);

-- ===========================================================================
-- Ground-truth planted-anomaly labels (used for precision/recall)
--   18 rows: 15 metric anomalies (10 large + 5 subtle) + 3 reconciliation breaks
-- ===========================================================================
CREATE OR REPLACE TABLE RAW.ANOMALY_LABELS (
    date            DATE            NOT NULL,
    metric          VARCHAR         NOT NULL,
    anomaly_type    VARCHAR         NOT NULL,
    severity        VARCHAR         NOT NULL,
    magnitude       NUMBER(10, 4)   NOT NULL,
    note            VARCHAR
);

-- ===========================================================================
-- CSV file format + internal stage (non-destructive).
-- ===========================================================================
-- This file is SETUP ONLY. Note: the four tables above use CREATE OR REPLACE
-- TABLE, so re-running ddl.sql recreates them EMPTY (any loaded rows are dropped)
-- -- re-run load.sql afterwards to reload. The file format and stage below are
-- non-destructive (stage uses IF NOT EXISTS), so STAGED files survive a re-run.
-- Loading the CSVs (PUT + COPY INTO) lives in load.sql.
--
-- A CSV file format that matches generate_data's output (header row, comma
-- delimited, double-quoted fields with embedded commas in `note`). CREATE OR
-- REPLACE on a FILE FORMAT only redefines the format - it holds no data.
CREATE OR REPLACE FILE FORMAT RAW.FINSENTRY_CSV
    TYPE = CSV
    FIELD_DELIMITER = ','
    SKIP_HEADER = 1
    FIELD_OPTIONALLY_ENCLOSED_BY = '"'
    NULL_IF = ('', 'NULL')
    EMPTY_FIELD_AS_NULL = TRUE;

-- Internal named stage to upload data/raw/*.csv into.
-- IF NOT EXISTS (NOT "OR REPLACE"): re-running this file must not drop a stage
-- that already holds uploaded files.
CREATE STAGE IF NOT EXISTS RAW.FINSENTRY_RAW_STAGE
    FILE_FORMAT = RAW.FINSENTRY_CSV;

-- Next: run load.sql to PUT the CSVs and COPY INTO the tables.
-- (Re-running THIS file is safe and leaves any staged files in place.)
