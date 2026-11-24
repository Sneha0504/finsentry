-- ===========================================================================
-- FinSentry - load the RAW CSVs into Snowflake (run AFTER ddl.sql)
-- ===========================================================================
-- Separated from ddl.sql so setup (schema/tables/format/stage) and data loading
-- are independent: re-running ddl.sql can never delete staged files, and this
-- file can be re-run to reload. Run from the repo ROOT in a snowsql session so
-- the `file://data/raw/...` paths resolve.
--
--   snowsql -a <account> -u <user> -f snowflake/load.sql
--
-- Data is SYNTHETIC (fixed seed 42). Expected row counts: 21900 / 1825 / 1825 / 18.
-- ===========================================================================

USE DATABASE FINSENTRY;
USE WAREHOUSE FINSENTRY_WH;
USE SCHEMA RAW;

-- --- 1. Upload the local CSVs to the internal stage --------------------------
-- OVERWRITE=TRUE so re-running replaces the staged copy rather than erroring on a
-- name clash (it does not touch the tables).
PUT file://data/raw/billing_line_items.csv @RAW.FINSENTRY_RAW_STAGE AUTO_COMPRESS=TRUE OVERWRITE=TRUE;
PUT file://data/raw/revenue_ledger.csv     @RAW.FINSENTRY_RAW_STAGE AUTO_COMPRESS=TRUE OVERWRITE=TRUE;
PUT file://data/raw/consumption.csv        @RAW.FINSENTRY_RAW_STAGE AUTO_COMPRESS=TRUE OVERWRITE=TRUE;
PUT file://data/raw/anomaly_labels.csv     @RAW.FINSENTRY_RAW_STAGE AUTO_COMPRESS=TRUE OVERWRITE=TRUE;

-- --- 2. Load each table (TRUNCATE first so a reload is idempotent) -----------
-- Columns are in CSV order, so a positional COPY works.
TRUNCATE TABLE IF EXISTS RAW.BILLING_LINE_ITEMS;
COPY INTO RAW.BILLING_LINE_ITEMS
    FROM @RAW.FINSENTRY_RAW_STAGE/billing_line_items.csv.gz
    FILE_FORMAT = (FORMAT_NAME = RAW.FINSENTRY_CSV)
    ON_ERROR = 'ABORT_STATEMENT';

TRUNCATE TABLE IF EXISTS RAW.REVENUE_LEDGER;
COPY INTO RAW.REVENUE_LEDGER
    FROM @RAW.FINSENTRY_RAW_STAGE/revenue_ledger.csv.gz
    FILE_FORMAT = (FORMAT_NAME = RAW.FINSENTRY_CSV)
    ON_ERROR = 'ABORT_STATEMENT';

TRUNCATE TABLE IF EXISTS RAW.CONSUMPTION;
COPY INTO RAW.CONSUMPTION
    FROM @RAW.FINSENTRY_RAW_STAGE/consumption.csv.gz
    FILE_FORMAT = (FORMAT_NAME = RAW.FINSENTRY_CSV)
    ON_ERROR = 'ABORT_STATEMENT';

TRUNCATE TABLE IF EXISTS RAW.ANOMALY_LABELS;
COPY INTO RAW.ANOMALY_LABELS
    FROM @RAW.FINSENTRY_RAW_STAGE/anomaly_labels.csv.gz
    FILE_FORMAT = (FORMAT_NAME = RAW.FINSENTRY_CSV)
    ON_ERROR = 'ABORT_STATEMENT';

-- --- 3. Sanity checks (expected: 21900 / 1825 / 1825 / 18) ------------------
SELECT 'billing_line_items' AS tbl, COUNT(*) AS n FROM RAW.BILLING_LINE_ITEMS
UNION ALL SELECT 'revenue_ledger', COUNT(*) FROM RAW.REVENUE_LEDGER
UNION ALL SELECT 'consumption',    COUNT(*) FROM RAW.CONSUMPTION
UNION ALL SELECT 'anomaly_labels', COUNT(*) FROM RAW.ANOMALY_LABELS;
