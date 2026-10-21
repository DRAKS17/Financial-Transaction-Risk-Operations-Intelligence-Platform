-- ============================================================================
-- 01_create_schemas.sql
-- Financial Transaction Risk & Operations Intelligence Platform
-- Architecture: Staging (stg) -> Dimensional Warehouse (core) -> Data Marts (mart)
-- ============================================================================

-- Staging schema: 1:1 landing zone for cleaned source and operational feeds
CREATE SCHEMA IF NOT EXISTS stg;

-- Core schema: Normalized dimensional star schema (facts & dimensions)
CREATE SCHEMA IF NOT EXISTS core;

-- Mart schema: Analytical views and pre-aggregated KPI tables for Power BI
CREATE SCHEMA IF NOT EXISTS mart;

-- Grant permissions (if running under multi-user environment)
COMMENT ON SCHEMA stg IS 'Staging layer: Raw normalized tabular extractions';
COMMENT ON SCHEMA core IS 'Dimensional warehouse layer: Star schema facts and dimensions';
COMMENT ON SCHEMA mart IS 'Analytics mart layer: Optimized business views for reporting and Power BI';
