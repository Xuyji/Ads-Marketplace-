-- Metric layer: one governed definition per KPI, so every dashboard, memo and experiment
-- readout computes "fill rate" or "iCPA" the same way.
-- Source tables (written by adlab/metrics.py):
--   fact_country_day(day, date, country, arm, queries, filled, ghost, taps, spend,
--                    installs, attributed, inc, value_inc, value_attr)
--   fact_adv_day(day, adv, pool, wins, taps, spend, attributed, inc, value_inc,
--                value_attr, budget, exhaust_hour, category)

DROP VIEW IF EXISTS kpi_daily;
CREATE VIEW kpi_daily AS
SELECT
    date,
    SUM(queries)                                   AS searches,
    SUM(spend)                                     AS revenue,
    1.0 * SUM(filled) / SUM(queries)               AS fill_rate,
    1.0 * SUM(taps)   / NULLIF(SUM(filled), 0)     AS ttr,           -- tap-through rate
    SUM(spend) / NULLIF(SUM(taps), 0)              AS avg_cpt,       -- cost per tap
    SUM(spend) / NULLIF(SUM(queries), 0) * 1000    AS rpm,           -- revenue per 1k searches
    SUM(spend) / NULLIF(SUM(attributed), 0)        AS cpa_last_tap,
    SUM(spend) / NULLIF(SUM(inc), 0)               AS icpa,          -- incremental CPA
    SUM(attributed) / NULLIF(SUM(inc), 0)          AS attribution_inflation,
    SUM(value_inc) / NULLIF(SUM(spend), 0)         AS incremental_roas
FROM fact_country_day
GROUP BY date;

DROP VIEW IF EXISTS kpi_weekly;
CREATE VIEW kpi_weekly AS
SELECT
    strftime('%Y-W%W', date)                       AS week,
    MIN(date)                                      AS week_start,
    SUM(queries)                                   AS searches,
    SUM(spend)                                     AS revenue,
    1.0 * SUM(filled) / SUM(queries)               AS fill_rate,
    SUM(spend) / NULLIF(SUM(taps), 0)              AS avg_cpt,
    SUM(spend) / NULLIF(SUM(inc), 0)               AS icpa,
    SUM(value_inc) / NULLIF(SUM(spend), 0)         AS incremental_roas
FROM fact_country_day
GROUP BY week;

-- Marketplace health: how much demand is budget-capped (revenue left on the table and
-- the population most exposed to interference in experiments).
DROP VIEW IF EXISTS budget_health_weekly;
CREATE VIEW budget_health_weekly AS
SELECT
    strftime('%Y-W%W', d.date)                             AS week,
    AVG(CASE WHEN a.exhaust_hour < 24 THEN 1.0 ELSE 0 END) AS share_advertiser_days_capped,
    AVG(CASE WHEN a.exhaust_hour < 24 THEN a.exhaust_hour END) AS avg_cap_hour,
    SUM(a.spend) / SUM(a.budget)                           AS budget_utilization
FROM fact_adv_day a
JOIN (SELECT DISTINCT day, date FROM fact_country_day) d USING (day)
GROUP BY week;

DROP VIEW IF EXISTS category_scorecard;
CREATE VIEW category_scorecard AS
SELECT
    category,
    SUM(spend)                                     AS revenue,
    SUM(spend) / NULLIF(SUM(attributed), 0)        AS cpa_last_tap,
    SUM(spend) / NULLIF(SUM(inc), 0)               AS icpa,
    SUM(value_inc) / NULLIF(SUM(spend), 0)         AS incremental_roas,
    AVG(CASE WHEN exhaust_hour < 24 THEN 1.0 ELSE 0 END) AS share_days_capped
FROM fact_adv_day
GROUP BY category
ORDER BY revenue DESC;
