"""Run every study end to end and write outputs/ (report.html, figures, CSVs, warehouse).

    python run_all.py           # full run (~6 min on a laptop)
    python run_all.py --fast    # fewer simulations, for a quick check
    python run_all.py --report-only   # re-render figures and report from cached results
"""
from __future__ import annotations

import argparse
import pickle
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from adlab import attribution, causal, experiments, forecasting, metrics, optimization, quasi, report
from adlab.sim import Policy, calendar, make_world, simulate, usd

warnings.filterwarnings("ignore")
OUT = Path(__file__).parent / "outputs"
N = 20_000            # simulated queries per day for the analysis windows
N_HIST = 8_000        # lighter volume for the two-year history
WINDOW = range(400, 414)  # Feb 2025


def step(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def scorecard(R) -> pd.DataFrame:
    rows = []

    def add(study, method, est, truth, lo=None, hi=None):
        ok = (lo <= truth <= hi) if lo is not None and not np.isnan(lo) else abs(est - truth) <= .15 * abs(truth)
        rows.append({"study": study, "method": method, "estimate": est, "truth": truth,
                     "error": (est - truth) * 100, "recovers": bool(ok)})

    t = R["reserve_test"]["table"].query("metric=='spend'")
    for _, r in t.iterrows():
        add("Reserve-price A/B: revenue lift", r.design, r.estimate, r.truth, r.ci_lo, r.ci_hi)
    for key, label in [("causal_full", "all auctions"), ("causal_trim", "non-brand overlap")]:
        for _, r in R[key]["table"].iloc[[0, 3]].iterrows():
            add(f"Effect of winning on installs ({label})", r.method, r.estimate, r.truth,
                r.ci_lo, r.ci_hi)
    a = R["attr_rel"].set_index("rel").loc["own_brand"]
    imp = a.impressions
    add("Own-brand incremental installs / impression", "Last-tap attribution", a.last_tap / imp, a.true_inc / imp)
    add("Own-brand incremental installs / impression", "Ghost-ads holdout", a.ghost_ads_inc / imp,
        a.true_inc / imp, a.ghost_ci_lo / imp, a.ghost_ci_hi / imp)
    for _, r in R["geo"]["table"].iterrows():
        add("Geo launch: revenue lift (headline world)", r.method, r.estimate, r.truth,
            r.get("ci_lo", np.nan), r.get("ci_hi", np.nan))
    mc = R["geo_mc"]
    n = mc.seed.nunique()
    for method, g in mc.groupby("method", sort=False):
        add(f"Geo launch: mean over {n} simulated marketplaces", method, g.estimate.mean(), g.truth.mean())
    return pd.DataFrame(rows)


def main(fast: bool, report_only: bool = False):
    OUT.mkdir(exist_ok=True)
    if report_only:
        R = pickle.loads((OUT / "results.pkl").read_bytes())
        report.write_html(R, report.make_figures(R, OUT / "figures"), OUT / "report.html")
        return step("report regenerated from outputs/results.pkl")
    R = {}
    step("building world")
    world = make_world(seed=7)

    step("reserve-price experiment: query-split vs budget-split vs global truth")
    R["reserve_test"] = experiments.reserve_price_test(world, WINDOW, .10, .50, N)

    step("reserve-price frontier")
    R["frontier"] = optimization.reserve_frontier(
        world, [.10, .25, .40, .50, .60, .75, 1.0, 1.5], WINDOW[:7], N)
    R["reco"] = optimization.recommend(R["frontier"])

    step("CUPED / power / peeking")
    R["cuped"] = experiments.cuped_power_study(n_adv=6000, effect=.04, n_sims=150 if fast else 1000)
    R["peek_null"] = experiments.peeking_study(n_sims=500 if fast else 2000)
    R["peek_alt"] = experiments.peeking_study(n_sims=500 if fast else 2000, effect=.05)

    step("observational causal inference")
    obs = simulate(world, WINDOW[:7], Policy(), N, obs=True).obs
    R["causal_full"] = causal.estimate_att(obs)
    R["causal_trim"] = causal.estimate_att(causal.overlap_trimmed(obs, R["causal_full"]["e"]))
    R["causal_seg"] = causal.att_by_segment(obs)

    step("ghost-ads holdout & attribution")
    log = simulate(world, WINDOW, Policy(holdout_frac=.10), N, log=True).log
    R["attr_rel"] = attribution.incrementality_table(log, "rel")
    R["attr_cat"] = attribution.incrementality_table(log, "category")

    step("geo launch: DiD, event study, synthetic control")
    R["geo"] = quasi.geo_launch_study(world, n_queries=N)
    mc = quasi.geo_monte_carlo(range(101, 104) if fast else range(101, 109))
    mc["err"] = mc.estimate - mc.truth
    mc["covered"] = np.where(mc.ci_lo.notna(), (mc.ci_lo <= mc.truth) & (mc.truth <= mc.ci_hi), np.nan)
    R["geo_mc"] = mc
    R["geo_mc_summary"] = (mc.groupby("method", sort=False)
                           .agg(bias=("err", "mean"), rmse=("err", lambda e: np.sqrt((e ** 2).mean())),
                                ci_coverage=("covered", "mean")).reset_index())

    step("two-year history, KPI warehouse, forecasting")
    hist = simulate(world, range(731), Policy(), N_HIST)
    con = metrics.build_warehouse(hist, world, OUT / "warehouse.sqlite")
    R["category_scorecard"] = metrics.query(con, "category_scorecard").assign(
        revenue=lambda d: usd(d.revenue, N_HIST))
    metrics.query(con, "kpi_weekly").to_csv(OUT / "kpi_weekly.csv", index=False)
    metrics.query(con, "budget_health_weekly").to_csv(OUT / "budget_health_weekly.csv", index=False)
    daily = hist.country.groupby("day")[["spend", "queries"]].sum()
    daily.index = daily.index.map(calendar)
    rev = usd(daily.spend, N_HIST)
    rows = []
    for name, series in [("revenue", rev), ("searches", daily.queries * 1.0)]:
        bt = forecasting.backtest(series, n_origins=5 if fast else 8)
        s = forecasting.score(bt)
        s["coverage_80"] = [forecasting.interval_coverage(bt, m) for m in s.index]
        rows.append(s.assign(series=name).reset_index())
        if name == "revenue":
            # Finance plans on the unbiased ensemble: a systematically low plan costs more
            # than a fraction of a point of WAPE.
            R["forecast_scores"] = s
            R["best_model"] = "ensemble"
            R["coverage"] = s.loc["ensemble", "coverage_80"]
            R["plan"] = forecasting.plan_forecast(series, bt, R["best_model"], 91)
    ft = pd.concat(rows)[["series", "model", "WAPE", "MAPE", "bias", "coverage_80"]]
    R["forecast_table"] = ft.round(4)
    R["rev_series"] = rev

    step("scorecard, figures, report")
    R["scorecard"] = scorecard(R)
    R["scorecard"].to_csv(OUT / "scorecard.csv", index=False)
    R["reserve_test"]["table"].to_csv(OUT / "reserve_test.csv", index=False)
    R["frontier"].to_csv(OUT / "reserve_frontier.csv", index=False)
    R["attr_rel"].to_csv(OUT / "attribution_by_keyword.csv", index=False)
    R["attr_cat"].to_csv(OUT / "attribution_by_category.csv", index=False)
    R["plan"].to_csv(OUT / "revenue_plan_90d.csv", index=False)
    ft.to_csv(OUT / "forecast_backtest.csv", index=False)
    (OUT / "results.pkl").write_bytes(pickle.dumps(R))
    figs = report.make_figures(R, OUT / "figures")
    report.write_html(R, figs, OUT / "report.html")
    pd.set_option("display.width", 200)
    print(R["scorecard"].round(4).to_string(index=False))
    step(f"done -> {OUT / 'report.html'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--report-only", action="store_true", help="re-render from outputs/results.pkl")
    a = ap.parse_args()
    main(a.fast, a.report_only)
