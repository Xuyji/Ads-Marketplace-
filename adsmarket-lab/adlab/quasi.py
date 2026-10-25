"""Quasi-experimental evaluation of a non-randomized geo launch.

Scenario: an improved ad-relevance model ships to four European storefronts in early November.
It was not A/B tested. The launch lands right before the holiday ramp, so a pre/post read is
contaminated by seasonality. We estimate the revenue impact with difference-in-differences
(plus an event study for pre-trends) and synthetic control (plus placebo-in-space inference),
and grade both against the counterfactual simulation in which the launch never happened.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.optimize import minimize

from .sim import Policy, World, calendar, simulate

TREATED = ("GB", "DE", "FR", "IT")


def geo_panel(world: World, start: int, launch: int, end: int, lift: float, n_queries: int):
    days = range(start, end)
    pol = Policy(lift_countries=TREATED, lift=lift, lift_from_day=launch)
    actual = simulate(world, days, pol, n_queries).country
    cf = simulate(world, days, Policy(), n_queries).country
    panel = actual.groupby(["day", "country"]).spend.sum().rename("rev").reset_index()
    panel["rev_cf"] = cf.groupby(["day", "country"]).spend.sum().to_numpy()
    panel["treated"] = panel.country.isin(TREATED)
    panel["post"] = panel.day >= launch
    panel["log_rev"] = np.log(panel.rev)
    panel["date"] = panel.day.map(calendar)
    return panel


def true_att(panel: pd.DataFrame) -> float:
    p = panel[panel.treated & panel.post]
    return p.rev.sum() / p.rev_cf.sum() - 1


def naive_pre_post(panel: pd.DataFrame) -> float:
    p = panel[panel.treated].groupby("post").rev.mean()
    return p[True] / p[False] - 1


def did(panel: pd.DataFrame) -> dict:
    df = panel.assign(tp=(panel.treated & panel.post).astype(int))
    m = smf.ols("log_rev ~ tp + C(country) + C(day)", df).fit(
        cov_type="cluster", cov_kwds={"groups": df.country.astype("category").cat.codes})
    b, se = m.params["tp"], m.bse["tp"]
    return {"estimate": np.exp(b) - 1, "ci_lo": np.exp(b - 1.96 * se) - 1,
            "ci_hi": np.exp(b + 1.96 * se) - 1}


def event_study(panel: pd.DataFrame, launch: int, bin_days: int = 7) -> pd.DataFrame:
    df = panel.copy()
    df["k"] = np.floor((df.day - launch) / bin_days).astype(int)
    ks = sorted(df.k.unique())
    ref = -1
    for k in ks:
        if k != ref:
            df[f"e{k + 100}"] = ((df.k == k) & df.treated).astype(int)
    terms = " + ".join(f"e{k + 100}" for k in ks if k != ref)
    m = smf.ols(f"log_rev ~ {terms} + C(country) + C(day)", df).fit(
        cov_type="cluster", cov_kwds={"groups": df.country.astype("category").cat.codes})
    rows = [{"week": ref, "coef": 0., "ci_lo": 0., "ci_hi": 0.}]
    for k in ks:
        if k == ref:
            continue
        b, se = m.params[f"e{k + 100}"], m.bse[f"e{k + 100}"]
        rows.append({"week": k, "coef": b, "ci_lo": b - 1.96 * se, "ci_hi": b + 1.96 * se})
    return pd.DataFrame(rows).sort_values("week")


def _sc_weights(y_pre: np.ndarray, X_pre: np.ndarray) -> np.ndarray:
    """Simplex weights on demeaned donor series (allows a level difference, Doudchenko-Imbens)."""
    yd = y_pre - y_pre.mean()
    Xd = X_pre - X_pre.mean(0)
    J = X_pre.shape[1]
    res = minimize(lambda w: ((yd - Xd @ w) ** 2).sum(), np.full(J, 1 / J), method="SLSQP",
                   bounds=[(0, 1)] * J, constraints={"type": "eq", "fun": lambda w: w.sum() - 1})
    return res.x


def synthetic_control(panel: pd.DataFrame) -> dict:
    wide = panel.pivot(index="day", columns="country", values="log_rev")
    post = wide.index >= panel[panel.post].day.min()
    donors = [c for c in wide.columns if c not in TREATED]

    def fit(y: pd.Series, pool: list[str]):
        X = wide[pool].to_numpy()
        w = _sc_weights(y[~post].to_numpy(), X[~post])
        synth = X @ w + (y[~post].mean() - (X[~post] @ w).mean())
        gap = y.to_numpy() - synth
        rmspe = lambda g: np.sqrt((g ** 2).mean())
        return w, synth, gap, rmspe(gap[post]) / rmspe(gap[~post])

    # treated unit: log of total treated revenue
    y_t = np.log(np.exp(wide[list(TREATED)]).sum(1))
    w, synth, gap, ratio = fit(y_t, donors)
    placebo = [fit(wide[d], [x for x in donors if x != d])[3] for d in donors]
    p_value = (1 + sum(r >= ratio for r in placebo)) / (1 + len(placebo))
    return {"estimate": float(np.exp(gap[post].mean()) - 1), "weights": pd.Series(w, donors),
            "series": pd.DataFrame({"day": wide.index, "treated": y_t.to_numpy(),
                                    "synthetic": synth, "post": post}),
            "p_value": p_value, "rmspe_ratio": ratio, "placebo_ratios": placebo}


def spillover(panel: pd.DataFrame) -> dict:
    """Shared advertiser budgets cross borders: extra spend in treated geos is budget the
    control geos no longer get. DiD then measures direct effect minus the control-geo loss."""
    p = panel[panel.post]
    lift = lambda s: s.rev.sum() / s.rev_cf.sum() - 1
    return {"treated": lift(p[p.treated]), "donors": lift(p[~p.treated]), "global": lift(p)}


def geo_launch_study(world: World, start=250, launch=310, end=370, lift=.10, n_queries=20_000):
    panel = geo_panel(world, start, launch, end, lift, n_queries)
    sc = synthetic_control(panel)
    d = did(panel)
    table = pd.DataFrame([
        {"method": "Naive pre/post (treated only)", "estimate": naive_pre_post(panel)},
        {"method": "Difference-in-differences (TWFE)", **d},
        {"method": "Synthetic control", "estimate": sc["estimate"]}])
    table["truth"] = true_att(panel)
    table["rel_error"] = table.estimate / table.truth - 1
    return {"panel": panel, "table": table, "event_study": event_study(panel, launch),
            "sc": sc, "launch": launch, "spillover": spillover(panel)}


def geo_monte_carlo(seeds, n_queries: int = 10_000, **kw) -> pd.DataFrame:
    """Grade the estimators across independently simulated marketplaces: bias and coverage."""
    from .sim import make_world
    rows = []
    for s in seeds:
        r = geo_launch_study(make_world(seed=s), n_queries=n_queries, **kw)
        for _, row in r["table"].iterrows():
            rows.append({"seed": s, **row.to_dict()})
    return pd.DataFrame(rows)
