"""Experiment design for a two-sided ads marketplace.

1. reserve_price_test   -- marketplace interference: query-randomized vs budget-split designs,
                           graded against the true global treatment effect.
2. cuped_power_study    -- advertiser-randomized test: CUPED variance reduction, A/A calibration,
                           power and MDE.
3. peeking_study        -- why daily peeking breaks fixed-horizon tests, and an always-valid
                           mixture SPRT (mSPRT) that fixes it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .sim import Policy, World, simulate


# --------------------------------------------------------------------------------------
# 1. Marketplace interference
# --------------------------------------------------------------------------------------
def _ratio_lift(t: np.ndarray, c: np.ndarray) -> tuple[float, float]:
    """Lift of sum(t)/sum(c) - 1 with a delta-method SE, treating days as iid clusters."""
    r = t.sum() / c.sum()
    n = len(t)
    resid = t - r * c
    se = np.sqrt(resid.var(ddof=1) / n) / c.mean()
    return r - 1, se


def reserve_price_test(world: World, days, r0: float, r1: float, n_queries: int) -> dict:
    """Compare experiment designs for raising the CPT reserve from r0 to r1.

    Truth: the global treatment effect, i.e. the whole marketplace at r1 vs the whole
    marketplace at r0 (same users and coin flips via common random numbers).
    """
    metrics = ["spend", "filled", "inc", "value_inc"]
    g0 = simulate(world, days, Policy(reserve=r0), n_queries).country
    g1 = simulate(world, days, Policy(reserve=r1), n_queries).country
    # SUTVA fails through shared budgets: a constrained advertiser's spend in one arm is
    # budget the other arm can no longer win. Budget-split gives each arm its own half.
    truth = {m: g1[m].sum() / g0[m].sum() - 1 for m in metrics}
    surplus0 = (g0.value_inc - g0.spend).sum()
    truth["adv_surplus"] = (g1.value_inc - g1.spend).sum() / surplus0 - 1

    rows = []
    for split in ["query", "budget"]:
        res = simulate(world, days, Policy(reserve=r0, reserve_treat=r1, split=split), n_queries)
        d = res.country.groupby(["day", "arm"])[metrics].sum().unstack("arm")
        d_s = (res.country.assign(s=lambda x: x.value_inc - x.spend)
               .groupby(["day", "arm"])["s"].sum().unstack("arm"))
        for m in metrics + ["adv_surplus"]:
            src = d_s if m == "adv_surplus" else d[m]
            est, se = _ratio_lift(src[1].to_numpy(), src[0].to_numpy())
            rows.append({"design": f"{split}-randomized", "metric": m, "estimate": est,
                         "ci_lo": est - 1.96 * se, "ci_hi": est + 1.96 * se, "truth": truth[m]})
    out = pd.DataFrame(rows)
    out["bias"] = out.estimate - out.truth
    out["covers"] = (out.ci_lo <= out.truth) & (out.truth <= out.ci_hi)
    return {"table": out, "truth": truth, "r0": r0, "r1": r1}


# --------------------------------------------------------------------------------------
# 2. CUPED + power on an advertiser-randomized test
# --------------------------------------------------------------------------------------
def advertiser_panel(n: int, effect: float, rng: np.random.Generator) -> pd.DataFrame:
    """Heavy-tailed advertiser spend with persistence (pre period predicts post period)."""
    base = rng.normal(4, 1.2, n)
    pre = np.exp(base + rng.normal(0, .35, n))
    treat = rng.random(n) < .5
    post = np.exp(base + rng.normal(0, .35, n)) * np.where(treat, 1 + effect, 1)
    return pd.DataFrame({"pre": pre, "post": post, "treat": treat})


def diff_test(df: pd.DataFrame, cuped: bool, winsor: float | None = .995) -> tuple[float, float]:
    """Relative lift estimate and SE (difference in means / control mean)."""
    y = df.post.to_numpy()
    x = df.pre.to_numpy()
    if winsor:
        cap = np.quantile(y, winsor)
        y = np.minimum(y, cap)
        x = np.minimum(x, np.quantile(x, winsor))
    if cuped:
        theta = np.cov(x, y)[0, 1] / x.var(ddof=1)
        y = y - theta * (x - x.mean())
    t = df.treat.to_numpy()
    yt, yc = y[t], y[~t]
    diff = yt.mean() - yc.mean()
    se = np.sqrt(yt.var(ddof=1) / len(yt) + yc.var(ddof=1) / len(yc))
    base = df.post.to_numpy()[~t].mean()
    return diff / base, se / base


def cuped_power_study(n_adv: int = 4000, effect: float = .03, n_sims: int = 1000,
                      seed: int = 11) -> dict:
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_sims):
        for eff, kind in [(0., "A/A"), (effect, "A/B")]:
            df = advertiser_panel(n_adv, eff, rng)
            for cuped in [False, True]:
                est, se = diff_test(df, cuped)
                rows.append({"sim": s, "kind": kind, "cuped": cuped, "est": est, "se": se,
                             "reject": abs(est / se) > 1.96,
                             "covers": abs(est - eff) < 1.96 * se})
    sims = pd.DataFrame(rows)
    summary = (sims.groupby(["kind", "cuped"])
               .agg(mean_est=("est", "mean"), mean_se=("se", "mean"),
                    reject_rate=("reject", "mean"), coverage=("covers", "mean")).reset_index())
    se_raw = summary.query("kind=='A/B' and not cuped").mean_se.item()
    se_cup = summary.query("kind=='A/B' and cuped").mean_se.item()
    var_reduction = 1 - (se_cup / se_raw) ** 2

    # analytic power curve: SE scales with 1/sqrt(n)
    ns = np.array([500, 1000, 2000, 4000, 8000, 16000])
    curve = []
    for label, se in [("raw", se_raw), ("CUPED", se_cup)]:
        se_n = se * np.sqrt(n_adv / ns)
        power = 1 - stats.norm.cdf(1.96 - effect / se_n) + stats.norm.cdf(-1.96 - effect / se_n)
        mde = 2.8 * se_n          # 80% power, two-sided alpha = .05
        curve += [{"method": label, "n_adv": n, "power": p, "mde": m}
                  for n, p, m in zip(ns, power, mde)]
    return {"summary": summary, "var_reduction": var_reduction, "power_curve": pd.DataFrame(curve),
            "n_adv": n_adv, "effect": effect, "n_sims": n_sims}


# --------------------------------------------------------------------------------------
# 3. Peeking and always-valid inference
# --------------------------------------------------------------------------------------
def msprt_stat(diff: np.ndarray, var: np.ndarray, tau2: float) -> np.ndarray:
    """Normal-mixture SPRT likelihood ratio (Johari et al.) for a difference in means."""
    return np.sqrt(var / (var + tau2)) * np.exp(diff ** 2 * tau2 / (2 * var * (var + tau2)))


def peeking_study(n_sims: int = 2000, days: int = 28, per_day: int = 400, effect: float = 0.,
                  tau: float = .05, seed: int = 5) -> dict:
    """Daily looks at a test on a unit-variance metric with `per_day` new users per arm."""
    rng = np.random.default_rng(seed)
    xt = rng.normal(effect, 1, (n_sims, days, per_day)).mean(2)
    xc = rng.normal(0, 1, (n_sims, days, per_day)).mean(2)
    n = per_day * np.arange(1, days + 1)
    diff = (np.cumsum(xt, 1) - np.cumsum(xc, 1)) * per_day / n
    var = 2 / n
    z = diff / np.sqrt(var)
    naive_any = (np.abs(z) > 1.96).any(1)
    naive_final = np.abs(z[:, -1]) > 1.96
    msprt_any = (msprt_stat(diff, var, tau ** 2) >= 1 / .05).any(1)
    fpr_by_day = pd.DataFrame({
        "day": np.arange(1, days + 1),
        "naive_peeking": np.maximum.accumulate(np.abs(z) > 1.96, axis=1).mean(0),
        "msprt": np.maximum.accumulate(msprt_stat(diff, var, tau ** 2) >= 20, axis=1).mean(0)})
    return {"effect": effect, "fixed_horizon": naive_final.mean(), "naive_peeking": naive_any.mean(),
            "msprt": msprt_any.mean(), "by_day": fpr_by_day}
