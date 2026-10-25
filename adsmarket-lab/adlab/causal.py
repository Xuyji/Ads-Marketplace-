"""Causal inference on observational auction logs.

Question: among auctions where an advertiser was eligible, what did *winning* the auction do
to the probability the user installed that advertiser's app (ATT)?

Winning is confounded: advertisers bid up returning users (who install organically more),
and engaged users have higher predicted tap rates, which raises the ad's auction score.
Estimators: naive difference, OLS adjustment, IPW, and cross-fitted AIPW (doubly robust),
each compared against the simulator's true ATT = mean(p1 - p0 | won).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


def features(obs: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame({
        "returning": obs.returning.astype(float), "eng": obs.eng,
        "log_bid": np.log(obs.bid), "logit_pctr": np.log(obs.pctr / (1 - obs.pctr)),
        "log_n_elig": np.log(obs.n_elig), "hour": obs.hour})
    dummies = pd.get_dummies(obs[["rel", "country", "category"]], dtype=float, drop_first=True)
    return pd.concat([X, dummies], axis=1)


def smd(X: pd.DataFrame, t: np.ndarray, w: np.ndarray | None = None) -> pd.Series:
    """Standardized mean difference of each covariate, treated vs (weighted) control."""
    w = np.ones(len(t)) if w is None else w
    mt = X[t].mean()
    mc = (X[~t].mul(w[~t], axis=0)).sum() / w[~t].sum()
    sd = np.sqrt((X[t].var() + X[~t].var()) / 2).replace(0, np.nan)
    return ((mt - mc) / sd).fillna(0)


def estimate_att(obs: pd.DataFrame, n_folds: int = 2, seed: int = 0) -> dict:
    X = features(obs)
    t = obs.won.to_numpy().astype(bool)
    y = obs.y.to_numpy().astype(float)
    truth = float((obs.p1 - obs.p0)[t].mean())
    n1 = t.sum()
    rows = []

    # naive
    est = y[t].mean() - y[~t].mean()
    se = np.sqrt(y[t].var() / n1 + y[~t].var() / (~t).sum())
    rows.append(("Naive won-vs-lost", est, se))

    # OLS regression adjustment (linear outcome model fit on controls)
    Xs = StandardScaler().fit_transform(X)
    lin = LinearRegression().fit(Xs[~t], y[~t])
    est = (y[t] - lin.predict(Xs[t])).mean()
    rows.append(("OLS adjustment", est, np.nan))

    # cross-fitted nuisances: propensity e(x) and control outcome model m0(x)
    e = np.zeros(len(y))
    m0 = np.zeros(len(y))
    for tr, te in KFold(n_folds, shuffle=True, random_state=seed).split(X):
        ps = HistGradientBoostingClassifier(max_iter=200, learning_rate=.08, random_state=seed)
        e[te] = ps.fit(X.iloc[tr], t[tr]).predict_proba(X.iloc[te])[:, 1]
        om = HistGradientBoostingClassifier(max_iter=200, learning_rate=.08, random_state=seed)
        ctrl = tr[~t[tr]]
        m0[te] = om.fit(X.iloc[ctrl], y[ctrl]).predict_proba(X.iloc[te])[:, 1]
    e = np.clip(e, .01, .99)
    odds = e / (1 - e)

    # IPW (ATT weights: treated 1, controls e/(1-e)), Hajek-normalized
    w_c = odds[~t]
    est = y[t].mean() - (w_c * y[~t]).sum() / w_c.sum()
    rows.append(("IPW (ATT weights)", est, np.nan))

    # AIPW / doubly robust ATT with influence-function SE
    p = n1 / len(y)
    resid = y - m0
    psi = (t * resid - (1 - t) * odds * resid) / p
    est = psi.mean()
    psi_c = (t * resid - (1 - t) * odds * resid - t * est) / p
    rows.append(("AIPW, cross-fitted (DR)", est, psi_c.std() / np.sqrt(len(y))))

    out = pd.DataFrame(rows, columns=["method", "estimate", "se"])
    out["truth"] = truth
    out["ci_lo"] = out.estimate - 1.96 * out.se
    out["ci_hi"] = out.estimate + 1.96 * out.se
    out["rel_error"] = out.estimate / truth - 1

    w = np.where(t, 1., odds)
    balance = pd.DataFrame({"before": smd(X, t), "after_weighting": smd(X, t, w)})
    balance = balance.reindex(balance.before.abs().sort_values(ascending=False).index)
    return {"table": out, "truth": truth, "balance": balance, "e": e, "t": t,
            "n": len(y), "n_treated": int(n1)}


def overlap_trimmed(obs: pd.DataFrame, e: np.ndarray, lo: float = .05, hi: float = .95,
                    drop_rel: tuple = ("own_brand",)) -> pd.DataFrame:
    """Restrict to common support (Crump et al. style). The estimand changes to the ATT on
    the overlap population -- the truth is recomputed on exactly that population.

    Own-brand auctions are dropped outright: owners almost never lose their own brand
    keyword, so the handful of losers that survive trimming carry huge weights and leave
    residual imbalance. Brand incrementality is not identified from logs; it needs an
    experiment (see attribution.py, ghost-ads holdout)."""
    keep = (e >= lo) & (e <= hi) & ~obs.rel.isin(drop_rel).to_numpy()
    return obs[keep].reset_index(drop=True)


def att_by_segment(obs: pd.DataFrame, col: str = "rel") -> pd.DataFrame:
    """Doubly robust ATT within each segment (heterogeneous incrementality)."""
    rows = []
    for seg, g in obs.groupby(col):
        if g.won.sum() < 200 or (~g.won).sum() < 200:
            continue
        r = estimate_att(g)
        dr = r["table"].set_index("method").loc["AIPW, cross-fitted (DR)"]
        naive = r["table"].set_index("method").loc["Naive won-vs-lost"]
        rows.append({col: seg, "n": len(g), "naive": naive.estimate, "aipw": dr.estimate,
                     "ci_lo": dr.ci_lo, "ci_hi": dr.ci_hi, "truth": r["truth"],
                     "baseline_install_rate": g.p0[g.won].mean()})
    return pd.DataFrame(rows)
