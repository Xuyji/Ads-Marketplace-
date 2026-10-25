"""Attribution vs incrementality, measured with a ghost-ads holdout.

In a ghost-ads holdout the auction still runs and the would-be winner is logged, but the ad
is withheld for a random share of auctions. Comparing install rates of shown vs ghost
auctions for the same winner gives an experimental incrementality estimate, which we grade
against the truth and against what last-tap and view-through attribution would report.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def incrementality_table(log: pd.DataFrame, by: str) -> pd.DataFrame:
    rows = []
    for seg, g in log.groupby(by):
        s, h = g[g.shown], g[~g.shown]
        if len(h) < 50:
            continue
        rs, rh = s.install.mean(), h.install.mean()
        se = np.sqrt(rs * (1 - rs) / len(s) + rh * (1 - rh) / len(h))
        inc_est = (rs - rh) * len(s)
        spend = s.spend.sum()
        true_inc = (s.p1 - s.p0).sum()
        attributed = (s.tap & s.install).sum()
        rows.append({
            by: seg, "impressions": len(s), "spend": spend,
            "view_through": int(s.install.sum()), "last_tap": int(attributed),
            "ghost_ads_inc": inc_est, "ghost_ci_lo": (rs - rh - 1.96 * se) * len(s),
            "ghost_ci_hi": (rs - rh + 1.96 * se) * len(s), "true_inc": true_inc,
            "cpa_last_tap": spend / attributed, "icpa_ghost": spend / inc_est,
            "icpa_true": spend / true_inc, "inflation_last_tap": attributed / true_inc})
    out = pd.DataFrame(rows)
    out["ghost_covers"] = (out.ghost_ci_lo <= out.true_inc) & (out.true_inc <= out.ghost_ci_hi)
    return out
