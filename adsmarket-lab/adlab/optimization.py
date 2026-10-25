"""Reserve-price optimization as a constrained marketplace-health decision.

A higher CPT floor raises platform revenue only up to a point, and costs fill rate,
incremental installs and advertiser surplus well before that point. We trace the frontier
with global (whole-marketplace) simulations and pick the revenue-maximizing reserve subject
to guardrails on advertiser outcomes.
"""
from __future__ import annotations

import pandas as pd

from .sim import Policy, World, simulate


def reserve_frontier(world: World, reserves, days, n_queries: int) -> pd.DataFrame:
    rows = []
    for r in reserves:
        c = simulate(world, days, Policy(reserve=r), n_queries).country.sum(numeric_only=True)
        rows.append({"reserve": r, "revenue": c.spend, "fill_rate": c.filled / c.queries,
                     "inc_installs": c.inc, "adv_surplus": c.value_inc - c.spend,
                     "true_roas": c.value_inc / c.spend})
    df = pd.DataFrame(rows)
    base = df.iloc[0]
    for m in ["revenue", "inc_installs", "adv_surplus", "fill_rate"]:
        df[f"{m}_vs_base"] = df[m] / base[m] - 1
    return df


def recommend(df: pd.DataFrame, max_surplus_loss: float = .08, max_fill_loss: float = .05) -> pd.Series:
    ok = df[(df.adv_surplus_vs_base >= -max_surplus_loss) & (df.fill_rate_vs_base >= -max_fill_loss)]
    return ok.loc[ok.revenue.idxmax()]
