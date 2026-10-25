import numpy as np
import pandas as pd
import pytest

from adlab import attribution, causal, experiments
from adlab.sim import Policy, make_world, simulate


@pytest.fixture(scope="module")
def world():
    return make_world(seed=3, n_cal=5000)


def test_common_random_numbers(world):
    a = simulate(world, range(2), Policy(reserve=.1), 5000).country
    b = simulate(world, range(2), Policy(reserve=.8), 5000).country
    again = simulate(world, range(2), Policy(reserve=.1), 5000).country
    pd.testing.assert_frame_equal(a, again)                 # deterministic
    assert (a.queries.to_numpy() == b.queries.to_numpy()).all()  # policy never changes demand


def test_prices_respect_reserve(world):
    log = simulate(world, range(2), Policy(reserve=.5), 5000, log=True).log
    taps = log[log.tap]
    assert (taps.spend >= .5 - 1e-9).all()


def test_budgets_bind_but_pacing_limits_overspend(world):
    adv = simulate(world, range(3), Policy(), 5000).adv
    capped = adv.exhaust_hour < 24
    assert .2 < capped.mean() < .8
    # hourly pacing can overshoot by the taps in flight, but not by much in aggregate
    assert adv[capped].spend.sum() <= 1.15 * adv[capped].budget.sum()


def test_budget_split_pools_are_isolated(world):
    adv = simulate(world, range(1), Policy(split="budget", reserve_treat=.5), 5000).adv
    assert set(adv.pool) == {0, 1}
    per_pool = adv.groupby("pool").budget.sum()
    assert np.isclose(per_pool[0], per_pool[1])


def test_ghost_ads_holdout_share(world):
    log = simulate(world, range(3), Policy(holdout_frac=.2), 5000, log=True).log
    assert abs((~log.shown).mean() - .2) < .02
    assert log.loc[~log.shown, "spend"].sum() == 0


def test_aipw_recovers_truth_on_overlap(world):
    obs = simulate(world, range(4), Policy(), 10000, obs=True).obs
    full = causal.estimate_att(obs)
    trimmed = causal.estimate_att(causal.overlap_trimmed(obs, full["e"]))
    dr = trimmed["table"].iloc[3]
    assert dr.ci_lo <= trimmed["truth"] <= dr.ci_hi
    assert full["table"].estimate.iloc[0] > 2 * full["truth"]   # naive is badly confounded


def test_last_tap_overcredits_brand(world):
    log = simulate(world, range(4), Policy(holdout_frac=.1), 5000, log=True).log
    t = attribution.incrementality_table(log, "rel").set_index("rel")
    assert t.loc["own_brand", "inflation_last_tap"] > 2


def test_cuped_keeps_aa_calibrated():
    s = experiments.cuped_power_study(n_adv=2000, n_sims=200)["summary"].set_index(["kind", "cuped"])
    for cuped in [False, True]:
        assert abs(s.loc[("A/A", cuped), "reject_rate"] - .05) < .04
    assert s.loc[("A/B", True), "mean_se"] < s.loc[("A/B", False), "mean_se"]


def test_msprt_controls_peeking():
    r = experiments.peeking_study(n_sims=400)
    assert r["naive_peeking"] > .12
    assert r["msprt"] <= .06
