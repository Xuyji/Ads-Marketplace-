"""Ground-truth simulator of a single-slot search-ads marketplace (App Store style).

Every search query runs a generalized second-price (GSP) auction for one ad slot at the
top of results, priced per tap (CPT). Advertisers have daily budgets that pace out over
the day, bid multipliers for returning users, and keyword targeting across their own
brand, competitor brands, generic and off-category terms.

The key design choice: for every auction we know BOTH potential outcomes -- the probability
the user installs the winning app organically (no ad) and with the ad. So the true
incremental effect of any ad, policy, experiment or launch is known exactly, and every
estimator in this repo is graded against the truth rather than against another estimator.

Common random numbers: all randomness for a day is drawn from a seed that depends only on
(world seed, day), never on the policy. Two simulations that differ only in policy see the
same users, searches and coin flips, so their difference is the policy's causal effect.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

COUNTRIES = np.array(["US", "GB", "DE", "FR", "IT", "JP", "KR", "CA", "AU", "BR", "MX", "IN"])
COUNTRY_WEIGHT = np.array([.30, .07, .07, .06, .04, .10, .05, .05, .04, .08, .06, .08])
COUNTRY_VALUE = np.array([1.0, .85, .85, .80, .70, .95, .80, .85, .85, .35, .40, .20])
COUNTRY_GROWTH = np.array([.18, .15, .15, .15, .13, .10, .14, .18, .17, .32, .28, .45])

CATEGORIES = np.array(["Games", "Finance", "Shopping", "Social",
                       "Health", "Travel", "Productivity", "Entertainment"])
CAT_VALUE = np.array([3.0, 18.0, 6.0, 2.5, 5.0, 8.0, 7.0, 3.5])   # median $ per install
CAT_SEASON_PEAK = np.array([355, 15, 335, 200, 10, 170, 250, 355])  # day of year
CAT_SEASON_AMP = np.array([.10, .08, .25, .05, .20, .20, .08, .12])

# How the searched keyword relates to the app whose ad is shown.
KW_REL = np.array(["own_brand", "competitor_brand", "generic", "off_category"])
REL_TARGET_P = np.array([1.0, .25, .60, .03])     # P(advertiser targets keyword)
REL_TTR = np.array([.40, .05, .08, .015])         # base tap-through rate of the ad
REL_P0 = np.array([.55, .004, .012, .001])        # base organic install probability
REL_BID_MULT = np.array([1.0, 1.15, 1.0, .6])

WEEKLY = np.array([.95, .93, .94, .97, 1.03, 1.12, 1.08])           # Mon..Sun
DIURNAL = np.array([2, 1, .7, .5, .5, .8, 1.5, 2.5, 3.5, 4, 4.2, 4.5,
                    4.8, 4.6, 4.4, 4.5, 4.8, 5.2, 5.8, 6.3, 6.5, 6, 4.8, 3.3])
DIURNAL = DIURNAL / DIURNAL.sum()

SCALE = 1000                      # each simulated query stands for 1,000 real searches
EPOCH = pd.Timestamp("2024-01-01")  # simulated day 0


@dataclass(frozen=True)
class Policy:
    """Marketplace configuration for a simulation run (what the platform controls)."""
    reserve: float = 0.10              # CPT floor for control / everyone
    reserve_treat: float | None = None  # CPT floor for the treatment arm
    split: str = "none"                # "none" | "query" | "budget"
    treat_frac: float = 0.5
    holdout_frac: float = 0.0          # ghost-ads holdout: winner computed, ad withheld
    lift_countries: tuple = ()         # countries that get an improved relevance model
    lift: float = 0.0                  # relative tap-through improvement in those countries
    lift_from_day: int = 10**9


@dataclass
class World:
    seed: int
    adv_cat: np.ndarray          # (A,)
    value: np.ndarray            # (A,) $ value per install to the advertiser
    cvr: np.ndarray              # (A,) P(install | tap)
    returning_mult: np.ndarray   # (A,) bid multiplier on returning users
    budget: np.ndarray           # (A,) daily budget at calibration volume, day 0
    kw_cat: np.ndarray           # (K,)
    kw_pop: np.ndarray           # (K,) base search share
    rel: np.ndarray              # (K, A) code into KW_REL
    target: np.ndarray           # (K, A) bool
    ttr: np.ndarray              # (K, A)
    p0: np.ndarray               # (K, A)
    bid: np.ndarray              # (K, A) CPT bid, 0 when not targeted
    shocks: np.ndarray           # (D, C) AR(1) log demand shocks
    n_cal: int                   # queries/day the budgets were calibrated at

    @property
    def n_adv(self) -> int:
        return len(self.adv_cat)


# --------------------------------------------------------------------------------------
# World construction
# --------------------------------------------------------------------------------------
def make_world(seed: int = 7, n_adv_per_cat: int = 15, n_generic_per_cat: int = 15,
               horizon: int = 800, n_cal: int = 20_000) -> World:
    rng = np.random.default_rng(seed)
    n_cat = len(CATEGORIES)
    A = n_adv_per_cat * n_cat
    adv_cat = np.repeat(np.arange(n_cat), n_adv_per_cat)
    app_size = rng.lognormal(0, 1.0, A)
    value = CAT_VALUE[adv_cat] * rng.lognormal(0, .4, A)
    cvr = rng.uniform(.35, .65, A)
    returning_mult = rng.uniform(1.0, 1.8, A)

    # one brand keyword per app plus generic keywords per category
    kw_cat = np.concatenate([adv_cat, np.repeat(np.arange(n_cat), n_generic_per_cat)])
    kw_owner = np.concatenate([np.arange(A), -np.ones(n_cat * n_generic_per_cat, int)])
    kw_pop = np.concatenate([app_size * .6, rng.lognormal(.3, .9, n_cat * n_generic_per_cat)])
    kw_pop = kw_pop / kw_pop.sum()
    K = len(kw_cat)

    same_cat = kw_cat[:, None] == adv_cat[None, :]
    own = kw_owner[:, None] == np.arange(A)[None, :]
    is_brand = (kw_owner >= 0)[:, None]
    rel = np.where(own, 0, np.where(same_cat & is_brand, 1, np.where(same_cat, 2, 3)))
    target = rng.random((K, A)) < REL_TARGET_P[rel]
    ttr = REL_TTR[rel] * rng.lognormal(0, .3, (K, A))
    p0 = REL_P0[rel] * rng.lognormal(0, .3, (K, A))
    # Advertisers bid on *attributed* value per tap, shaded. They cannot see incrementality.
    shade = rng.uniform(.4, .8, A)
    bid = np.where(target, value * cvr * shade * REL_BID_MULT[rel] * rng.lognormal(0, .25, (K, A)), 0.)

    shocks = np.zeros((horizon, len(COUNTRIES)))
    eps = rng.normal(0, .03, (horizon, len(COUNTRIES)))
    for d in range(1, horizon):
        shocks[d] = .7 * shocks[d - 1] + eps[d]

    world = World(seed, adv_cat, value, cvr, returning_mult, np.full(A, np.inf), kw_cat,
                  kw_pop, rel, target, ttr, p0, bid, shocks, n_cal)

    # Calibrate budgets: unconstrained spend over a few days, then budgets that bind for
    # roughly half the advertisers (budget-constrained marketplaces are where interference bites).
    cal = simulate(world, range(3), Policy(), n_queries=n_cal)
    spend = cal.adv.groupby("adv")["spend"].sum().reindex(range(A), fill_value=0).to_numpy() / 3
    floor = np.quantile(spend[spend > 0], .1)
    world.budget = np.maximum(spend, floor) * rng.uniform(.5, 1.9, A)
    return world


# --------------------------------------------------------------------------------------
# Demand
# --------------------------------------------------------------------------------------
def usd(x, n_queries: int = 20_000):
    """Simulated spend -> display dollars (SCALE real searches per query at 20k queries/day)."""
    return x * SCALE * 20_000 / n_queries


def calendar(day: int) -> pd.Timestamp:
    return EPOCH + pd.Timedelta(days=int(day))


def holiday_mult(date: pd.Timestamp) -> float:
    md = (date.month, date.day)
    if md in [(12, 25), (12, 26)]:
        return 1.35          # new-device day: install spike
    if md in [(12, 31), (1, 1)]:
        return 1.12
    # Black Friday week: the 4th Thursday of November through the following Wednesday
    if date.month == 11:
        first_thu = 1 + (3 - pd.Timestamp(date.year, 11, 1).dayofweek) % 7
        bf = first_thu + 21 + 1
        if bf - 1 <= date.day <= bf + 4:
            return 1.15
    return 1.0


def demand(world: World, day: int) -> tuple[np.ndarray, np.ndarray]:
    """Expected share of daily volume per country, and keyword probabilities, for a day."""
    date = calendar(day)
    doy = date.dayofyear
    yearly = 1 + .06 * np.cos(2 * np.pi * (doy - 355) / 365)
    country = (COUNTRY_WEIGHT * (1 + COUNTRY_GROWTH) ** (day / 365) * WEEKLY[date.dayofweek]
               * yearly * holiday_mult(date) * np.exp(world.shocks[day]))
    cat_mult = 1 + CAT_SEASON_AMP * np.cos(2 * np.pi * (doy - CAT_SEASON_PEAK) / 365)
    kw_p = world.kw_pop * cat_mult[world.kw_cat]
    return country, kw_p / kw_p.sum()


def budget_growth(day: int) -> float:
    """Advertisers grow budgets with the platform's traffic trend (not with seasonality)."""
    return float((COUNTRY_WEIGHT * (1 + COUNTRY_GROWTH) ** (day / 365)).sum())


# --------------------------------------------------------------------------------------
# One day of auctions
# --------------------------------------------------------------------------------------
@dataclass
class SimResult:
    country: pd.DataFrame       # day x country x arm aggregates
    adv: pd.DataFrame           # day x advertiser x budget-pool aggregates
    log: pd.DataFrame | None    # auction-level log of winners (shown or ghost)
    obs: pd.DataFrame | None    # observational (query, eligible advertiser) pairs


def run_day(world: World, day: int, policy: Policy, n_queries: int,
            log: bool = False, obs: bool = False) -> SimResult:
    rng = np.random.default_rng([world.seed, 10_000 + day])
    country_mult, kw_p = demand(world, day)
    counts = rng.poisson(n_queries * country_mult)
    ctry = np.repeat(np.arange(len(COUNTRIES)), counts)
    Q = len(ctry)
    kw = rng.choice(len(kw_p), size=Q, p=kw_p)
    hour = rng.choice(24, size=Q, p=DIURNAL)
    ret = rng.random(Q) < .30                         # returning user (has used app before)
    eng = rng.beta(2, 5, Q)                            # engagement score
    u = rng.random((8, Q))                             # common random numbers
    u_arm, u_hold, u_tap, u_org, u_cvr, u_pick, u_org_j, _ = u

    tap_mult = np.exp(.6 * (eng - .3)) * np.where(ret, 1.3, 1.)
    org_mult = np.exp(1.2 * (eng - .3)) * np.where(ret, 1.5, 1.)
    if policy.lift and day >= policy.lift_from_day:
        lifted = np.isin(COUNTRIES[ctry], policy.lift_countries)
        tap_mult = tap_mult * np.where(lifted, 1 + policy.lift, 1.)

    arm = (u_arm < policy.treat_frac).astype(int) if policy.split != "none" else np.zeros(Q, int)
    r_t = policy.reserve if policy.reserve_treat is None else policy.reserve_treat
    reserve = np.where(arm == 1, r_t, policy.reserve)
    budget_day = world.budget * (n_queries / world.n_cal) * budget_growth(day)
    if policy.split == "budget":
        pool = arm
        remaining = np.vstack([budget_day * (1 - policy.treat_frac), budget_day * policy.treat_frac])
    else:
        pool = np.zeros(Q, int)
        remaining = budget_day[None, :].copy()
    budget_pool = remaining.copy()
    exhaust_hour = np.full(remaining.shape, 24.)

    win = np.full(Q, -1)
    price = np.zeros(Q)
    p_tap = np.zeros(Q)
    p0w = np.zeros(Q)
    p1w = np.zeros(Q)
    n_elig = np.zeros(Q, int)
    obs_rows = []

    for h in range(24):
        idx = np.flatnonzero(hour == h)
        if idx.size == 0:
            continue
        k = kw[idx]
        n = idx.size
        ar = np.arange(n)
        b = (world.bid[k] * (1 + (world.returning_mult - 1)[None, :] * ret[idx, None])
             * COUNTRY_VALUE[ctry[idx], None])
        t = np.minimum(world.ttr[k] * tap_mult[idx, None], .95)
        active = remaining[pool[idx]] > 0
        elig = world.target[k] & active & (b >= reserve[idx, None])
        ne = elig.sum(1)
        n_elig[idx] = ne
        score = np.where(elig, b * t, 0.)
        w = score.argmax(1)
        s1 = score[ar, w]
        has = s1 > 0

        if obs:
            # one random eligible advertiser per contested auction: did winning cause installs?
            m = ne >= 2
            target_rank = np.floor(u_pick[idx] * ne).astype(int) + 1
            j = (elig.cumsum(1) >= target_rank[:, None]).argmax(1)
            p0j = np.minimum(world.p0[k, j] * org_mult[idx], .95)
            p1j = p0j + (1 - p0j) * t[ar, j] * world.cvr[j]
            obs_rows.append(pd.DataFrame({
                "q": idx[m], "adv": j[m], "won": (w == j)[m] & has[m],
                "bid": b[ar, j][m], "pctr": t[ar, j][m], "n_elig": ne[m],
                "rel": world.rel[k, j][m], "p0": p0j[m], "p1": p1j[m]}))

        score[ar, w] = 0.
        s2 = score.max(1)
        tw = t[ar, w]
        pr = np.minimum(np.maximum(reserve[idx], s2 / tw), b[ar, w])   # GSP, never above bid
        q0 = np.minimum(world.p0[k, w] * org_mult[idx], .95)

        win[idx] = np.where(has, w, -1)
        price[idx] = np.where(has, pr, 0.)
        p_tap[idx] = np.where(has, tw, 0.)
        p0w[idx] = np.where(has, q0, 0.)
        p1w[idx] = np.where(has, q0 + (1 - q0) * tw * world.cvr[w], 0.)

        shown_h = has & (u_hold[idx] >= policy.holdout_frac)
        tap_h = shown_h & (u_tap[idx] < tw)
        spend_h = np.where(tap_h, pr, 0.)
        np.add.at(remaining, (pool[idx][tap_h], w[tap_h]), -spend_h[tap_h])
        newly = (remaining <= 0) & (exhaust_hour == 24.)
        exhaust_hour[newly] = h + 1

    has = win >= 0
    ghost = has & (u_hold < policy.holdout_frac)
    shown = has & ~ghost
    tap = shown & (u_tap < p_tap)
    organic = has & (u_org < p0w)
    ad_install = tap & (u_cvr < world.cvr[np.maximum(win, 0)])
    install = np.where(shown, organic | ad_install, organic)   # installs of the winning app
    spend = np.where(tap, price, 0.)
    inc = np.where(shown, p1w - p0w, 0.)                     # true expected incremental installs
    value = world.value[np.maximum(win, 0)] * COUNTRY_VALUE[ctry]

    q = pd.DataFrame({
        "country": COUNTRIES[ctry], "arm": arm, "pool": pool, "adv": win,
        "filled": shown, "ghost": ghost, "tap": tap, "spend": spend,
        "install": install & shown, "attributed": tap & install,
        "inc": inc, "value_inc": inc * value, "value_attr": (tap & install) * value})
    country = (q.groupby(["country", "arm"])
                .agg(queries=("tap", "size"), filled=("filled", "sum"), ghost=("ghost", "sum"),
                     taps=("tap", "sum"), spend=("spend", "sum"), installs=("install", "sum"),
                     attributed=("attributed", "sum"), inc=("inc", "sum"),
                     value_inc=("value_inc", "sum"), value_attr=("value_attr", "sum"))
                .reset_index())
    country.insert(0, "day", day)

    a = (q[q.filled].groupby(["adv", "pool"])
         .agg(wins=("filled", "sum"), taps=("tap", "sum"), spend=("spend", "sum"),
              attributed=("attributed", "sum"), inc=("inc", "sum"),
              value_inc=("value_inc", "sum"), value_attr=("value_attr", "sum")))
    pools = range(remaining.shape[0])
    full = pd.MultiIndex.from_product([range(world.n_adv), pools], names=["adv", "pool"])
    a = a.reindex(full, fill_value=0).reset_index()
    a["budget"] = budget_pool[a["pool"], a["adv"]]
    a["exhaust_hour"] = exhaust_hour[a["pool"], a["adv"]]
    a.insert(0, "day", day)

    log_df = None
    if log:
        m = has
        wv = win[m]
        log_df = pd.DataFrame({
            "day": day, "country": COUNTRIES[ctry[m]], "adv": wv,
            "category": CATEGORIES[world.adv_cat[wv]], "rel": KW_REL[world.rel[kw[m], wv]],
            "returning": ret[m], "eng": eng[m], "shown": shown[m], "tap": tap[m],
            "install": install[m], "spend": spend[m], "p0": p0w[m], "p1": p1w[m]})

    obs_df = None
    if obs and obs_rows:
        o = pd.concat(obs_rows, ignore_index=True)
        qi = o.pop("q").to_numpy()
        # observed outcome: actual install if the ad won, else the organic install draw
        y_won = install[qi]
        y_lost = u_org_j[qi] < o["p0"].to_numpy()
        obs_df = o.assign(
            day=day, country=COUNTRIES[ctry[qi]], returning=ret[qi], eng=eng[qi],
            hour=hour[qi], category=CATEGORIES[world.adv_cat[o["adv"]]],
            rel=KW_REL[o["rel"]], y=np.where(o["won"], y_won, y_lost).astype(int))

    return SimResult(country, a, log_df, obs_df)


def simulate(world: World, days, policy: Policy, n_queries: int = 20_000,
             log: bool = False, obs: bool = False) -> SimResult:
    parts = [run_day(world, d, policy, n_queries, log, obs) for d in days]
    cat = lambda name: (pd.concat([getattr(p, name) for p in parts], ignore_index=True)
                        if getattr(parts[0], name) is not None else None)
    return SimResult(cat("country"), cat("adv"), cat("log"), cat("obs"))


__all__ = ["World", "Policy", "SimResult", "make_world", "simulate", "run_day", "replace",
           "COUNTRIES", "CATEGORIES", "KW_REL", "SCALE", "EPOCH", "calendar"]
