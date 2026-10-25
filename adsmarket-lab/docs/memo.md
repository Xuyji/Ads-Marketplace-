# Memo: Measuring what our ads actually do

**To:** Ads leadership; Product, Engineering, Sales, Finance partners
**From:** Data Insights
**Re:** Four measurement gaps and what to do about them
*(Simulated marketplace. Every figure below comes from `run_all.py` and is graded against a known ground truth.)*

---

## TL;DR

1. **Our A/B tests can over-ship pricing and ranking changes by ~2×.** Randomize by budget, not by query, wherever advertisers are budget-capped.
2. **Raise the CPT reserve to $0.25, not to the revenue peak.** It captures half the available revenue (+2.1%) at a quarter of the advertiser cost (−5% surplus vs −19%).
3. **Last-tap attribution tells brand advertisers their ads are 4.6× more effective than they are.** Move advertiser reporting toward incrementality (iCPA), backed by an always-on ghost-ads holdout.
4. **Plan next quarter on $558M P50 revenue** (P10 $529M / P90 $594M), with the caveat that intervals currently run narrow.

---

## 1 · Product & Engineering: fix the experiment unit for marketplace tests

About 45% of advertisers exhaust their daily budget. When we randomize *queries*, the treatment arm spends budget the control arm would otherwise have spent, so control looks artificially weak.

| Design | Revenue lift from $0.10 → $0.50 reserve | Truth |
|---|---|---|
| Query-randomized (current default) | **+9.5%** | +4.1% |
| Budget-split | **+4.1%** (CI +1.5% to +6.8%) | +4.1% |

**Ask:** experimentation platform support for budget-split assignment (split each advertiser's budget into isolated pools per arm). Make it the required design for any test touching pricing, ranking, pacing or reserve prices. Geo tests leak through the same mechanism: control geos lost 1% of revenue when treated geos improved.

## 2 · Product: reserve price recommendation

| Reserve | Revenue | Advertiser surplus | Fill rate | Incremental installs |
|---|---|---|---|---|
| $0.10 (today) | — | — | — | — |
| **$0.25 (recommended)** | **+2.1%** | **−5.1%** | −0.8% | −4.7% |
| $0.60 (revenue peak) | +4.0% | −19.3% | −7.3% | −19.4% |

The last 1.9 points of revenue cost 14 more points of advertiser surplus. That's long-run churn risk the short-run revenue metric can't see. Guardrails used: ≤8% surplus loss and ≤5% fill loss. Validate with a **budget-split** test before rollout (see §1).

## 3 · Sales & GTM: lead with incrementality

| Keyword type | CPA (last-tap) | True incremental CPA | Over-credit |
|---|---|---|---|
| Own brand | $0.37 | **$1.72** | **4.6×** |
| Generic | $3.99 | $4.14 | 1.04× |
| Competitor brand | $3.63 | $3.63 | 1.00× |

Brand-defense ads look like the cheapest installs on the platform, but most of those users were already searching for the app and would have installed it organically. Generic and competitor terms are where ads create installs.

- **Talking point:** "Your brand campaign protects the top slot. Your generic campaigns grow you." Steering budget toward generic terms is where incremental value grows, for advertisers and for our long-run retention.
- **Ask:** an always-on 10% ghost-ads holdout (no revenue cost on held-out auctions beyond the withheld impressions). It recovered true incrementality in every segment we tested.
- Don't estimate brand incrementality from logs. Owners almost never lose their own brand auction, so there's no valid comparison group. Use the holdout.

## 4 · Finance: Q1 revenue plan

- **90-day revenue: P50 $558M, P10 $529M, P90 $594M** (ensemble of damped Holt-Winters and a YoY gradient-boosting model).
- Backtest over 8 rolling origins: **WAPE 3.8%, bias 0.0%**. We plan on the unbiased ensemble rather than the slightly more accurate but 1.2%-low ETS model.
- **Caveat:** the 80% intervals covered 71% of outcomes out of sample, so treat P10/P90 as optimistic bounds. Fix in progress: season-aware interval calibration.

## 5 · Leadership: what changes in how we decide

| Today | Proposed |
|---|---|
| Query-level A/B for marketplace changes | Budget-split for anything touching price or budgets |
| Pre/post reads of geo launches | DiD + synthetic control, with event-study pre-trend checks and placebo inference |
| Fixed-horizon tests read daily | Always-valid mSPRT (daily peeking currently means a 26% false-positive rate) |
| Advertiser success = last-tap CPA | Advertiser success = iCPA / incremental ROAS from holdouts |
| Revenue-only launch criteria | Revenue + advertiser surplus + fill guardrails |
