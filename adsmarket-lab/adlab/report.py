"""Figures and a self-contained HTML decision report."""
from __future__ import annotations

import base64
import html
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# Reference palette (validated categorical order; truth is always neutral ink, never a series hue)
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": .8, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 12,
    "axes.titleweight": "semibold", "axes.titlecolor": INK, "axes.titlelocation": "left",
    "font.size": 10, "legend.frameon": False, "lines.linewidth": 2, "figure.dpi": 130,
    "text.parse_math": False})

pct = lambda x: f"{x:+.1%}"


def _truth_mark(ax, y, x, label=False):
    ax.scatter([x], [y], marker="D", s=60, color=INK, zorder=5, label="Truth" if label else None)


def fig_interference(R, path):
    t = R["reserve_test"]["table"]
    names = {"spend": "Platform revenue", "inc": "Incremental installs",
             "value_inc": "Advertiser incremental value", "adv_surplus": "Advertiser surplus"}
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    for i, m in enumerate(names):
        for off, design, col in [(.15, "query-randomized", S2), (-.15, "budget-randomized", S1)]:
            r = t[(t.metric == m) & (t.design == design)].iloc[0]
            ax.errorbar(r.estimate, i + off, xerr=[[r.estimate - r.ci_lo], [r.ci_hi - r.estimate]],
                        fmt="o", ms=8, color=col, elinewidth=2, capsize=0, mec=SURFACE, mew=2,
                        label=design.replace("-", " ").capitalize() if i == 0 else None)
        _truth_mark(ax, i, t[t.metric == m].truth.iloc[0], label=i == 0)
    ax.set_yticks(range(len(names)), names.values())
    ax.invert_yaxis()
    ax.axvline(0, color=MUTED, lw=1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0))
    ax.set_title(f"CPT reserve ${R['reserve_test']['r0']:.2f} → ${R['reserve_test']['r1']:.2f}: lift by experiment design")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_frontier(R, path):
    f = R["frontier"]
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    for col, c, lab in [("revenue_vs_base", S1, "Platform revenue"),
                        ("adv_surplus_vs_base", S2, "Advertiser surplus"),
                        ("fill_rate_vs_base", S3, "Fill rate")]:
        ax.plot(f.reserve, f[col], color=c, marker="o", ms=5, mec=SURFACE, label=lab)
    rec = R["reco"]
    ax.axvline(rec.reserve, color=INK, lw=1, ls="--")
    ax.annotate(f"Recommended ${rec.reserve:.2f}\nrevenue {pct(rec.revenue_vs_base)}",
                (rec.reserve, -.19), xytext=(6, 0), textcoords="offset points",
                fontsize=9, color=INK, va="top")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0))
    ax.set_xlabel("CPT reserve price ($)")
    ax.set_ylabel("Change vs $0.10 reserve")
    ax.set_title("Reserve frontier: small revenue gains, large advertiser costs")
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_causal(R, path):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
    for ax, key, title in [(axes[0], "causal_full", "All eligible auctions"),
                           (axes[1], "causal_trim", "Non-brand, overlap (0.05 ≤ e ≤ 0.95)")]:
        tb = R[key]["table"]
        for i, r in tb.iterrows():
            err = None if np.isnan(r.se) else [[1.96 * r.se], [1.96 * r.se]]
            ax.errorbar(r.estimate, i, xerr=err, fmt="o", ms=8, color=S1, elinewidth=2,
                        mec=SURFACE, mew=2)
        ax.axvline(tb.truth.iloc[0], color=INK, ls="--", lw=1.2)
        ax.text(tb.truth.iloc[0], len(tb) - .75, " truth", color=INK, fontsize=9)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("ATT: Δ install probability from winning")
        ax.set_yticks(range(len(tb)), tb.method)
    axes[0].invert_yaxis()
    fig.suptitle("Observational incrementality: naive comparison vs doubly robust", x=.01,
                 ha="left", fontweight="semibold", color=INK)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_attribution(R, path):
    t = R["attr_rel"].set_index("rel").loc[["own_brand", "competitor_brand", "generic", "off_category"]]
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    x = np.arange(len(t))
    w = .26
    series = [("view_through", "View-through", S3), ("last_tap", "Last-tap", S2),
              ("ghost_ads_inc", "Ghost-ads experiment", S1)]
    for j, (col, lab, c) in enumerate(series):
        ratio = t[col] / t.true_inc
        ax.bar(x + (j - 1) * w, ratio, w - .03, color=c, label=lab)
        if col == "last_tap":
            ax.text(x[0], ratio.iloc[0] + .1, f"{ratio.iloc[0]:.1f}×", ha="center", fontsize=9, color=INK)
    ax.axhline(1, color=INK, lw=1.2, ls="--", label="Truth (1.0×)")
    ax.set_xticks(x, [s.replace("_", " ") for s in t.index])
    ax.set_ylabel("Reported ÷ true incremental installs")
    ax.set_title("Attribution inflation by keyword type")
    ax.legend(fontsize=9, loc="upper right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_geo(R, path):
    g = R["geo"]
    s = g["sc"]["series"]
    p = g["panel"]
    dates = p.drop_duplicates("day").set_index("day").date
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), gridspec_kw={"width_ratios": [1.4, 1]})
    ax = axes[0]
    ax.plot(dates.loc[s.day], np.exp(s.treated) / 1e3, color=S1, lw=1.6, label="Treated geos (actual)")
    ax.plot(dates.loc[s.day], np.exp(s.synthetic) / 1e3, color=S2, lw=1.6, label="Synthetic control")
    ax.axvline(dates.loc[g["launch"]], color=INK, lw=1, ls="--")
    ax.set_title("GB/DE/FR/IT revenue vs synthetic control")
    ax.set_ylabel("Daily revenue (sim units, k)")
    ax.legend(fontsize=9, loc="upper left")
    ax.tick_params(axis="x", rotation=30)
    ax = axes[1]
    es = g["event_study"]
    pre = es.week < 0
    for mask, c in [(pre, MUTED), (~pre, S1)]:
        e = es[mask]
        ax.errorbar(e.week, e.coef, yerr=[e.coef - e.ci_lo, e.ci_hi - e.coef], fmt="o", ms=6,
                    color=c, elinewidth=1.6, mec=SURFACE)
    ax.axhline(0, color=MUTED, lw=1)
    ax.axvline(-.5, color=INK, lw=1, ls="--")
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(2))
    ax.set_title("Event study (weeks from launch)")
    ax.set_ylabel("Δ log revenue vs week −1")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_forecast(R, path):
    y = R["rev_series"]
    plan = R["plan"]
    fig, ax = plt.subplots(figsize=(9, 3.4))
    hist = y.iloc[-240:]
    ax.plot(hist.index, hist / 1e6, color=S1, lw=1.4, label="Actual")
    ax.fill_between(plan.date, plan.p10 / 1e6, plan.p90 / 1e6, color=S2, alpha=.22, lw=0,
                    label="P10–P90")
    ax.plot(plan.date, plan.p50 / 1e6, color=S2, lw=1.8, label="P50 forecast")
    ax.set_ylabel("Daily revenue ($M)")
    ax.set_title("Revenue plan: next 90 days with conformal intervals")
    ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_experiment_stats(R, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
    pc = R["cuped"]["power_curve"]
    ax = axes[0]
    for m, c in [("raw", S2), ("CUPED", S1)]:
        d = pc[pc.method == m]
        ax.plot(d.n_adv, d.power, color=c, marker="o", ms=5, mec=SURFACE, label=m)
    ax.axhline(.8, color=MUTED, lw=1, ls="--")
    ax.set_xscale("log")
    ax.set_xticks(pc.n_adv.unique(), [f"{n:,}" for n in pc.n_adv.unique()])
    ax.set_xlabel("Advertisers randomized")
    ax.set_ylabel("Power")
    ax.set_title(f"Power for a {R['cuped']['effect']:.0%} spend lift")
    ax.legend(fontsize=9, loc="upper left")
    ax = axes[1]
    bd = R["peek_null"]["by_day"]
    ax.plot(bd.day, bd.naive_peeking, color=S2, label="Daily peeking, fixed-horizon z-test")
    ax.plot(bd.day, bd.msprt, color=S1, label="Always-valid mSPRT")
    ax.axhline(.05, color=INK, lw=1, ls="--")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0))
    ax.set_xlabel("Day of test")
    ax.set_ylabel("False positive rate (A/A)")
    ax.set_title("Peeking inflates false positives")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


FIGURES = {"interference": fig_interference, "frontier": fig_frontier, "causal": fig_causal,
           "attribution": fig_attribution, "geo": fig_geo, "forecast": fig_forecast,
           "experiment_stats": fig_experiment_stats}


def make_figures(R, outdir: Path) -> dict[str, Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, fn in FIGURES.items():
        p = outdir / f"{name}.png"
        fn(R, p)
        paths[name] = p
    return paths


# --------------------------------------------------------------------------------------
# HTML
# --------------------------------------------------------------------------------------
CSS = """
:root{--bg:#f4f3ef;--card:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--line:#e6e5e0;--accent:#2a78d6;
--good:#0a7a3e;--bad:#b3261e;--callout:#eef4fc}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#121211;--card:#1a1a19;
--ink:#f0efec;--ink2:#c3c2b7;--line:#2e2d2a;--accent:#3987e5;--good:#4cc38a;--bad:#ff8a80;--callout:#1d2633}}
:root[data-theme="dark"]{--bg:#121211;--card:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--line:#2e2d2a;
--accent:#3987e5;--good:#4cc38a;--bad:#ff8a80;--callout:#1d2633}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Inter,sans-serif}
main{max-width:1040px;margin:0 auto;padding:32px 16px 64px}
h1{font-size:28px;margin:0 0 4px}h2{font-size:20px;margin:40px 0 8px}
.sub{color:var(--ink2);margin:0 0 24px}.card{background:var(--card);border:1px solid var(--line);
border-radius:12px;padding:20px;margin:16px 0}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px}
.kpi b{display:block;font-size:26px;font-variant-numeric:tabular-nums}.kpi span{color:var(--ink2);font-size:13px}
table{width:100%;border-collapse:collapse;font-size:13.5px;font-variant-numeric:tabular-nums}
th,td{padding:7px 8px;border-bottom:1px solid var(--line);text-align:right}
th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){text-align:left}th{color:var(--ink2);font-weight:600}
.tw{overflow-x:auto}.ok{color:var(--good);font-weight:600}.no{color:var(--bad);font-weight:600}
.decision{background:var(--callout);border-left:3px solid var(--accent);padding:12px 16px;border-radius:8px;margin:12px 0}
img{width:100%;height:auto;border-radius:8px;background:#fcfcfb}code{font-size:13px}
"""


def _img(p: Path) -> str:
    return f'<img alt="{p.stem}" src="data:image/png;base64,{base64.b64encode(p.read_bytes()).decode()}">'


def _table(df: pd.DataFrame, fmt: dict | None = None) -> str:
    fmt = fmt or {}
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns)
    body = []
    for _, r in df.iterrows():
        cells = []
        for c in df.columns:
            v = r[c]
            if c in fmt:
                v = fmt[c](v)
            elif isinstance(v, (float, np.floating)):
                v = f"{v:,.3f}"
            cells.append(f"<td>{v}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f'<div class="tw"><table><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


def verdict(ok: bool) -> str:
    return '<span class="ok">✓ recovers truth</span>' if ok else '<span class="no">✗ misses truth</span>'


def write_html(R, figs: dict[str, Path], path: Path):
    sc = R["scorecard"].copy()
    sc["verdict"] = sc.recovers.map(verdict)
    sc = sc.drop(columns="recovers")
    pfmt = {"estimate": pct, "truth": pct, "error": lambda v: f"{v:+.1f} pp"}
    rt = R["reserve_test"]
    rq = rt["table"].query("design=='query-randomized' and metric=='spend'").iloc[0]
    rb = rt["table"].query("design=='budget-randomized' and metric=='spend'").iloc[0]
    rec = R["reco"]
    peak = R["frontier"].loc[R["frontier"].revenue.idxmax()]
    attr = R["attr_rel"].set_index("rel")
    fc = R["forecast_scores"]
    plan_q = R["plan"]

    kpis = f"""<div class="kpis">
<div class="kpi"><b>{pct(rq.estimate)} vs {pct(rq.truth)}</b><span>Query-randomized A/B overstates reserve-price revenue lift {rq.estimate / rq.truth:.1f}×</span></div>
<div class="kpi"><b>{attr.loc['own_brand','inflation_last_tap']:.1f}×</b><span>Last-tap over-credit on own-brand keywords</span></div>
<div class="kpi"><b>{R['cuped']['var_reduction']:.0%}</b><span>Variance removed by CUPED (advertiser tests)</span></div>
<div class="kpi"><b>{fc.loc[R['best_model'],'WAPE']:.1%}</b><span>28-day revenue forecast WAPE, {R['coverage']:.0%} coverage of 80% intervals</span></div>
</div>"""

    sections = f"""
<h2>1 · Marketplace experiments: interference through shared budgets</h2>
<div class="card">{_img(figs['interference'])}
<p>About half of advertisers exhaust their daily budget. When <em>queries</em> are randomized, the treatment arm's higher prices drain budgets the control arm would otherwise have spent, so control looks worse than a real all-control world. The query-split test reads <b>{pct(rq.estimate)}</b> revenue; the true global effect is <b>{pct(rq.truth)}</b>. A <b>budget-split</b> design (each advertiser's budget halved across two isolated mini-markets) reads <b>{pct(rb.estimate)}</b> (95% CI {pct(rb.ci_lo)} to {pct(rb.ci_hi)}).</p>
<div class="decision"><b>Decision:</b> any test that touches pricing, ranking or pacing in budget-constrained segments must run as budget-split, or the experimentation platform will systematically over-ship.</div></div>

<h2>2 · Reserve-price optimization with marketplace-health guardrails</h2>
<div class="card">{_img(figs['frontier'])}
<p>Revenue-maximizing is not the same as marketplace-optimal. Under guardrails of ≤8% advertiser-surplus loss and ≤5% fill-rate loss, the recommended reserve is <b>${rec.reserve:.2f}</b>: revenue {pct(rec.revenue_vs_base)}, advertiser surplus {pct(rec.adv_surplus_vs_base)}, incremental installs {pct(rec.inc_installs_vs_base)}. Going all the way to the revenue peak (${peak.reserve:.2f}) adds only {(peak.revenue_vs_base - rec.revenue_vs_base) * 100:.1f} more points of revenue and costs {(rec.adv_surplus_vs_base - peak.adv_surplus_vs_base) * 100:.1f} more points of advertiser surplus. That surplus loss is a churn risk the short-run revenue number doesn't show.</p></div>

<h2>3 · Incrementality from observational auction logs</h2>
<div class="card">{_img(figs['causal'])}
<p>Comparing won vs lost auctions overstates the effect of winning <b>{R['causal_full']['table'].rel_error.iloc[0] + 1:.1f}×</b>, because advertisers bid up returning, high-intent users who install anyway. Even full-sample AIPW misses. Overlap diagnostics show why: owners almost never lose auctions on their own brand keyword, so positivity fails and nearly all the confounding sits in brand auctions. The observational estimand is therefore restricted to non-brand auctions with common support. There, cross-fitted AIPW estimates {R['causal_trim']['table'].estimate.iloc[3]:.4f} against a truth of {R['causal_trim']['truth']:.4f}, and its 95% CI covers the truth (it also covered in 5 of 5 independently simulated worlds). Brand incrementality is handed to the ghost-ads experiment in section 4, the design that can actually identify it.</p>
{_table(R['causal_seg'][['rel','n','naive','aipw','ci_lo','ci_hi','truth']].round(4))}</div>

<h2>4 · Attribution vs ghost-ads incrementality</h2>
<div class="card">{_img(figs['attribution'])}
<p>A 10% ghost-ads holdout recovers true incremental installs in every keyword segment (CIs cover truth in {int(R['attr_rel'].ghost_covers.sum())}/{len(R['attr_rel'])}). Last-tap CPA on own-brand terms is <b>${attr.loc['own_brand','cpa_last_tap']:.2f}</b>, but the incremental CPA is <b>${attr.loc['own_brand','icpa_true']:.2f}</b>.</p>
{_table(R['attr_rel'][['rel','spend','last_tap','ghost_ads_inc','true_inc','cpa_last_tap','icpa_ghost','icpa_true','inflation_last_tap']].round(2))}
<div class="decision"><b>For Sales/GTM:</b> lead advertiser conversations with iCPA and incremental ROAS. Brand-defense spend looks cheapest under last-tap and is the most over-credited. Generic and competitor terms are where ads actually create installs.</div></div>

<h2>5 · Quasi-experimental read of a non-randomized geo launch</h2>
<div class="card">{_img(figs['geo'])}
{_table(R['geo']['table'][['method','estimate','truth','rel_error']], {'estimate': pct, 'truth': pct, 'rel_error': lambda v: f'{v:+.0%}'})}
<p>Pre/post doubles the effect because the launch landed on the holiday ramp. Shared budgets also leak across borders: control geos lost <b>{pct(R['geo']['spillover']['donors'])}</b> revenue, which inflates DiD slightly. One world is a single draw, so DiD and synthetic control were also run on {R['geo_mc'].seed.nunique()} independently simulated marketplaces:</p>
{_table(R['geo_mc_summary'], {'bias': lambda v: f'{v * 100:+.1f} pp', 'rmse': lambda v: f'{v * 100:.1f} pp', 'ci_coverage': lambda v: '—' if pd.isna(v) else f'{v:.0%}'})}
</div>

<h2>6 · Revenue & supply forecasting for planning</h2>
<div class="card">{_img(figs['forecast'])}
<p>Rolling-origin backtest (8 origins × 28-day horizon) on two years of daily data. Intervals are split-conformal from backtest errors, and each origin is calibrated only on earlier origins. Next-90-day revenue plan: <b>${plan_q.p50.sum() / 1e6:,.0f}M</b> P50 (P10–P90 band summed daily: ${plan_q.p10.sum() / 1e6:,.0f}M – ${plan_q.p90.sum() / 1e6:,.0f}M).</p>
{_table(R['forecast_table'])}</div>

<h2>7 · Experiment statistics: CUPED, power, peeking</h2>
<div class="card">{_img(figs['experiment_stats'])}
{_table(R['cuped']['summary'].round(4))}
<p>A/A false-positive rates hold at 5% with and without CUPED ({R['cuped']['n_sims']} simulations). Checking a fixed-horizon test every day for 28 days produces a false positive in <b>{R['peek_null']['naive_peeking']:.0%}</b> of A/A tests; the always-valid mSPRT keeps it at <b>{R['peek_null']['msprt']:.1%}</b> and still has {R['peek_alt']['msprt']:.0%} power at the alternative.</p></div>

<h2>8 · Governed KPI layer (SQL)</h2>
<div class="card"><p>Every number above is computed from <code>sql/metrics.sql</code> views over a SQLite warehouse (<code>outputs/warehouse.sqlite</code>). Category scorecard over the 2-year history:</p>
{_table(R['category_scorecard'].round(3), {'revenue': lambda v: f'${v:,.0f}'})}</div>
"""
    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Ads Marketplace Lab</title>
<style>{CSS}</style></head><body><main>
<h1>Ads Marketplace Lab</h1>
<p class="sub">A ground-truth simulator of a search-ads marketplace in which every causal, experimental and forecasting method is graded against the known answer. All data is simulated.</p>
{kpis}
<h2>Estimator scorecard</h2>
<div class="card">{_table(sc, pfmt)}</div>
{sections}
</main></body></html>"""
    path.write_text(doc, encoding="utf-8")
