"""Revenue and supply (search volume) forecasting with rolling-origin backtests.

Models
  seasonal_naive  last observed week, repeated
  ets             damped-trend Holt-Winters on log(y), weekly seasonality
  yoy_gbm         gradient boosting on the log year-over-year ratio with calendar + holiday
                  features, anchored on last year's actuals (handles yearly seasonality and
                  holidays that a 7-day seasonal model cannot)
  ensemble        mean of ets and yoy_gbm in log space

Intervals are split-conformal: quantiles of backtest log-errors by horizon bucket, so their
coverage is checked out of sample rather than assumed.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from .sim import holiday_mult


def calendar_features(dates: pd.DatetimeIndex) -> pd.DataFrame:
    doy = dates.dayofyear.to_numpy()
    return pd.DataFrame({
        "dow": dates.dayofweek, "doy_sin": np.sin(2 * np.pi * doy / 365.25),
        "doy_cos": np.cos(2 * np.pi * doy / 365.25), "month": dates.month,
        "holiday": [holiday_mult(d) for d in dates],
        "days_to_xmas": [(pd.Timestamp(d.year, 12, 25) - d).days for d in dates]}, index=dates)


def seasonal_naive(y: pd.Series, h: int) -> np.ndarray:
    return np.tile(y.to_numpy()[-7:], h // 7 + 1)[:h]


def ets(y: pd.Series, h: int) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = ExponentialSmoothing(np.log(y.to_numpy()), trend="add", damped_trend=True,
                                 seasonal="add", seasonal_periods=7).fit()
    return np.exp(m.forecast(h))


def yoy_gbm(y: pd.Series, h: int) -> np.ndarray:
    ly = np.log(y)
    target = (ly - ly.shift(364)).dropna()          # 52 weeks back keeps day-of-week aligned
    X = calendar_features(target.index)
    X["recent_yoy"] = target.rolling(28).mean().shift(1).bfill()
    m = HistGradientBoostingRegressor(max_iter=300, learning_rate=.05, max_depth=4,
                                      random_state=0).fit(X, target)
    fut = pd.date_range(y.index[-1] + pd.Timedelta(days=1), periods=h)
    Xf = calendar_features(fut)
    Xf["recent_yoy"] = target.iloc[-28:].mean()
    anchor = ly.reindex(fut - pd.Timedelta(days=364)).to_numpy()
    return np.exp(anchor + m.predict(Xf))


MODELS = {"seasonal_naive": seasonal_naive, "ets": ets, "yoy_gbm": yoy_gbm}


def forecast_all(y: pd.Series, h: int) -> dict[str, np.ndarray]:
    f = {name: fn(y, h) for name, fn in MODELS.items()}
    f["ensemble"] = np.exp((np.log(f["ets"]) + np.log(f["yoy_gbm"])) / 2)
    return f


def backtest(y: pd.Series, h: int = 28, n_origins: int = 8, step: int = 28) -> pd.DataFrame:
    rows = []
    last = len(y) - h
    for i in range(n_origins):
        cut = last - i * step
        train, test = y.iloc[:cut], y.iloc[cut:cut + h]
        for name, pred in forecast_all(train, h).items():
            rows.append(pd.DataFrame({"origin": y.index[cut], "model": name, "date": test.index,
                                      "horizon": np.arange(1, h + 1), "actual": test.to_numpy(),
                                      "pred": pred}))
    return pd.concat(rows, ignore_index=True)


def score(bt: pd.DataFrame) -> pd.DataFrame:
    g = bt.assign(err=bt.pred - bt.actual, ape=(bt.pred - bt.actual).abs() / bt.actual)
    return (g.groupby("model")
             .apply(lambda d: pd.Series({"WAPE": d.err.abs().sum() / d.actual.sum(),
                                         "MAPE": d.ape.mean(),
                                         "bias": d.err.sum() / d.actual.sum()}),
                    include_groups=False)
             .sort_values("WAPE"))


def conformal_quantiles(bt: pd.DataFrame, model: str, levels=(.1, .5, .9)) -> pd.DataFrame:
    d = bt[bt.model == model].assign(le=lambda x: np.log(x.actual / x.pred),
                                     bucket=lambda x: (x.horizon - 1) // 7)
    return d.groupby("bucket").le.quantile(list(levels)).unstack()


def interval_coverage(bt: pd.DataFrame, model: str, min_calib: int = 3) -> float:
    """Out-of-sample coverage of nominal 80% intervals: each origin is calibrated only on
    strictly older origins (no look-ahead), then coverage is averaged over origins."""
    origins = sorted(bt.origin.unique())
    hits = []
    for o in origins[min_calib:]:
        q = conformal_quantiles(bt[bt.origin < o], model)
        t = bt[(bt.model == model) & (bt.origin == o)]
        b = (t.horizon - 1) // 7
        lo = t.pred * np.exp(q.loc[b, .1].to_numpy())
        hi = t.pred * np.exp(q.loc[b, .9].to_numpy())
        hits.append(((t.actual >= lo) & (t.actual <= hi)).to_numpy())
    return float(np.concatenate(hits).mean())


def plan_forecast(y: pd.Series, bt: pd.DataFrame, model: str, h: int) -> pd.DataFrame:
    """Forward forecast with P10/P50/P90 from conformal backtest errors."""
    pred = forecast_all(y, h)[model]
    q = conformal_quantiles(bt, model)
    b = np.minimum(np.arange(h) // 7, q.index.max())
    dates = pd.date_range(y.index[-1] + pd.Timedelta(days=1), periods=h)
    return pd.DataFrame({"date": dates, "p10": pred * np.exp(q.loc[b, .1].to_numpy()),
                         "p50": pred * np.exp(q.loc[b, .5].to_numpy()),
                         "p90": pred * np.exp(q.loc[b, .9].to_numpy())})
