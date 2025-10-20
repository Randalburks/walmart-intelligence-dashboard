from __future__ import annotations
import numpy as np
import pandas as pd

def safe_num(x, default=0.0):
    v = float(np.nanmean(x)) if hasattr(x, "__iter__") else float(x)
    if not np.isfinite(v):
        return float(default)
    return float(v)

def kpis_for_slice(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"avg_units": 0.0, "avg_price": 0.0, "wow_pct": 0.0}
    avg_units = safe_num(df["sales"])
    avg_price = safe_num(df["sell_price"])
    s = df.set_index("date")["sales"].resample("W").sum()
    wow = 0.0
    if len(s) >= 2:
        prev, curr = float(s.iloc[-2]), float(s.iloc[-1])
        wow = 100.0 * (curr - prev) / (prev + 1e-9)
    return {"avg_units": avg_units, "avg_price": avg_price, "wow_pct": wow}

def anomaly_flags(df: pd.DataFrame, window=7, z=3.0) -> pd.DataFrame:
    if df.empty or "date" not in df or "sales" not in df:
        return df.assign(anomaly=0)
    s = df.sort_values("date").copy()
    y = s["sales"].astype(float).values
    if len(y) < max(3, window + 2):
        s["anomaly"] = 0
        return s
    roll = pd.Series(y).rolling(window, center=True, min_periods=1).median().values
    resid = y - roll
    mad = np.median(np.abs(resid - np.median(resid))) + 1e-9
    zscore = np.abs(resid / (1.4826 * mad))
    s["anomaly"] = (zscore >= z).astype(int)
    return s

def seasonal_naive(series: pd.Series, horizon=28, season=7) -> np.ndarray:
    series = series.astype(float).fillna(0.0)
    if len(series) < season:
        last = float(series.iloc[-1]) if len(series) else 0.0
        return np.repeat(last, horizon)
    tail = series.iloc[-season:].to_numpy()
    reps = int(np.ceil(horizon / season))
    fc = np.tile(tail, reps)[:horizon]
    return fc

def moving_avg(series: pd.Series, horizon=28, window=7) -> np.ndarray:
    series = series.astype(float).fillna(0.0)
    if len(series) < window:
        m = float(series.mean()) if len(series) else 0.0
    else:
        m = float(series.rolling(window).mean().iloc[-1])
        if not np.isfinite(m):
            m = float(series.mean())
    return np.repeat(m, horizon)