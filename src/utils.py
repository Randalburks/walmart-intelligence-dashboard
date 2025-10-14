from __future__ import annotations
import numpy as np
import pandas as pd

def safe_num(x):
    try:
        v = float(x)
    except Exception:
        return 0.0
    if not np.isfinite(v):
        return 0.0
    return v

def kpis_for_slice(df: pd.DataFrame):
    if df is None or df.empty:
        return {"avg_units": 0.0, "avg_price": 0.0, "wow_pct": 0.0}
    units = pd.to_numeric(df["sales"], errors="coerce").mean()
    price = pd.to_numeric(df["sell_price"], errors="coerce").mean()
    s = df.set_index("date")["sales"].resample("W").sum()
    wow = 0.0
    if len(s) > 1:
        prev = float(s.iloc[-2])
        curr = float(s.iloc[-1])
        wow = 100.0 * (curr - prev) / (prev + 1e-9)
    return {
        "avg_units": float(units) if np.isfinite(units) else 0.0,
        "avg_price": float(price) if np.isfinite(price) else 0.0,
        "wow_pct": float(wow),
    }

def anomaly_flags(df: pd.DataFrame, window: int = 7, z: float = 3.0) -> pd.DataFrame:
    out = df[["date", "sales"]].copy()
    s = pd.to_numeric(out["sales"], errors="coerce")
    m = s.rolling(window, min_periods=max(1, window // 2)).mean()
    sd = s.rolling(window, min_periods=max(1, window // 2)).std(ddof=0)
    thresh = z * sd.replace(0, np.nan)
    out["anomaly"] = ((s - m).abs() > thresh).fillna(False).astype(int)
    return out

def seasonal_naive(series: pd.Series, horizon: int = 28, season: int = 7) -> np.ndarray:
    s = series.dropna()
    if len(s) < 1:
        return np.zeros(horizon)
    if len(s) < season:
        last = float(s.iloc[-1])
        return np.full(horizon, last)
    pattern = s.iloc[-season:].to_numpy()
    reps = int(np.ceil(horizon / season))
    fc = np.tile(pattern, reps)[:horizon]
    return fc.astype(float)

def moving_avg(series: pd.Series, horizon: int = 28, window: int = 7) -> np.ndarray:
    s = series.dropna()
    if len(s) < 1:
        return np.zeros(horizon)
    ma = float(s.rolling(window, min_periods=1).mean().iloc[-1])
    return np.full(horizon, ma)