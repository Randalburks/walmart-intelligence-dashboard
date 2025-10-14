# src/utils.py
from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.express as px

# ---------- numeric safety ----------
def safe_num(x) -> float:
    """Convert to float; return 0.0 on NaN/None/invalid."""
    try:
        v = float(x)
        return 0.0 if np.isnan(v) else v
    except Exception:
        return 0.0

# ---------- KPIs for a (store,item,date-range) slice ----------
def kpis_for_slice(df: pd.DataFrame) -> dict:
    """Return avg units, avg price, and WoW% for a daily sales slice."""
    if df is None or len(df) == 0:
        return {"avg_units": 0.0, "avg_price": 0.0, "wow_pct": 0.0}

    avg_units = safe_num(df["sales"].mean()) if "sales" in df else 0.0
    avg_price = safe_num(df["sell_price"].mean()) if "sell_price" in df else 0.0

    wow_pct = 0.0
    if "date" in df.columns and "sales" in df.columns:
        s = df.set_index("date")["sales"].resample("W").sum()
        if len(s) >= 2:
            prev, curr = float(s.iloc[-2]), float(s.iloc[-1])
            wow_pct = 100.0 * (curr - prev) / (prev + 1e-9)

    return {"avg_units": avg_units, "avg_price": avg_price, "wow_pct": wow_pct}

# ---------- anomaly flagging on daily sales ----------
def anomaly_flags(df: pd.DataFrame, window: int = 7, z: float = 3.0) -> pd.DataFrame:
    """
    Flag days whose sales deviate > z * rolling-std from a rolling mean.
    Adds an 'anomaly' (0/1) column; returns a copy.
    """
    out = df.copy()
    if "date" not in out.columns or "sales" not in out.columns or len(out) == 0:
        out["anomaly"] = 0
        return out

    s = out.set_index("date")["sales"].asfreq("D")
    mu = s.rolling(window, min_periods=1).mean()
    sd = s.rolling(window, min_periods=1).std().fillna(0.0)
    flag = ((s - mu).abs() > z * (sd + 1e-9)).astype(int)

    out = out.set_index("date")
    out["anomaly"] = flag.reindex(out.index).fillna(0).astype(int).values
    return out.reset_index()

# ---------- ultra-portable baseline forecasts ----------
def seasonal_naive(series: pd.Series, horizon: int = 28, season: int = 7) -> np.ndarray:
    """Repeat the last 'season' days to produce the next 'horizon' days."""
    y = series.dropna().values
    if len(y) == 0:
        return np.zeros(horizon)
    tail = y[-season:] if len(y) >= season else np.resize(y, season)
    reps = int(np.ceil(horizon / season))
    return np.tile(tail, reps)[:horizon]

def moving_avg(series: pd.Series, horizon: int = 28, window: int = 7) -> np.ndarray:
    """Forecast as the last rolling-mean level (flat line)."""
    s = series.rolling(window, min_periods=1).mean()
    last = float(s.iloc[-1]) if len(s) else 0.0
    return np.full(horizon, last)

# ---------- IDA helpers ----------
def info_table(df_: pd.DataFrame) -> pd.DataFrame:
    """.info()-like table with dtypes and null counts."""
    if df_ is None or len(df_) == 0:
        return pd.DataFrame(columns=["column", "dtype", "non_null", "nulls", "null_pct"])
    non_null = df_.notnull().sum()
    out = pd.DataFrame({
        "column": df_.columns,
        "dtype": [str(t) for t in df_.dtypes],
        "non_null": [int(non_null[c]) for c in df_.columns],
        "nulls": [int(len(df_) - non_null[c]) for c in df_.columns],
    })
    out["null_pct"] = (out["nulls"] / max(len(df_), 1)).round(4)
    return out

def missingness_bar(df_: pd.DataFrame):
    """Bar chart of null fraction per column."""
    if df_ is None or len(df_) == 0:
        return px.bar(title="Missingness by Column")
    frac = df_.isnull().mean().sort_values(ascending=False).reset_index()
    frac.columns = ["column", "null_fraction"]
    fig = px.bar(frac, x="column", y="null_fraction", title="Missingness by Column")
    fig.update_layout(xaxis_tickangle=-45, yaxis_tickformat=".0%")
    return fig

def missingness_heatmap(df_: pd.DataFrame, sample_rows: int = 1200):
    """Heatmap (columns × sampled rows) of nulls (1=null)."""
    if df_ is None or len(df_) == 0:
        return px.imshow(np.zeros((1, 1)), title="Missingness Heatmap")
    n = min(sample_rows, len(df_))
    mat = df_.sample(n, random_state=42).isnull().astype(int)
    fig = px.imshow(
        mat.T,
        color_continuous_scale="Blues",
        labels=dict(x="Sampled Row", y="Column", color="Is Null"),
        title="Missingness Heatmap (sampled rows)",
    )
    return fig