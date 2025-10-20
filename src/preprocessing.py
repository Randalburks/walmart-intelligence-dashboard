from __future__ import annotations
import os, hashlib, io
from typing import Optional, Tuple
import pandas as pd
import numpy as np
import requests

# ---------- optional release fetch helpers ----------
def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def _ensure_parquet_local(local_path: str = "data/_merged_m5_zstd.parquet") -> Optional[str]:
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    if os.path.exists(local_path):
        return local_path
    url = os.getenv("RELEASE_URL", "").strip()
    if not url:
        return None
    r = requests.get(url, stream=True, timeout=180)
    r.raise_for_status()
    tmp = local_path + ".part"
    with open(tmp, "wb") as f:
        for chunk in r.iter_content(1024 * 1024):
            if chunk:
                f.write(chunk)
    want = os.getenv("RELEASE_SHA256", "").strip().lower()
    if want:
        got = _sha256(tmp)
        if got != want:
            os.remove(tmp)
            raise RuntimeError(f"SHA256 mismatch for downloaded parquet. got={got} expected={want}")
    os.replace(tmp, local_path)
    return local_path

# ---------- public API ----------
def load_merge(use_cache: bool = True) -> pd.DataFrame:
    """
    Preferred path:
      1) data/_merged_m5_zstd.parquet (or fetched via RELEASE_URL/RELEASE_SHA256)
      2) If parquet unavailable, build from M5 CSVs in /data.
    Returns a tidy daily panel with columns: date, store_id, item_id, sales, sell_price,
    dept_id, cat_id, state_id, event_name_1, event_type_1, snap.
    """
    # Try parquet (local or fetched)
    parquet_path = _ensure_parquet_local("data/_merged_m5_zstd.parquet")
    if parquet_path and os.path.exists(parquet_path):
        return pd.read_parquet(parquet_path)

    # Otherwise, build from CSVs if present
    cal_p = "data/calendar.csv"
    price_p = "data/sell_prices.csv"
    sales_p = "data/sales_train_validation.csv"
    for p in (cal_p, price_p, sales_p):
        if not os.path.exists(p):
            raise FileNotFoundError(
                "Could not find a parquet cache or the raw CSVs.\n"
                "Provide data/_merged_m5_zstd.parquet (or set RELEASE_URL/RELEASE_SHA256), "
                "or place M5 CSVs into /data: calendar.csv, sell_prices.csv, sales_train_validation.csv."
            )

    calendar = pd.read_csv(cal_p)
    prices = pd.read_csv(price_p)
    sales = pd.read_csv(sales_p)

    # Melt wide daily sales (d_1 ... d_xxx) to long
    id_cols = [c for c in sales.columns if not c.startswith("d_")]
    day_cols = [c for c in sales.columns if c.startswith("d_")]
    sales_long = sales.melt(id_vars=id_cols, value_vars=day_cols,
                            var_name="d", value_name="sales")
    # Join calendar to get dates and events
    cal_small = calendar[["d", "date", "event_name_1", "event_type_1", "snap_CA", "snap_TX", "snap_WI"]].copy()
    sales_long = sales_long.merge(cal_small, on="d", how="left")
    sales_long["date"] = pd.to_datetime(sales_long["date"])

    # Normalize identifiers
    sales_long = sales_long.rename(columns={
        "item_id": "item_id",
        "store_id": "store_id",
        "dept_id": "dept_id",
        "cat_id": "cat_id",
        "state_id": "state_id",
        "sales": "sales",
    })

    # Build a single SNAP flag (true if any state SNAP is 1 for that store's state)
    state_to_snapcol = {"CA": "snap_CA", "TX": "snap_TX", "WI": "snap_WI"}
    def _snap_row(row):
        s = str(row.get("state_id", ""))
        col = state_to_snapcol.get(s, None)
        return int(row.get(col, 0)) if col else 0
    sales_long["snap"] = sales_long.apply(_snap_row, axis=1).astype("int8")

    # Attach weekly prices; calendar links prices via 'wm_yr_wk'
    cal_price = calendar[["d", "wm_yr_wk"]].copy()
    prices_full = prices.merge(cal_price, on="wm_yr_wk", how="left")
    prices_full = prices_full.merge(calendar[["d", "date"]], on="d", how="left")
    prices_full["date"] = pd.to_datetime(prices_full["date"])
    prices_full = prices_full[["store_id", "item_id", "date", "sell_price"]]

    df = sales_long.merge(
        prices_full, on=["store_id", "item_id", "date"], how="left"
    )

    # Carry-fill prices within each store-item so daily panel is aligned
    df.sort_values(["store_id", "item_id", "date"], inplace=True)
    df["sell_price"] = (
        df.groupby(["store_id", "item_id"])["sell_price"]
          .apply(lambda s: s.ffill().bfill())
          .values
    )

    # Types & ordering
    keep_cols = [
        "date","store_id","item_id","dept_id","cat_id","state_id",
        "sales","sell_price","event_name_1","event_type_1","snap"
    ]
    df = df[keep_cols].copy()
    df["sales"] = df["sales"].astype("float32")
    df["sell_price"] = df["sell_price"].astype("float32")

    # Save a compressed parquet cache locally if allowed
    if use_cache:
        outp = "data/_merged_m5_zstd.parquet"
        os.makedirs("data", exist_ok=True)
        try:
            df.to_parquet(outp, compression="zstd", index=False)
        except Exception:
            # Some hosting environments don’t allow writes; ignore
            pass

    return df

def sample_panel(df: pd.DataFrame, n_stores: int = 3, n_items: int = 30, seed: int = 42) -> pd.DataFrame:
    """
    Fast sub-sampler for interactivity: pick n_stores and n_items per store.
    Guarantees at least one product per chosen store.
    """
    rng = np.random.default_rng(seed)
    stores = sorted(pd.unique(df["store_id"]))
    if not stores:
        return df.iloc[0:0]

    chosen_stores = (
        list(stores) if n_stores >= len(stores)
        else list(rng.choice(stores, size=n_stores, replace=False))
    )

    out = []
    for s in chosen_stores:
        g = df[df["store_id"] == s]
        items = sorted(pd.unique(g["item_id"]))
        if not items:
            continue
        pick = items if n_items >= len(items) else list(rng.choice(items, size=n_items, replace=False))
        out.append(g[g["item_id"].isin(pick)])
    if not out:
        return df.iloc[0:0]
    return pd.concat(out, axis=0).sort_values(["store_id","item_id","date"]).reset_index(drop=True)