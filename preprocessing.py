from __future__ import annotations
import os
import pandas as pd

DATA_DIR = "data"
REQ = ["calendar.csv","sell_prices.csv","sales_train_validation.csv"]
CACHE = os.path.join(DATA_DIR, "_merged_m5.parquet")

def _verify_files():
    missing = [f for f in REQ if not os.path.exists(os.path.join(DATA_DIR, f))]
    if missing:
        raise FileNotFoundError(f"Missing file(s) in ./data: {', '.join(missing)}")

def load_merge(use_cache: bool = True) -> pd.DataFrame:
    _verify_files()
    if use_cache and os.path.exists(CACHE):
        return pd.read_parquet(CACHE)

    cal   = pd.read_csv(os.path.join(DATA_DIR,"calendar.csv"))
    sales = pd.read_csv(os.path.join(DATA_DIR,"sales_train_validation.csv"))
    price = pd.read_csv(os.path.join(DATA_DIR,"sell_prices.csv"))

    day_cols = [c for c in sales.columns if c.startswith("d_")]
    id_cols  = [c for c in sales.columns if c not in day_cols]
    sales_long = sales.melt(id_vars=id_cols, value_vars=day_cols,
                            var_name="d", value_name="sales")

    cal_keep = ["d","date","wm_yr_wk","event_name_1","event_type_1",
                "event_name_2","event_type_2","snap_CA","snap_TX","snap_WI"]
    df = sales_long.merge(cal[cal_keep], on="d", how="left")
    df = df.merge(price, on=["store_id","item_id","wm_yr_wk"], how="left")

    df["date"] = pd.to_datetime(df["date"])
    df.sort_values(["store_id","item_id","date"], inplace=True)

    df["sell_price"] = df.groupby(["store_id","item_id"])["sell_price"].ffill().bfill()

    def snap_flag(row):
        if row["state_id"] == "CA": return row["snap_CA"]
        if row["state_id"] == "TX": return row["snap_TX"]
        if row["state_id"] == "WI": return row["snap_WI"]
        return 0
    df["snap"] = df.apply(snap_flag, axis=1).fillna(0).astype(int)

    keep = ["date","wm_yr_wk","state_id","store_id","cat_id","dept_id","item_id",
            "sales","sell_price","event_name_1","event_type_1","event_name_2","event_type_2","snap"]
    df = df[keep].copy()

    if use_cache:
        df.to_parquet(CACHE, index=False)
    return df

def sample_panel(df: pd.DataFrame, sample_stores: int = 3, sample_items_per_store: int = 30) -> pd.DataFrame:
    stores = df["store_id"].dropna().unique().tolist()[:sample_stores]
    parts = []
    for s in stores:
        g = df[df["store_id"] == s]
        items = g["item_id"].dropna().unique().tolist()[:sample_items_per_store]
        parts.append(g[g["item_id"].isin(items)])
    return pd.concat(parts, ignore_index=True)