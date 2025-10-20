from __future__ import annotations
import os
import numpy as np
import pandas as pd

DATA_DIR = os.environ.get("M5_DATA_DIR", "data")
OUT_PATH = os.path.join(DATA_DIR, "_merged_m5_zstd.parquet")

N_STORES = int(os.environ.get("LITE_STORES", "3"))
N_ITEMS_PER_STORE = int(os.environ.get("LITE_ITEMS", "30"))

def _require_files():
    need = ["calendar.csv", "sell_prices.csv", "sales_train_validation.csv"]
    missing = [f for f in need if not os.path.exists(os.path.join(DATA_DIR, f))]
    if missing:
        raise FileNotFoundError(f"Missing file(s) in ./data: {', '.join(missing)}")

def _select_top_items_per_store(sales_wide: pd.DataFrame, stores: list[str], n_items: int) -> pd.DataFrame:
    day_cols = [c for c in sales_wide.columns if c.startswith("d_")]
    small = sales_wide[sales_wide["store_id"].isin(stores)].copy()
    keep = []
    for s in stores:
        sub = small[small["store_id"] == s].copy()
        if sub.empty:
            continue
        sub["total_units"] = sub[day_cols].sum(axis=1)
        keep.append(sub.sort_values("total_units", ascending=False).head(n_items).drop(columns="total_units"))
    return pd.concat(keep, ignore_index=True) if keep else pd.DataFrame(columns=sales_wide.columns)

def _build_panel(calendar: pd.DataFrame, prices: pd.DataFrame, sales_small: pd.DataFrame) -> pd.DataFrame:
    id_cols  = [c for c in sales_small.columns if not c.startswith("d_")]
    day_cols = [c for c in sales_small.columns if c.startswith("d_")]

    long = sales_small.melt(id_vars=id_cols, value_vars=day_cols, var_name="d", value_name="sales")

    cal_cols = ["d","date","wm_yr_wk","event_name_1","event_type_1"]
    for c in ["snap_CA","snap_TX","snap_WI"]:
        if c in calendar.columns:
            cal_cols.append(c)
    cal = calendar[cal_cols].copy()
    cal["date"] = pd.to_datetime(cal["date"])
    long = long.merge(cal, on="d", how="left")

    pr = prices[["store_id","item_id","wm_yr_wk","sell_price"]].copy()
    long = long.merge(pr, on=["store_id","item_id","wm_yr_wk"], how="left")

    long = long.sort_values(["store_id","item_id","date"]).reset_index(drop=True)

    # *** FIX: use transform, not apply, to preserve index alignment ***
    long["sell_price"] = (
        long.groupby(["store_id","item_id"], group_keys=False)["sell_price"]
            .transform(lambda s: pd.to_numeric(s, errors="coerce").ffill().bfill())
    )

    long["sales"] = pd.to_numeric(long["sales"], errors="coerce").fillna(0.0).astype(float)
    if "date" in long.columns:
        long["dow"] = long["date"].dt.day_name()

    # bring cat/dept if available in the wide table
    for col in ["cat_id","dept_id"]:
        if col not in long.columns and col in sales_small.columns:
            long = long.merge(sales_small[["item_id", col]].drop_duplicates(), on="item_id", how="left")

    # choose available columns safely
    preferred = ["date","store_id","item_id","sales","sell_price",
                 "event_name_1","event_type_1","snap_CA","snap_TX","snap_WI",
                 "cat_id","dept_id","dow"]
    cols = [c for c in preferred if c in long.columns]
    return long[cols]

def main():
    _require_files()
    calendar = pd.read_csv(os.path.join(DATA_DIR, "calendar.csv"))
    prices   = pd.read_csv(os.path.join(DATA_DIR, "sell_prices.csv"))
    sales    = pd.read_csv(os.path.join(DATA_DIR, "sales_train_validation.csv"))

    stores_all = sales["store_id"].dropna().unique().tolist()
    stores_sel = stores_all[:min(N_STORES, len(stores_all))]

    sales_small = _select_top_items_per_store(sales, stores_sel, N_ITEMS_PER_STORE)
    if sales_small.empty:
        raise RuntimeError("No rows selected; check LITE_STORES/LITE_ITEMS and input files.")

    panel = _build_panel(calendar, prices, sales_small)

    os.makedirs(DATA_DIR, exist_ok=True)
    panel.to_parquet(OUT_PATH, compression="zstd", index=False)

    print(f"✅ wrote: {OUT_PATH}")
    print(f"rows={len(panel):,} cols={len(panel.columns)} "
          f"stores={panel['store_id'].nunique()} items={panel['item_id'].nunique()}")

if __name__ == "__main__":
    main()