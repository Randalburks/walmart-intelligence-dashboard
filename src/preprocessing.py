import os
import pandas as pd

def load_merge(use_cache: bool = False) -> pd.DataFrame:
    """
    Build the daily panel from CSV ZIPs only:
      - data/calendar.csv.zip
      - data/sell_prices.csv.zip
      - data/sales_train_validation.csv.zip
    Returns a DataFrame with: date, store_id, item_id, sales, sell_price, wm_yr_wk, event_name_1, snap (when present)
    """
    data_dir = os.environ.get("DATA_DIR", "data")

    cal_path   = os.path.join(data_dir, "calendar.csv.zip")
    price_path = os.path.join(data_dir, "sell_prices.csv.zip")
    sales_path = os.path.join(data_dir, "sales_train_validation.csv.zip")

    for p in (cal_path, price_path, sales_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing required file: {p}")

   
    calendar = pd.read_csv(cal_path, compression="zip")
    prices   = pd.read_csv(price_path, compression="zip")
    sales    = pd.read_csv(sales_path, compression="zip")


    id_cols = [c for c in sales.columns if not c.startswith("d_")]
    value_cols = [c for c in sales.columns if c.startswith("d_")]
    long_sales = sales.melt(id_vars=id_cols, value_vars=value_cols, var_name="d", value_name="sales")


    cal_keep = calendar[["d","date","wm_yr_wk","event_name_1"] + [c for c in calendar.columns if c.startswith("snap_")]].copy()
    long_sales = long_sales.merge(cal_keep, on="d", how="left")


    if "store_id" not in long_sales.columns:
        if "store_id" in sales.columns:
            long_sales["store_id"] = sales["store_id"].repeat(len(value_cols)).values
        elif "id" in long_sales.columns:
            long_sales["store_id"] = long_sales["id"].str.extract(r"_(CA|TX|WI)_\d+")[0].fillna("STORE_1")


    price_keys = [c for c in ["store_id","item_id","wm_yr_wk"] if c in prices.columns]
    prices_keep = prices[[c for c in prices.columns if c in price_keys + ["sell_price"]]]
    long_sales = long_sales.merge(prices_keep, on=price_keys, how="left")

    long_sales["date"] = pd.to_datetime(long_sales["date"])

    snap_cols = [c for c in long_sales.columns if c.startswith("snap_")]
    if snap_cols:
        long_sales["snap"] = long_sales[snap_cols].max(axis=1).fillna(0).astype(int)
    elif "snap" not in long_sales.columns:
        long_sales["snap"] = 0

    keep = [c for c in ["date","store_id","item_id","sales","sell_price","wm_yr_wk","event_name_1","snap"] if c in long_sales.columns]
    panel = long_sales[keep].sort_values(["store_id","item_id","date"]).reset_index(drop=True)

    return panel
