from __future__ import annotations
import os
import pandas as pd
import numpy as np

# ---------- small helpers ----------
def _read_csv_maybe_zip(path_csv: str) -> pd.DataFrame:
    if os.path.exists(path_csv):
        return pd.read_csv(path_csv)
    z = path_csv + ".zip"
    if os.path.exists(z):
        return pd.read_csv(z, compression="zip")
    raise FileNotFoundError(f"Missing file: {path_csv} (or {path_csv}.zip)")

def _ensure_datetime(df: pd.DataFrame, col: str = "date") -> pd.DataFrame:
    if col in df.columns and not np.issubdtype(df[col].dtype, np.datetime64):
        df[col] = pd.to_datetime(df[col])
    return df

# ---------- loaders ----------
def _load_calendar(data_dir: str) -> pd.DataFrame:
    cal = _read_csv_maybe_zip(os.path.join(data_dir, "calendar.csv"))
    keep = ["date", "wm_yr_wk", "d", "event_name_1", "event_type_1", "snap_CA", "snap_TX", "snap_WI"]
    keep = [c for c in keep if c in cal.columns]
    cal = cal[keep].copy()
    _ensure_datetime(cal, "date")
    for c in ["snap_CA", "snap_TX", "snap_WI"]:
        if c in cal.columns:
            cal[c] = cal[c].fillna(0).astype(int)
    snaps = [c for c in ["snap_CA", "snap_TX", "snap_WI"] if c in cal.columns]
    cal["snap"] = cal[snaps].max(axis=1) if snaps else 0
    return cal

def _load_sell_prices(data_dir: str) -> pd.DataFrame:
    prices = _read_csv_maybe_zip(os.path.join(data_dir, "sell_prices.csv"))
    need = ["store_id", "item_id", "wm_yr_wk", "sell_price"]
    miss = [c for c in need if c not in prices.columns]
    if miss:
        raise ValueError(f"sell_prices.csv missing columns: {miss}")
    return prices[need].copy()

def _read_sales_wide(data_dir: str) -> pd.DataFrame:
    # accept either validation or evaluation file name
    candidates = [
        os.path.join(data_dir, "sales_train_validation.csv"),
        os.path.join(data_dir, "sales_train_evaluation.csv"),
    ]
    errs = []
    for path in candidates:
        try:
            return _read_csv_maybe_zip(path)
        except FileNotFoundError as e:
            errs.append(str(e))
    raise FileNotFoundError(
        "Neither sales_train_validation.csv(.zip) nor sales_train_evaluation.csv(.zip) found.\n"
        + "\n".join(errs)
    )

def _load_sales_long(data_dir: str, lite_stores: int | None, lite_items: int | None) -> pd.DataFrame:
    wide = _read_sales_wide(data_dir)
    meta = [c for c in ["id","item_id","dept_id","cat_id","store_id","state_id"] if c in wide.columns]
    dcols = [c for c in wide.columns if c.startswith("d_")]
    if not dcols:
        raise ValueError("Sales file has no day columns like 'd_1', 'd_2', ...")

    # optional downsample
    if lite_stores or lite_items:
        totals = wide[meta + dcols].copy()
        totals["total"] = totals[dcols].sum(axis=1)

        if lite_stores:
            top_stores = (totals.groupby("store_id", as_index=False)["total"]
                          .sum().sort_values("total", ascending=False)
                          .head(int(lite_stores))["store_id"].tolist())
            wide = wide[wide["store_id"].isin(top_stores)]
            totals = totals[totals["store_id"].isin(top_stores)]

        if lite_items:
            keep_ids = []
            for _, sub in totals.groupby("store_id"):
                ids = sub.sort_values("total", ascending=False).head(int(lite_items))["id"].tolist()
                keep_ids.extend(ids)
            wide = wide[wide["id"].isin(keep_ids)]

    long = wide.melt(id_vars=meta, value_vars=dcols, var_name="d", value_name="sales")
    long["sales"] = pd.to_numeric(long["sales"], errors="coerce").fillna(0.0)
    return long

# ---------- build tidy panel ----------
def _build_panel(calendar: pd.DataFrame, prices: pd.DataFrame, sales_long: pd.DataFrame) -> pd.DataFrame:
    cal_map = calendar[["d","date","wm_yr_wk","event_name_1","event_type_1","snap"]].copy()
    sales = sales_long.merge(cal_map, on="d", how="left", validate="many_to_one")

    prices_daily = prices.merge(calendar[["wm_yr_wk","date"]].drop_duplicates(),
                                on="wm_yr_wk", how="left", validate="many_to_many")
    prices_daily = prices_daily.drop(columns=["wm_yr_wk"], errors="ignore")

    panel = sales.merge(prices_daily, on=["store_id","item_id","date"], how="left")
    panel = panel.sort_values(["store_id","item_id","date"])
    panel["sell_price"] = panel.groupby(["store_id","item_id"])["sell_price"].transform(lambda s: s.ffill().bfill())

    keep = [c for c in ["date","store_id","item_id","dept_id","cat_id",
                        "sales","sell_price","event_name_1","event_type_1","snap"] if c in panel.columns]
    panel = panel[keep].copy()
    _ensure_datetime(panel, "date")
    return panel

# ---------- public API ----------
def load_merge(use_cache: bool = False) -> pd.DataFrame:
    data_dir = os.getenv("DATA_DIR", "data").strip() or "data"
    lite_stores = os.getenv("LITE_STORES")
    lite_items  = os.getenv("LITE_ITEMS")
    lite_stores = int(lite_stores) if lite_stores not in (None, "", "0") else None
    lite_items  = int(lite_items)  if lite_items  not in (None, "", "0") else None

    # check that required files (or .zip) exist
    required_sets = [
        ("calendar.csv",),
        ("sell_prices.csv",),
        ("sales_train_validation.csv", "sales_train_evaluation.csv"),
    ]
    missing = []
    for names in required_sets:
        # any one of the names in the set is acceptable
        found = False
        for n in names:
            if os.path.exists(os.path.join(data_dir, n)) or os.path.exists(os.path.join(data_dir, n + ".zip")):
                found = True
                break
        if not found:
            # if there are two alternatives, report both
            if len(names) == 1:
                missing.append(names[0])
            else:
                missing.append(f"{names[0]} or {names[1]}")

    if missing:
        raise FileNotFoundError(f"Missing file(s) in {data_dir}: {', '.join(missing)}")

    cal   = _load_calendar(data_dir)
    price = _load_sell_prices(data_dir)
    sales = _load_sales_long(data_dir, lite_stores, lite_items)
    return _build_panel(cal, price, sales)

def sample_panel(panel: pd.DataFrame, n_stores: int = 3, n_items: int = 30) -> pd.DataFrame:
    if panel.empty:
        return panel
    top_stores = (panel.groupby("store_id", as_index=False)["sales"]
                  .sum().sort_values("sales", ascending=False)["store_id"].head(max(1, int(n_stores))).tolist())
    p2 = panel[panel["store_id"].isin(top_stores)].copy()
    out = []
    for _, g in p2.groupby("store_id"):
        keep_items = (g.groupby("item_id", as_index=False)["sales"]
                      .sum().sort_values("sales", ascending=False)["item_id"].head(max(1, int(n_items))).tolist())
        out.append(g[g["item_id"].isin(keep_items)])
    return pd.concat(out, ignore_index=True) if out else p2