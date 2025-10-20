# src/preprocessing.py
from __future__ import annotations
from typing import Tuple

import io
import os
import zipfile
import pandas as pd
import numpy as np

DATA_DIR = "data"
CAL_ZIP = os.path.join(DATA_DIR, "calendar.csv.zip")
PRC_ZIP = os.path.join(DATA_DIR, "sell_prices.csv.zip")
SAL_ZIP = os.path.join(DATA_DIR, "sales_train_validation.csv.zip")


def _read_csv_from_zip(zip_path: str, inner_csv_name: str | None = None, **read_kwargs) -> pd.DataFrame:
    """
    Robust reader for a single-CSV ZIP. If inner_csv_name is None, it will read the first .csv it finds.
    """
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"Missing required file: {zip_path}")
    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not names:
            raise FileNotFoundError(f"No CSV found inside {zip_path}")
        name = inner_csv_name or names[0]
        with zf.open(name) as fh:
            return pd.read_csv(fh, **read_kwargs)


def _prep_calendar(cal: pd.DataFrame) -> pd.DataFrame:
    # Expect columns: d, date, wm_yr_wk, event_name_1, snap_CA, snap_TX, snap_WI ...
    cal = cal.copy()
    cal["date"] = pd.to_datetime(cal["date"])
    return cal[["d", "date", "wm_yr_wk", "event_name_1", "snap_CA", "snap_TX", "snap_WI"]].copy()


def _prep_prices(prc: pd.DataFrame) -> pd.DataFrame:
    # Expect columns: store_id, item_id, wm_yr_wk, sell_price
    prc = prc.copy()
    return prc[["store_id", "item_id", "wm_yr_wk", "sell_price"]].copy()


def _prep_sales_long(sales_wide: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    """
    M5-style: sales columns are d_1, d_2, ...
    We melt to long and attach actual dates via calendar (d -> date).
    Keep id columns: item_id, dept_id, cat_id, store_id, state_id (if present).
    """
    id_cols = [c for c in ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"] if c in sales_wide.columns]
    val_cols = [c for c in sales_wide.columns if c.startswith("d_")]
    long_sales = sales_wide.melt(id_vars=id_cols, value_vars=val_cols, var_name="d", value_name="sales")

    # attach date / wm_yr_wk / events / snap via calendar
    long_sales = long_sales.merge(calendar[["d", "date", "wm_yr_wk", "event_name_1"]], on="d", how="left")

    # derive SNAP by state if state_id exists, else from store_id prefix (CA/TX/WI)
    if "state_id" in long_sales.columns:
        long_sales["state_id"] = long_sales["state_id"].astype(str)
        snap = []
        cal_snap = calendar[["d", "snap_CA", "snap_TX", "snap_WI"]].set_index("d")
        for _, row in long_sales[["d", "state_id"]].iterrows():
            sid = row["state_id"]
            if sid == "CA":
                snap.append(cal_snap.loc[row["d"], "snap_CA"])
            elif sid == "TX":
                snap.append(cal_snap.loc[row["d"], "snap_TX"])
            elif sid == "WI":
                snap.append(cal_snap.loc[row["d"], "snap_WI"])
            else:
                snap.append(0)
        long_sales["snap"] = snap
    else:
        # fallback: infer state from store_id prefix (e.g., CA_1)
        def _store_to_state(s: str) -> str:
            try:
                return str(s).split("_")[0]
            except Exception:
                return ""
        long_sales["__state__"] = long_sales["store_id"].apply(_store_to_state)
        cal_snap = calendar[["d", "snap_CA", "snap_TX", "snap_WI"]].set_index("d")
        snap = []
        for _, row in long_sales[["d", "__state__"]].iterrows():
            sid = row["__state__"]
            if sid == "CA":
                snap.append(cal_snap.loc[row["d"], "snap_CA"])
            elif sid == "TX":
                snap.append(cal_snap.loc[row["d"], "snap_TX"])
            elif sid == "WI":
                snap.append(cal_snap.loc[row["d"], "snap_WI"])
            else:
                snap.append(0)
        long_sales["snap"] = snap
        long_sales.drop(columns="__state__", inplace=True)

    long_sales["sales"] = pd.to_numeric(long_sales["sales"], errors="coerce").fillna(0).astype(float)
    # keep only necessary columns; date comes from calendar
    keep = [c for c in ["date", "store_id", "item_id", "dept_id", "cat_id", "wm_yr_wk", "event_name_1", "snap", "sales"] if c in long_sales.columns]
    long_sales = long_sales[keep].copy()
    long_sales["date"] = pd.to_datetime(long_sales["date"])
    return long_sales


def _attach_prices(long_sales: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """
    Merge item/store-week prices to daily rows via wm_yr_wk. Then carry-fill within item_id+store_id to daily.
    """
    out = long_sales.merge(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")
    # fill daily price within each item/store by forward/back fill on date sort
    out = out.sort_values(["store_id", "item_id", "date"])
    out["sell_price"] = (
        out.groupby(["store_id", "item_id"], observed=True)["sell_price"]
        .apply(lambda s: s.ffill().bfill())
        .values
    )
    return out


def load_merge(use_cache: bool = True) -> pd.DataFrame:
    """
    Read three ZIP files and build a tidy daily panel:
      date, store_id, item_id, sales, sell_price, wm_yr_wk, event_name_1, cat_id, dept_id, snap
    """
    cal = _read_csv_from_zip(CAL_ZIP)
    prc = _read_csv_from_zip(PRC_ZIP)
    sal = _read_csv_from_zip(SAL_ZIP)

    cal = _prep_calendar(cal)
    prc = _prep_prices(prc)
    long_sales = _prep_sales_long(sal, cal)
    panel = _attach_prices(long_sales, prc)

    # Basic hygiene
    panel = panel.dropna(subset=["date", "store_id", "item_id"]).copy()
    # enforce dtypes
    panel["store_id"] = panel["store_id"].astype(str)
    panel["item_id"] = panel["item_id"].astype(str)
    for c in ["dept_id", "cat_id"]:
        if c in panel.columns:
            panel[c] = panel[c].astype(str)
    if "sell_price" in panel.columns:
        panel["sell_price"] = pd.to_numeric(panel["sell_price"], errors="coerce")
    if "snap" in panel.columns:
        panel["snap"] = pd.to_numeric(panel["snap"], errors="coerce").fillna(0).astype(int)

    # Remove any accidental duplicate rows per (date, store_id, item_id) by summing sales and averaging price
    if not panel.empty:
        agg = {
            "sales": "sum",
            "sell_price": "mean",
            "wm_yr_wk": "first",
            "event_name_1": "first",
            "cat_id": "first",
            "dept_id": "first",
            "snap": "max",
        }
        exist_cols = [k for k in agg.keys() if k in panel.columns]
        panel = panel.groupby(["date", "store_id", "item_id"], as_index=False)[exist_cols].agg(agg)

    return panel


def sample_panel(panel: pd.DataFrame, n_stores: int, n_items: int) -> pd.DataFrame:
    """
    Return a smaller panel for UI responsiveness:
    - Pick top n_stores by total sales
    - Within each store, pick top n_items by total sales
    """
    if panel.empty:
        return panel.copy()

    store_rank = panel.groupby("store_id", as_index=False)["sales"].sum().sort_values("sales", ascending=False)
    keep_stores = store_rank["store_id"].head(n_stores).tolist()

    out_frames = []
    for s in keep_stores:
        g = panel[panel["store_id"] == s]
        item_rank = g.groupby("item_id", as_index=False)["sales"].sum().sort_values("sales", ascending=False)
        keep_items = item_rank["item_id"].head(n_items).tolist()
        out_frames.append(g[g["item_id"].isin(keep_items)])
    small = pd.concat(out_frames, ignore_index=True) if out_frames else panel.head(0)

    # ensure valid date type
    small["date"] = pd.to_datetime(small["date"])
    return small