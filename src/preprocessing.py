# src/preprocessing.py
from __future__ import annotations

import os
import io
import zipfile
from typing import Optional, Iterable, List

import numpy as np
import pandas as pd


# ------------ Utilities ------------

def _data_dir() -> str:
    """Where data lives. Defaults to 'data' unless DATA_DIR is set."""
    return os.environ.get("DATA_DIR", "data")


def _require_file(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Required data file not found: {path}\n"
            "Expected three files inside your repo:\n"
            "  - data/calendar.csv.zip\n"
            "  - data/sell_prices.csv.zip\n"
            "  - data/sales_train_validation.csv.zip"
        )


def _read_single_csv_from_zip(zip_path: str, member_name: Optional[str] = None, **read_csv_kw) -> pd.DataFrame:
    """
    Read exactly one CSV from a .zip.
    If member_name is None, will auto-pick the single CSV inside.
    """
    _require_file(zip_path)
    with zipfile.ZipFile(zip_path, "r") as zf:
        # If a specific member is requested, use it.
        if member_name is not None:
            with zf.open(member_name) as f:
                return pd.read_csv(f, **read_csv_kw)

        # Otherwise, find exactly one CSV in the archive.
        csv_members = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if len(csv_members) != 1:
            raise FileNotFoundError(
                f"{zip_path} must contain exactly one CSV, found: {csv_members}"
            )
        with zf.open(csv_members[0]) as f:
            return pd.read_csv(f, **read_csv_kw)


def _coalesce_snap(df: pd.DataFrame) -> pd.Series:
    """Create a single SNAP indicator from any snap_* columns (M5 style)."""
    snap_cols = [c for c in df.columns if c.lower().startswith("snap")]
    if not snap_cols:
        return pd.Series(0, index=df.index, dtype=int)
    # max across available SNAP flags → 0/1
    out = df[snap_cols].fillna(0)
    # Coerce to int in case of booleans
    out = out.astype(int)
    return out.max(axis=1).astype(int)


# ------------ Public API ------------

def load_merge(use_cache: bool = True) -> pd.DataFrame:
    """
    Build the analysis-ready daily panel as a DataFrame with columns:
      ['date','store_id','item_id','dept_id','cat_id',
       'sales','sell_price','wm_yr_wk','event_name_1','snap']

    Reads *only* from CSV ZIPs inside DATA_DIR. No parquet is written or read.
    """
    ddir = _data_dir()
    cal_zip = os.path.join(ddir, "calendar.csv.zip")
    price_zip = os.path.join(ddir, "sell_prices.csv.zip")
    sales_zip = os.path.join(ddir, "sales_train_validation.csv.zip")

    # 1) Calendar
    calendar = _read_single_csv_from_zip(cal_zip)
    # Normalize column names just in case
    calendar.columns = [c.strip() for c in calendar.columns]
    # Required columns in calendar for M5
    # ('d','date','wm_yr_wk', optional 'event_name_1', and snap flags)
    if "date" not in calendar.columns:
        raise ValueError("calendar.csv is missing required column 'date'")
    if "d" not in calendar.columns:
        raise ValueError("calendar.csv is missing required column 'd'")
    if "wm_yr_wk" not in calendar.columns:
        raise ValueError("calendar.csv is missing required column 'wm_yr_wk'")

    calendar["date"] = pd.to_datetime(calendar["date"])
    calendar["snap"] = _coalesce_snap(calendar)
    cal_keep = ["d", "date", "wm_yr_wk", "event_name_1", "snap"]
    cal_keep = [c for c in cal_keep if c in calendar.columns]
    calendar = calendar[cal_keep].copy()

    # 2) Prices
    prices = _read_single_csv_from_zip(price_zip)
    prices.columns = [c.strip() for c in prices.columns]
    # Required: store_id, item_id, wm_yr_wk, sell_price
    needed_price_cols = {"store_id", "item_id", "wm_yr_wk", "sell_price"}
    if not needed_price_cols.issubset(prices.columns):
        raise ValueError(
            f"sell_prices.csv is missing columns: {needed_price_cols - set(prices.columns)}"
        )

    # 3) Sales (wide M5 format expected)
    sales = _read_single_csv_from_zip(sales_zip)
    sales.columns = [c.strip() for c in sales.columns]

    # Detect M5 wide vs already-long.
    day_cols = [c for c in sales.columns if c.startswith("d_")]
    id_cols = [c for c in ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"] if c in sales.columns]

    if day_cols and id_cols:
        # M5 wide → long
        long_sales = (
            sales.melt(id_vars=id_cols, value_vars=day_cols, var_name="d", value_name="sales")
                 .rename(columns={"value": "sales"})
        )
        # Ensure numeric sales
        long_sales["sales"] = pd.to_numeric(long_sales["sales"], errors="coerce").fillna(0.0)

        # Merge calendar on 'd' to get dates & week keys
        long_sales = long_sales.merge(calendar, on="d", how="left")

        # Prices merged by (store_id, item_id, wm_yr_wk)
        price_keys = ["store_id", "item_id", "wm_yr_wk"]
        long_sales = long_sales.merge(prices[price_keys + ["sell_price"]], on=price_keys, how="left")

        # Keep tidy subset and sort
        keep = ["date", "store_id", "item_id", "dept_id", "cat_id",
                "sales", "sell_price", "wm_yr_wk"]
        if "event_name_1" in long_sales.columns:
            keep.append("event_name_1")
        if "snap" in long_sales.columns:
            keep.append("snap")

        panel = (
            long_sales[keep]
            .sort_values(["store_id", "item_id", "date"])
            .reset_index(drop=True)
        )

    else:
        # Already long (fallback). Expect at minimum: date, store_id, item_id, sales
        needed = {"date", "store_id", "item_id", "sales"}
        if not needed.issubset(sales.columns):
            raise ValueError(
                "sales_train_validation.csv appears not to be in M5 wide format, "
                "and also lacks the minimal long-format columns {date, store_id, item_id, sales}."
            )
        tmp = sales.copy()
        tmp["date"] = pd.to_datetime(tmp["date"])

        # Best-effort joins to bring in calendar & prices
        tmp = tmp.merge(calendar.drop(columns=["d"]), on="date", how="left")
        tmp = tmp.merge(
            prices[["store_id", "item_id", "wm_yr_wk", "sell_price"]],
            on=["store_id", "item_id", "wm_yr_wk"], how="left"
        )

        keep = ["date", "store_id", "item_id", "sales", "sell_price", "wm_yr_wk"]
        for opt in ["dept_id", "cat_id", "event_name_1", "snap"]:
            if opt in tmp.columns:
                keep.append(opt)

        panel = (
            tmp[keep]
            .sort_values(["store_id", "item_id", "date"])
            .reset_index(drop=True)
        )

    # Final sanity: coerce types we chart a lot
    panel["sales"] = pd.to_numeric(panel["sales"], errors="coerce").fillna(0.0)
    if "sell_price" in panel.columns:
        panel["sell_price"] = pd.to_numeric(panel["sell_price"], errors="coerce")

    # Guard against empty merges
    if panel.empty:
        raise ValueError(
            "After merging calendar/prices/sales, the panel is empty. "
            "Double-check that the three source files are the standard M5 versions."
        )

    return panel


def sample_panel(panel: pd.DataFrame, n_stores: int, n_items: int) -> pd.DataFrame:
    """
    Downsample for UI speed: keep up to `n_stores`, and for each, up to `n_items`.
    """
    if panel.empty:
        return panel

    stores = (panel["store_id"].dropna().unique().tolist())[:max(1, int(n_stores))]
    pieces: List[pd.DataFrame] = []
    for s in stores:
        items = (
            panel.loc[panel["store_id"] == s, "item_id"]
                 .dropna().unique().tolist()
        )[:max(1, int(n_items))]
        part = panel[(panel["store_id"] == s) & (panel["item_id"].isin(items))]
        if not part.empty:
            pieces.append(part)

    return pd.concat(pieces, ignore_index=True) if pieces else panel.head(0)
