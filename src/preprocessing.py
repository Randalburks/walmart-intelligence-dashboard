from __future__ import annotations
import io, os, zipfile, hashlib
from typing import Optional
import requests
import numpy as np
import pandas as pd

# ---------- ZIP/CSV readers (robust to macOS zips) ----------
def _read_csv_maybe_zip_local(path_csv: str) -> pd.DataFrame:
    if os.path.exists(path_csv):
        return pd.read_csv(path_csv)
    z = path_csv + ".zip"
    if os.path.exists(z):
        return _read_csv_from_zip_bytes(open(z, "rb").read())
    raise FileNotFoundError(f"Missing file: {path_csv} (or {path_csv}.zip)")

def _download_bytes(url: str) -> bytes:
    r = requests.get(url, stream=True, timeout=180)
    r.raise_for_status()
    return r.content

def _read_csv_from_zip_bytes(b: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(b)) as z:
        candidates = []
        for name in z.namelist():
            base = os.path.basename(name)
            if not name.lower().endswith(".csv"):
                continue
            if name.startswith("__MACOSX/") or base.startswith("._"):
                continue
            candidates.append(name)
        if not candidates:
            raise FileNotFoundError("No CSV found in ZIP (after excluding __MACOSX/ and hidden files).")
        best = max(candidates, key=lambda n: z.getinfo(n).file_size)
        with z.open(best) as f:
            return pd.read_csv(f)

def _read_csv_maybe_download(url_env: str, fallback_local: str) -> pd.DataFrame:
    url = os.getenv(url_env, "").strip()
    if url:
        data = _download_bytes(url)
        # optional integrity check
        want = os.getenv(url_env + "_SHA256", "").strip().lower()
        if want:
            got = hashlib.sha256(data).hexdigest()
            if got != want:
                raise RuntimeError(f"SHA256 mismatch for {url_env}: got {got} expected {want}")
        # if it’s a direct CSV, load; if it’s a zip, unzip+load
        if url.lower().endswith(".csv"):
            return pd.read_csv(io.BytesIO(data))
        return _read_csv_from_zip_bytes(data)
    # otherwise read from local (csv or csv.zip)
    return _read_csv_maybe_zip_local(fallback_local)

# ---------- loaders ----------
def _ensure_datetime(df: pd.DataFrame, col: str = "date") -> pd.DataFrame:
    if col in df.columns and not np.issubdtype(df[col].dtype, np.datetime64):
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df

def _load_calendar(data_dir: str) -> pd.DataFrame:
    # Prefer env URL CALENDAR_URL; else data/calendar.csv(.zip)
    cal = _read_csv_maybe_download("CALENDAR_URL", os.path.join(data_dir, "calendar.csv"))
    keep = ["date","wm_yr_wk","d","event_name_1","event_type_1","snap_CA","snap_TX","snap_WI"]
    keep = [c for c in keep if c in cal.columns]
    cal = cal[keep].copy()
    _ensure_datetime(cal, "date")
    for c in ["snap_CA","snap_TX","snap_WI"]:
        if c in cal.columns:
            cal[c] = pd.to_numeric(cal[c], errors="coerce").fillna(0).astype(int)
    snaps = [c for c in ["snap_CA","snap_TX","snap_WI"] if c in cal.columns]
    cal["snap"] = cal[snaps].max(axis=1) if snaps else 0
    return cal

def _load_sell_prices(data_dir: str) -> pd.DataFrame:
    prices = _read_csv_maybe_download("SELL_PRICES_URL", os.path.join(data_dir, "sell_prices.csv"))
    need = ["store_id","item_id","wm_yr_wk","sell_price"]
    miss = [c for c in need if c not in prices.columns]
    if miss:
        raise ValueError(f"sell_prices.csv missing columns: {miss}")
    prices["sell_price"] = pd.to_numeric(prices["sell_price"], errors="coerce")
    return prices[need].copy()

def _load_sales_long(data_dir: str, lite_stores: Optional[int], lite_items: Optional[int]) -> pd.DataFrame:
    sales = _read_csv_maybe_download("SALES_VALIDATION_URL", os.path.join(data_dir, "sales_train_validation.csv"))
    meta = [c for c in ["id","item_id","dept_id","cat_id","store_id","state_id"] if c in sales.columns]
    dcols = [c for c in sales.columns if c.startswith("d_")]
    if not dcols:
        raise ValueError("sales_train_validation.csv has no day columns like 'd_1'…")

    # Lite mode (top stores/items) to reduce memory if desired
    if lite_stores or lite_items:
        totals = sales[meta + dcols].copy()
        totals["total"] = totals[dcols].sum(axis=1)
        if lite_stores:
            top_stores = (totals.groupby("store_id", as_index=False)["total"]
                          .sum().sort_values("total", ascending=False)
                          .head(int(lite_stores))["store_id"].tolist())
            sales = sales[sales["store_id"].isin(top_stores)]
            totals = totals[totals["store_id"].isin(top_stores)]
        if lite_items:
            keep_ids = []
            for st, sub in totals.groupby("store_id"):
                ids = sub.sort_values("total", ascending=False).head(int(lite_items))["id"].tolist()
                keep_ids.extend(ids)
            sales = sales[sales["id"].isin(keep_ids)]

    long = sales.melt(id_vars=meta, value_vars=dcols, var_name="d", value_name="sales")
    long["sales"] = pd.to_numeric(long["sales"], errors="coerce").fillna(0.0)
    return long

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
def load_merge(use_cache: bool = True) -> pd.DataFrame:
    data_dir = os.getenv("DATA_DIR", "data")
    lite_stores = os.getenv("LITE_STORES")
    lite_items  = os.getenv("LITE_ITEMS")
    lite_stores = int(lite_stores) if lite_stores not in (None, "", "0") else None
    lite_items  = int(lite_items)  if lite_items  not in (None, "", "0") else None

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
    for st, g in p2.groupby("store_id"):
        keep_items = (g.groupby("item_id", as_index=False)["sales"]
                      .sum().sort_values("sales", ascending=False)["item_id"].head(max(1, int(n_items))).tolist())
        out.append(g[g["item_id"].isin(keep_items)])
    return pd.concat(out, ignore_index=True) if out else p2