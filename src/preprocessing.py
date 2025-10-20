import os, hashlib, requests, pandas as pd

def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def _ensure_parquet_local(local_path="data/_merged_m5_zstd.parquet"):
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    if os.path.exists(local_path):
        return local_path
    url = os.getenv("RELEASE_URL", "")
    if not url:
        raise FileNotFoundError("Parquet not found locally and RELEASE_URL not set.")
    r = requests.get(url, stream=True, timeout=120)
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
            raise RuntimeError(f"SHA256 mismatch: got {got} expected {want}")
    os.replace(tmp, local_path)
    return local_path

def load_merge(use_cache=True) -> pd.DataFrame:
    path = _ensure_parquet_local("data/_merged_m5_zstd.parquet")
    return pd.read_parquet(path)

def sample_panel(df: pd.DataFrame, n_stores: int, n_items: int) -> pd.DataFrame:
    if {"store_id","item_id"}.issubset(df.columns):
        stores = df["store_id"].dropna().unique()[: n_stores]
        out = []
        for s in stores:
            sub = df[df["store_id"] == s]
            items = sub["item_id"].dropna().unique()[: n_items]
            out.append(sub[sub["item_id"].isin(items)])
        return pd.concat(out, ignore_index=True) if out else df.head(0)
    return df