# src/preprocessing.py
from __future__ import annotations
import os, hashlib, requests
import pandas as pd
import streamlit as st

def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def _get_secret(name: str, default: str = "") -> str:
    try:
        # Streamlit secrets first
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass
    # fallback to environment (local dev)
    return os.getenv(name, default)

def _ensure_parquet_local(local_path: str = "data/_merged_m5_zstd.parquet") -> str:
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    if os.path.exists(local_path) and os.path.getsize(local_path) > 1024:
        return local_path

    url = _get_secret("RELEASE_URL", "").strip()
    want = _get_secret("RELEASE_SHA256", "").strip().lower()

    if not url:
        st.error("Data file not found and no RELEASE_URL provided in **App settings → Secrets**.")
        raise FileNotFoundError("Missing RELEASE_URL secret")

    st.info(f"Downloading data… ({os.path.basename(local_path)})")
    r = requests.get(url, stream=True, timeout=300)
    r.raise_for_status()
    tmp = local_path + ".part"
    with open(tmp, "wb") as f:
        for chunk in r.iter_content(1024 * 1024):
            if chunk:
                f.write(chunk)

    if want:
        got = _sha256(tmp)
        if got != want:
            try: os.remove(tmp)
            except Exception: pass
            st.error(f"SHA256 mismatch. Got {got}, expected {want}. Check the secret or reupload the asset.")
            raise RuntimeError("SHA256 mismatch")

    os.replace(tmp, local_path)
    st.success("Data ready.")
    return local_path

def load_merge(use_cache: bool = True) -> pd.DataFrame:
    path = _ensure_parquet_local("data/_merged_m5_zstd.parquet")
    # If you uploaded a different codec/filename, you can still use the same path after download
    return pd.read_parquet(path)