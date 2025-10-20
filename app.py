# app.py
from __future__ import annotations
from typing import Optional, Tuple
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

from src.preprocessing import load_merge, sample_panel
from src.utils import kpis_for_slice, anomaly_flags, seasonal_naive, moving_avg, safe_num

# --- import bootstrap (keeps working on Streamlit Cloud and locally) ---
import os, sys
REPO_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(REPO_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

try:
    from src.preprocessing import load_merge, sample_panel
    from src.utils import kpis_for_slice, anomaly_flags, seasonal_naive, moving_avg, safe_num
except ModuleNotFoundError:
    from preprocessing import load_merge, sample_panel
    from utils import kpis_for_slice, anomaly_flags, seasonal_naive, moving_avg, safe_num
# -----------------------------------------------------------------------
st.set_page_config(
    page_title="Walmart Intelligence — Executive Sales, Pricing, Forecasting & Scenarios",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title("Walmart Intelligence — Executive Sales, Pricing, Forecasting & Scenarios")

st.markdown("""
### Purpose
This dashboard turns Walmart sales, pricing, and event data into decisions. It links daily operations to strategic questions like **how prices affect units**, **which stores or products are outperforming**, and **where demand is trending**. Each view is designed so leaders can explore performance, test price ideas, and export a one-page summary with confidence.
""")

# ---------- Data & sidebar ----------
try:
    base = load_merge(use_cache=True)
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

with st.sidebar:
    st.header("Filters")
    n_stores = st.number_input("How many stores to include", 1, 10, 3, step=1)
    n_items  = st.number_input("How many products per store", 5, 200, 30, step=5)
    st.markdown("---")
    st.markdown("Upload optional costs to enable profit tiles.")
    cost_file = st.file_uploader("costs.csv (store_id,item_id,unit_cost)", type=["csv"])

panel = sample_panel(base, n_stores, n_items)

costs_df = None
if cost_file is not None:
    try:
        tmp = pd.read_csv(cost_file)
        if {"store_id","item_id","unit_cost"}.issubset(tmp.columns):
            costs_df = tmp.copy()
        else:
            st.sidebar.warning("Expected columns: store_id, item_id, unit_cost")
    except Exception as e:
        st.sidebar.warning(f"Could not read costs.csv: {e}")

stores = sorted(panel["store_id"].dropna().unique().tolist())
store_sel = st.sidebar.selectbox("Store", stores, index=0 if stores else None)
items = sorted(panel.loc[panel["store_id"] == store_sel, "item_id"].dropna().unique().tolist())
item_sel = st.sidebar.selectbox("Product", items, index=0 if items else None)

dmin, dmax = pd.to_datetime(panel["date"].min()), pd.to_datetime(panel["date"].max())
dr = st.sidebar.date_input("Date range", (dmin.date(), dmax.date()),
                           min_value=dmin.date(), max_value=dmax.date())
start_ts = pd.Timestamp(dr[0])
end_ts   = pd.Timestamp(dr[1])

slice_df = panel[
    (panel["store_id"] == store_sel) &
    (panel["item_id"] == item_sel) &
    (panel["date"].between(start_ts, end_ts))
].copy()

if costs_df is not None and not slice_df.empty:
    slice_df = slice_df.merge(costs_df, on=["store_id","item_id"], how="left")

# ---------- Helpers ----------
def estimate_elasticity(df_slice: pd.DataFrame) -> Optional[dict]:
    g = df_slice.dropna(subset=["sell_price","sales"])
    g = g[(g["sell_price"] > 0) & (g["sales"] > 0)]
    if len(g) < 30:
        return None
    X = np.log(g[["sell_price"]].values)
    y = np.log(g["sales"].values)
    m = LinearRegression().fit(X, y)
    yhat = m.predict(X)
    return {
        "elasticity": float(m.coef_[0]),
        "intercept": float(m.intercept_),
        "mape": float(mean_absolute_percentage_error(y, yhat)),
        "rmse": float(np.sqrt(mean_squared_error(y, yhat))),
        "n": int(len(g))
    }

def simulate_revenue(df_slice: pd.DataFrame, elasticity: float, pct: float) -> dict:
    base_qty = safe_num(df_slice["sales"].mean())
    base_price = safe_num(df_slice["sell_price"].mean())
    new_price = base_price * (1 + pct)
    new_qty = base_qty * (1 + pct) ** elasticity
    base_rev = base_qty * base_price
    new_rev  = new_qty * new_price
    return {
        "new_price": float(new_price),
        "new_qty": float(new_qty),
        "delta_rev_pct": float((new_rev - base_rev) / (base_rev + 1e-9) * 100)
    }

def profit_kpis(df_slice: pd.DataFrame) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    if "unit_cost" not in df_slice.columns or df_slice["unit_cost"].isna().all():
        return None, None, None
    qty = safe_num(df_slice["sales"].mean())
    price = safe_num(df_slice["sell_price"].mean())
    cost  = safe_num(df_slice["unit_cost"].mean())
    margin = price - cost
    profit = margin * qty
    margin_pct = (margin / (price + 1e-9)) * 100 if price > 0 else None
    return float(profit), float(margin), float(margin_pct) if margin_pct is not None else None

def concise_summary(sl: pd.DataFrame, elasticity_val: Optional[float]) -> str:
    if sl.empty:
        return "No data in view. Adjust the date or product."
    k = kpis_for_slice(sl)
    parts = []
    if k["wow_pct"] > 3:
        parts.append(f"Sales up {k['wow_pct']:.1f}% vs last week.")
    elif k["wow_pct"] < -3:
        parts.append(f"Sales down {abs(k['wow_pct']):.1f}% vs last week.")
    else:
        parts.append("Sales stable week-over-week.")
    parts.append(f"Avg price ${k['avg_price']:.2f}.")
    if elasticity_val is not None:
        parts.append("Price-sensitive product." if elasticity_val < -1.0 else "Demand relatively steady vs price.")
    if "event_name_1" in sl.columns:
        recent = sl.loc[sl["event_name_1"].notna(), "event_name_1"].tail(3).unique().tolist()
        if recent:
            parts.append("Recent events: " + ", ".join(recent[:3]) + ".")
    return " ".join(parts)

def add_event_overlays(fig: go.Figure, df_: pd.DataFrame) -> go.Figure:
    if "event_name_1" not in df_.columns:
        return fig
    ev = df_.loc[df_["event_name_1"].notna(), ["date", "event_name_1"]].dropna()
    ymax = max(df_["sales"].max() if "sales" in df_.columns else 0, 1)
    for _, row in ev.iterrows():
        fig.add_vrect(x0=row["date"], x1=row["date"], fillcolor="orange", opacity=0.08, line_width=0)
        fig.add_annotation(x=row["date"], y=ymax, text=str(row["event_name_1"]),
                           showarrow=False, yshift=18, font=dict(size=10, color="gray"))
    return fig

def compute_wow(group_df: pd.DataFrame, freq="W") -> float:
    s = group_df.set_index("date")["sales"].resample(freq).sum()
    if len(s) < 2:
        return 0.0
    prev, curr = float(s.iloc[-2]), float(s.iloc[-1])
    return float(100 * (curr - prev) / (prev + 1e-9))

# ---------- IDA helpers (local) ----------
def info_table(df_: pd.DataFrame) -> pd.DataFrame:
    non_null = df_.notnull().sum()
    out = pd.DataFrame({
        "column": df_.columns,
        "dtype": [str(t) for t in df_.dtypes],
        "non_null": [int(non_null[c]) for c in df_.columns],
        "nulls": [int(len(df_) - non_null[c]) for c in df_.columns]
    })
    out["null_pct"] = (out["nulls"] / max(1, len(df_))).round(4)
    return out

def missingness_bar(df_: pd.DataFrame):
    nn = df_.isnull().mean().sort_values(ascending=False).reset_index()
    nn.columns = ["column","null_fraction"]
    fig = px.bar(nn, x="column", y="null_fraction", title="Missingness by Column")
    fig.update_layout(xaxis_tickangle=-45, yaxis_tickformat=".0%")
    return fig

def missingness_heatmap(df_: pd.DataFrame, sample_rows:int=1200):
    smp = df_.sample(min(sample_rows, len(df_)), random_state=42).isnull().astype(int)
    fig = px.imshow(smp.T, color_continuous_scale="Blues",
                    labels=dict(x="Sampled Row", y="Column", color="Is Null"),
                    title="Missingness Heatmap (sampled rows)")
    return fig

def corr_heatmap(df_: pd.DataFrame, cols: list[str] | None = None):
    """Compute and plot a correlation heatmap for chosen numeric features."""
    if cols is None:
        # sensible defaults if present
        cols = [c for c in ["sales", "sell_price", "wm_yr_wk", "snap"] if c in df_.columns]
        # extend with any other numerics (small set to keep readable)
        extra = [c for c in df_.select_dtypes(include="number").columns if c not in cols]
        cols += extra[:8]  # cap to avoid a huge unreadable matrix
    if not cols:
        return None
    mat = df_[cols].select_dtypes(include="number").corr().round(3)
    fig = px.imshow(mat, text_auto=True, aspect="auto", title="Correlation Matrix (numeric features)")
    return fig

# ---------- Tabs ----------
tabs = st.tabs([
    "IDA + Preprocessing",
    "Overview",
    "Forecast",
    "Price Sensitivity",
    "Scenario Compare",
    "Compare Segments",
    "Top Performers",
    "Summary Export"
])

# ---------- IDA + Preprocessing (first tab) ----------
with tabs[0]:
    st.subheader("IDA + Preprocessing")
    st.markdown("""
I started by validating the raw files and building a clean daily panel that everything else uses. The point of this section is to show the health of the data, the exact steps I took to make it analysis-ready, and what each diagnostic implies for the rest of the app.
""")

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows in sampled panel", f"{len(panel):,}")
    c2.metric("Columns", f"{len(panel.columns):,}")
    c3.metric("Date coverage", f"{dmin.date()} → {dmax.date()}")

    # 1) Structure
    st.markdown("#### 1) Structure table")
    st.dataframe(info_table(panel), use_container_width=True)
    st.markdown("""
**What this shows.** Types, non-null counts, and where gaps exist.  
**Why it’s here.** I confirm that key fields (date, store_id, item_id, sales, sell_price) are populated and that event fields are sparse by design. Gaps in free-text event columns are expected; gaps in prices or sales would not be.
""")

    # 2) Numeric summary
    st.markdown("#### 2) Numeric summary")
    num_cols = panel.select_dtypes(include="number").columns
    if len(num_cols) > 0:
        st.dataframe(panel[num_cols].describe().T.round(3), use_container_width=True)
    else:
        st.info("No numeric columns found.")
    st.markdown("""
**What this shows.** Level and spread of numeric fields (units and price).  
**Why it’s here.** I scan medians vs means for skew, and min/max for outliers. This sets expectations for volatility in Forecast and flags where log-models (elasticity) make sense.
""")

    # 3) Missingness
    st.markdown("#### 3) Missingness")
    cA, cB = st.columns(2)
    with cA:
        st.plotly_chart(missingness_bar(panel), use_container_width=True)
    with cB:
        st.plotly_chart(missingness_heatmap(panel), use_container_width=True)
    st.markdown("""
**What this shows.** Which columns are sparse and whether missingness clusters in certain fields.  
**Why it’s here.** I don’t fill narrative event text; it’s inherently intermittent. For price, I align to daily rows by carrying forward/back within each item so forecasting and elasticity don’t inherit weekly gaps.
""")

    # 4) Correlation matrix (added)
    st.markdown("#### 4) Correlation matrix (features)")
    corr_fig = corr_heatmap(panel)
    if corr_fig is not None:
        st.plotly_chart(corr_fig, use_container_width=True)
    else:
        st.info("Not enough numeric columns to compute a correlation matrix.")
    st.markdown("""
**What this shows.** Linear relationships among numeric features (e.g., sales vs price, sales vs SNAP).  
**Why it’s here.** I check for expected negative price–sales association (elasticity signal) and whether any single feature is overly collinear with others. This helps validate the elasticity approach and ensures our scenario math won’t be distorted by redundant inputs.
""")

    # 5) Behavioral signals (single plot — removed the event-type bar per request)
    st.markdown("#### 5) Behavioral signals")
    ts_all = panel.groupby("date", as_index=False)["sales"].sum()
    st.plotly_chart(
        px.line(ts_all, x="date", y="sales", title="All-sampled items — daily sales"),
        use_container_width=True
    )
    st.markdown("""
**What the results mean.** The daily timeline shows baseline volatility and weekly rhythm, and surfaces structural shifts or shocks.  
**How this drives the dashboard.** Clear weekly rhythm justifies the seasonal baseline in the Forecast tab; shifts or spikes remind us to annotate with events when relevant.
""")

    # 6) Imputation diagnostics
    st.markdown("#### 6) Imputation diagnostics (sell_price)")
    price_missing_after = float(panel["sell_price"].isna().mean()) if "sell_price" in panel.columns else 1.0
    st.caption(f"Price still missing after within-item carry-fill: {price_missing_after:.1%}")
    samp = panel[["item_id","sell_price"]].dropna()
    if not samp.empty:
        samp = samp.sample(min(20000, len(samp)), random_state=42)
        st.plotly_chart(px.violin(samp, y="sell_price", points=False,
                                  box=True, title="Filled price distribution"),
                        use_container_width=True)
    st.markdown("""
**What this shows.** Residual price gaps (ideally near zero) and a sense of price dispersion by item.  
**Why it’s here.** Elasticity and scenarios depend on price being aligned daily. This check confirms the carry-fill worked and that no systematic holes remain.
""")

    # 7) Auto insights
    st.markdown("#### 7) Quick diagnostic insights (auto)")
    insights = []

    itab = info_table(panel)
    high_null_cols = itab.loc[itab["null_pct"] > 0.80, "column"].tolist()
    if high_null_cols:
        insights.append(
            f"High-null context fields detected: {', '.join(high_null_cols)}. They are kept for descriptive context, not modeling."
        )

    try:
        dow_means = panel.assign(dow=panel["date"].dt.day_name())\
                         .groupby("dow", as_index=False)["sales"].mean()
        wknd = dow_means.query("dow in ['Saturday','Sunday']")["sales"].mean()
        mid  = dow_means.query("dow in ['Tuesday','Wednesday','Thursday']")["sales"].mean()
        if np.isfinite(wknd) and np.isfinite(mid) and mid > 0:
            uplift = 100*(wknd - mid)/(mid + 1e-9)
            if abs(uplift) >= 5:
                insights.append(f"Weekly pattern present: weekend vs mid-week ≈ {uplift:+.1f}%. Supports a weekly seasonal baseline.")
    except Exception:
        pass

    if "event_name_1" in panel.columns:
        with_ev = panel.loc[panel["event_name_1"].notna(), "sales"].mean()
        no_ev   = panel.loc[panel["event_name_1"].isna(), "sales"].mean()
        if np.isfinite(with_ev) and np.isfinite(no_ev) and no_ev > 0:
            lift = 100*(with_ev - no_ev)/no_ev
            if abs(lift) >= 3:
                insights.append(f"Events relate to demand: event days differ by ≈ {lift:+.1f}% vs non-event days.")

    if "snap" in panel.columns and panel["snap"].nunique() > 1:
        snap_yes = panel.loc[panel["snap"] == 1, "sales"].mean()
        snap_no  = panel.loc[panel["snap"] == 0, "sales"].mean()
        if np.isfinite(snap_yes) and np.isfinite(snap_no) and snap_no > 0:
            snap_lift = 100*(snap_yes - snap_no)/snap_no
            insights.append(f"SNAP day signal present: ≈ {snap_lift:+.1f}% difference on average (varies by store/item).")

    try:
        flagged = anomaly_flags(slice_df, window=7, z=3.0)
        anom_rate = float(flagged["anomaly"].mean()) if len(flagged) else 0.0
        if anom_rate > 0:
            insights.append(f"Current selection has ≈ {anom_rate:.1%} unusual days (promos/outages/data checks).")
    except Exception:
        pass

    ready = 0
    for (s,i), g in panel.groupby(["store_id","item_id"]):
        g2 = g.dropna(subset=["sell_price","sales"])
        g2 = g2[(g2["sell_price"] > 0) & (g2["sales"] > 0)]
        if len(g2) >= 30:
            ready += 1
    insights.append(f"Elasticity can be estimated for ~{ready} product/store pairs in the sampled panel (≥ 30 valid days).")

    if insights:
        st.markdown("- " + "\n- ".join(insights))

# ---------- Overview ----------
with tabs[1]:
    st.subheader(f"Overview — {store_sel} / {item_sel}")
    with st.expander("Overview", expanded=True):
        st.markdown("Why use it: quick read on level, trend, price, and unusual days. How to use it: choose store, product, dates on the left.")
    if slice_df.empty:
        st.info("No data in the selected range.")
    else:
        k = kpis_for_slice(slice_df)
        profit_val, margin_val, margin_pct = profit_kpis(slice_df)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Avg Daily Units", f"{k['avg_units']:.1f}")
        c2.metric("Avg Price", f"${k['avg_price']:.2f}")
        c3.metric("Week-over-Week", f"{k['wow_pct']:+.1f}%")
        if profit_val is not None:
            c4.metric("Avg Daily Profit", f"${profit_val:,.2f}")
        else:
            c4.metric("Days", f"{len(slice_df):,}")

        if profit_val is not None:
            s1, s2 = st.columns(2)
            s1.metric("Avg Margin ($/unit)", f"${margin_val:,.2f}")
            s2.metric("Avg Margin (%)", f"{margin_pct:.1f}%" if margin_pct is not None else "—")
        else:
            st.caption("Upload costs.csv to enable profit and margin tiles.")

        flagged = anomaly_flags(slice_df, window=7, z=3.0)
        fig = px.line(flagged, x="date", y="sales", labels={"date":"Date","sales":"Units"})
        if flagged["anomaly"].sum() > 0:
            a = flagged[flagged["anomaly"] == 1]
            fig.add_scatter(x=a["date"], y=a["sales"], mode="markers", name="Unusual day")
        fig = add_event_overlays(fig, slice_df)
        st.plotly_chart(fig, use_container_width=True)

        elas = estimate_elasticity(slice_df)
        st.info(concise_summary(slice_df, elas["elasticity"] if elas else None))

# ---------- Forecast ----------
with tabs[2]:
    st.subheader("Forecast — next 4 weeks")
    with st.expander("Forecast", expanded=True):
        st.markdown("Why use it: directional outlook for planning. How to use it: pick ≥ 30 days of history.")
    if slice_df.empty:
        st.info("No data in the selected range.")
    else:
        series = slice_df.set_index("date")["sales"].asfreq("D").fillna(0.0)
        if len(series) < 30:
            st.warning("Select a longer range (30+ days) for a steadier forecast.")
        else:
            fut_idx = pd.date_range(series.index[-1] + pd.Timedelta(days=1), periods=28, freq="D")
            fc1 = seasonal_naive(series, horizon=28, season=7)
            fc2 = moving_avg(series, horizon=28, window=7)
            fig = px.line(x=series.index, y=series.values, labels={"x":"Date","y":"Units"})
            fig.add_scatter(x=fut_idx, y=fc1, name="Season-aware (weekly)")
            fig.add_scatter(x=fut_idx, y=fc2, name="Smoothed trend")
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Use for planning context; actuals will vary with price and events.")

# ---------- Price Sensitivity ----------
with tabs[3]:
    st.subheader("Price Sensitivity (elasticity)")
    with st.expander("Price Sensitivity", expanded=True):
        st.markdown("Why use it: understand unit response to price. How to use it: pick a product with enough history.")
    if slice_df.empty:
        st.info("No data in the selected range.")
    else:
        elas = estimate_elasticity(slice_df)
        if not elas:
            st.info("Insufficient price variation/history (needs ~30+ valid days).")
        else:
            st.success(f"Elasticity: {elas['elasticity']:.2f} (n={elas['n']}, MAPE={elas['mape']:.3f})")
            vals = []
            for (s, i), g in panel.groupby(["store_id","item_id"]):
                e = estimate_elasticity(g)
                if e:
                    vals.append(e["elasticity"])
            if vals:
                h = pd.Series(vals).clip(-5, 5)
                fig = px.histogram(h, nbins=40, labels={"value":"Elasticity","count":"Products"})
                fig.add_vline(x=elas["elasticity"], line_dash="dot", line_color="red")
                st.plotly_chart(fig, use_container_width=True)
                st.caption("Red dot-dash marks this product’s estimate.")

# ---------- Scenario Compare ----------
with tabs[4]:
    st.subheader("Scenario Compare")
    with st.expander("Scenario Compare", expanded=True):
        st.markdown("Why use it: test three price ideas and compare projected revenue change. How to use it: enter Δ price % values.")
    if slice_df.empty:
        st.info("No data in the selected range.")
    else:
        e = estimate_elasticity(slice_df)
        if not e:
            st.info("Price sensitivity not available for this product/date window.")
        else:
            elas_val = e["elasticity"]
            c1, c2, c3 = st.columns(3)
            s1 = c1.number_input("Scenario A (Δ Price %)", -50, 50, -10)
            s2 = c2.number_input("Scenario B (Δ Price %)", -50, 50, 0)
            s3 = c3.number_input("Scenario C (Δ Price %)", -50, 50, 10)
            scenarios = {"A": s1, "B": s2, "C": s3}
            out = []
            base_qty = safe_num(slice_df["sales"].mean())
            base_price = safe_num(slice_df["sell_price"].mean())
            base_rev = base_qty * base_price
            for name, pct in scenarios.items():
                r = simulate_revenue(slice_df, elas_val, pct/100.0)
                out.append({"Scenario": name, "Δ Price %": pct, "New Price": r["new_price"],
                            "Est Units": r["new_qty"], "Revenue Δ%": r["delta_rev_pct"],
                            "Revenue ($ idx)": 1 + r["delta_rev_pct"]/100.0})
            table = pd.DataFrame(out)
            st.dataframe(
                table.style.format({"New Price":"${:.2f}", "Est Units":"{:.1f}", "Revenue Δ%":"{:+.1f}"}),
                use_container_width=True
            )
            fig = px.bar(table, x="Scenario", y="Revenue ($ idx)",
                         text=table["Revenue Δ%"].map(lambda v: f"{v:+.1f}%"),
                         labels={"Revenue ($ idx)":"Revenue index"})
            st.plotly_chart(fig, use_container_width=True)
            best_row = max(out, key=lambda r: r["Revenue ($ idx)"])
            st.success(f"Best of these: Scenario {best_row['Scenario']} "
                       f"({best_row['Δ Price %']:+.0f}%), projected revenue {best_row['Revenue Δ%']:+.1f}%.")

            st.download_button(
                "Download price plan (CSV)",
                data=pd.DataFrame({
                    "store_id":[store_sel],
                    "item_id":[item_sel],
                    "scenario":[best_row["Scenario"]],
                    "recommended_price_change_pct":[best_row["Δ Price %"]]
                }).to_csv(index=False).encode(),
                file_name="price_plan.csv", mime="text/csv"
            )

# ---------- Compare Segments ----------
with tabs[5]:
    st.subheader("Compare Segments")
    with st.expander("Compare Segments", expanded=True):
        st.markdown("Why use it: find which stores, categories, or departments lead or lag. How to use it: pick grouping and metric.")
    dim = st.selectbox("Compare by", ["store_id", "cat_id", "dept_id"], index=0)
    metric = st.selectbox("Metric", ["Total units", "Week-over-week %"], index=0)
    df_range = panel[panel["date"].between(start_ts, end_ts)].copy()
    if df_range.empty:
        st.info("No data in the selected range.")
    else:
        if metric == "Total units":
            agg = df_range.groupby(dim, as_index=False)["sales"].sum().rename(columns={"sales":"total_units"})
            agg = agg.sort_values("total_units", ascending=False).head(15)
            fig = px.bar(agg, x=dim, y="total_units", labels={"total_units":"Units"})
            st.plotly_chart(fig, use_container_width=True)
        else:
            rows = []
            for gval, gdf in df_range.groupby(dim):
                rows.append({"segment": gval, "wow_pct": compute_wow(gdf, freq="W")})
            agg = pd.DataFrame(rows).sort_values("wow_pct", ascending=False).head(15)
            fig = px.bar(agg, x="segment", y="wow_pct", labels={"wow_pct":"WoW %"})
            st.plotly_chart(fig, use_container_width=True)

# ---------- Top Performers ----------
with tabs[6]:
    st.subheader("Top Performers in this Store")
    with st.expander("Top Performers", expanded=True):
        st.markdown("Why use it: spot fast risers and steep decliners inside one store. How to use it: read winners/decliners side by side.")
    srange = panel[(panel["store_id"] == store_sel) & (panel["date"].between(start_ts, end_ts))].copy()
    if srange.empty:
        st.info("No data in the selected range.")
    else:
        movers = []
        for it, g in srange.groupby("item_id"):
            movers.append({"item_id": it, "WoW %": compute_wow(g, "W"), "Total units": g["sales"].sum()})
        mv = pd.DataFrame(movers)
        top_winners = mv.sort_values("WoW %", ascending=False).head(10)
        top_decliners = mv.sort_values("WoW %", ascending=True).head(10)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("Week-over-week increases")
            st.dataframe(top_winners, use_container_width=True)
        with c2:
            st.markdown("Week-over-week declines")
            st.dataframe(top_decliners, use_container_width=True)

# ---------- Summary Export ----------
with tabs[7]:
    st.subheader("Summary Export")
    with st.expander("Summary Export", expanded=True):
        st.markdown("Why use it: share a one-pager and CSV with the key numbers. How to use it: pick your slice and download.")
    if slice_df.empty:
        st.info("No data in the selected range.")
    else:
        k = kpis_for_slice(slice_df)
        elas = estimate_elasticity(slice_df)
        elasticity_text = f"{elas['elasticity']:.2f}" if elas else "N/A"
        direction = "rose" if k["wow_pct"] > 0 else "fell" if k["wow_pct"] < 0 else "held steady"
        if elas:
            interpretation_bit = "Price sensitivity is meaningful; price moves have a noticeable effect." if abs(elas["elasticity"]) > 1 else "Demand is relatively steady against price changes."
        else:
            interpretation_bit = "Not enough history to estimate price sensitivity yet."

        summary_text = f"""
### Sales Snapshot — {store_sel} / {item_sel}

**Period:** {start_ts.date()} → {end_ts.date()}  
**Avg Units / Day:** {k['avg_units']:.1f}  
**Avg Price / Unit:** ${k['avg_price']:.2f}  
**Week-over-Week Change:** {k['wow_pct']:+.1f}%  
**Elasticity:** {elasticity_text}

**Interpretation:**  
Sales {direction} {abs(k['wow_pct']):.1f}% vs the prior week. {interpretation_bit}
"""
        st.markdown(summary_text)

        st.download_button(
            "Download Markdown Summary",
            data=summary_text.encode(),
            file_name=f"summary_{store_sel}_{item_sel}.md",
            mime="text/markdown"
        )

        row = {
            "Store": store_sel,
            "Item": item_sel,
            "Period Start": str(start_ts.date()),
            "Period End": str(end_ts.date()),
            "Avg Units/Day": k["avg_units"],
            "Avg Price/Unit": k["avg_price"],
            "Week-over-Week %": k["wow_pct"],
            "Elasticity": elas["elasticity"] if elas else None
        }
        st.download_button(
            "Download Data (CSV)",
            data=pd.DataFrame([row]).to_csv(index=False).encode(),
            file_name=f"summary_{store_sel}_{item_sel}.csv",
            mime="text/csv"
        )
        st.caption("Attach a price plan from Scenario Compare when sharing recommendations.")