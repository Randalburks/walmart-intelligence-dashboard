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

# ---------- data load ----------
try:
    base = load_merge(use_cache=False)
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()
except Exception as e:
    st.error(f"Failed to build dataset: {e}")
    st.stop()

# ---------- sidebar ----------
with st.sidebar:
    st.header("Filters")
    n_stores = st.number_input("How many stores to include", 1, 10, 3, step=1)
    n_items  = st.number_input("How many products per store", 5, 200, 30, step=5)
    st.caption("You can also set env vars LITE_STORES / LITE_ITEMS before launch to shrink data at load.")
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

# ---------- helpers ----------
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

# ---------- tabs ----------
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

# ---------- IDA + Preprocessing ----------
with tabs[0]:
    st.subheader("IDA + Preprocessing")
    st.markdown("This builds a clean daily panel from the original M5 CSVs and shows data health.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows in sampled panel", f"{len(panel):,}")
    c2.metric("Columns", f"{len(panel.columns):,}")
    dmin, dmax = panel["date"].min(), panel["date"].max()
    c3.metric("Date coverage", f"{pd.to_datetime(dmin).date()} → {pd.to_datetime(dmax).date()}")

    st.markdown("#### Structure")
    def _info_table(df_: pd.DataFrame) -> pd.DataFrame:
        nn = df_.notnull().sum()
        out = pd.DataFrame({
            "column": df_.columns,
            "dtype": [str(t) for t in df_.dtypes],
            "non_null": [int(nn[c]) for c in df_.columns],
            "nulls": [int(len(df_) - nn[c]) for c in df_.columns]
        })
        out["null_pct"] = (out["nulls"] / max(1, len(df_))).round(4)
        return out
    st.dataframe(_info_table(panel), use_container_width=True)

    st.markdown("#### Missingness")
    def _miss_bar(df_: pd.DataFrame):
        tmp = df_.isnull().mean().sort_values(ascending=False).reset_index()
        tmp.columns = ["column","null_fraction"]
        fig = px.bar(tmp, x="column", y="null_fraction", title="Missingness by Column")
        fig.update_layout(xaxis_tickangle=-45, yaxis_tickformat=".0%")
        return fig
    def _miss_heat(df_: pd.DataFrame, sample_rows:int=1200):
        smp = df_.sample(min(sample_rows, len(df_)), random_state=42).isnull().astype(int)
        return px.imshow(smp.T, color_continuous_scale="Blues",
                         labels=dict(x="Sampled Row", y="Column", color="Is Null"),
                         title="Missingness Heatmap (sampled rows)")
    cA, cB = st.columns(2)
    cA.plotly_chart(_miss_bar(panel), use_container_width=True)
    cB.plotly_chart(_miss_heat(panel), use_container_width=True)

    st.markdown("#### Behavioral signals")
    cL, cR = st.columns(2)
    ts_all = panel.groupby("date", as_index=False)["sales"].sum()
    cL.plotly_chart(px.line(ts_all, x="date", y="sales", title="All-sampled items — daily sales"),
                    use_container_width=True)
    if "event_type_1" in panel.columns:
        ev = (panel.groupby("event_type_1", as_index=False)["sales"].sum()
              .sort_values("sales", ascending=False).head(12))
        cR.plotly_chart(px.bar(ev, x="event_type_1", y="sales", title="Sales by event type (top 12)"),
                        use_container_width=True)

# ---------- Overview ----------
with tabs[1]:
    st.subheader(f"Overview — {store_sel} / {item_sel}")
    if slice_df.empty:
        st.info("No data in the selected range.")
    else:
        k = kpis_for_slice(slice_df)
        profit_val, margin_val, margin_pct = (None, None, None)
        if "unit_cost" in slice_df.columns:
            profit_val, margin_val, margin_pct = (None, None, None) if slice_df["unit_cost"].isna().all() \
                                                 else (profit_kpis(slice_df))
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Avg Daily Units", f"{k['avg_units']:.1f}")
        c2.metric("Avg Price", f"${k['avg_price']:.2f}")
        c3.metric("Week-over-Week", f"{k['wow_pct']:+.1f}%")
        c4.metric("Days", f"{len(slice_df):,}")
        flagged = anomaly_flags(slice_df, window=7, z=3.0)
        fig = px.line(flagged, x="date", y="sales", labels={"date":"Date","sales":"Units"})
        if flagged["anomaly"].sum() > 0:
            a = flagged[flagged["anomaly"] == 1]
            fig.add_scatter(x=a["date"], y=a["sales"], mode="markers", name="Unusual day")
        st.plotly_chart(add_event_overlays(fig, slice_df), use_container_width=True)

# ---------- Forecast ----------
with tabs[2]:
    st.subheader("Forecast — next 4 weeks")
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
            st.caption("Directional outlook for planning. Actuals vary with price & events.")

# ---------- Price Sensitivity ----------
with tabs[3]:
    st.subheader("Price Sensitivity (elasticity)")
    if slice_df.empty:
        st.info("No data in the selected range.")
    else:
        e = estimate_elasticity(slice_df)
        if not e:
            st.info("Needs ~30+ days with variation in price and sales.")
        else:
            st.success(f"Elasticity: {e['elasticity']:.2f} (n={e['n']}, MAPE={e['mape']:.3f})")
            vals = []
            for (s, i), g in panel.groupby(["store_id","item_id"]):
                ee = estimate_elasticity(g)
                if ee:
                    vals.append(ee["elasticity"])
            if vals:
                h = pd.Series(vals).clip(-5, 5)
                fig = px.histogram(h, nbins=40, labels={"value":"Elasticity","count":"Products"})
                fig.add_vline(x=e["elasticity"], line_dash="dot", line_color="red")
                st.plotly_chart(fig, use_container_width=True)

# ---------- Scenario Compare ----------
with tabs[4]:
    st.subheader("Scenario Compare")
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
            for gval, g in df_range.groupby(dim):
                rows.append({"segment": gval, "wow_pct": compute_wow(g, freq="W")})
            agg = pd.DataFrame(rows).sort_values("wow_pct", ascending=False).head(15)
            fig = px.bar(agg, x="segment", y="wow_pct", labels={"wow_pct":"WoW %"})
            st.plotly_chart(fig, use_container_width=True)

# ---------- Top Performers ----------
with tabs[6]:
    st.subheader("Top Performers in this Store")
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
        c1.markdown("Week-over-week increases"); c1.dataframe(top_winners, use_container_width=True)
        c2.markdown("Week-over-week declines");  c2.dataframe(top_decliners, use_container_width=True)

# ---------- Summary Export ----------
with tabs[7]:
    st.subheader("Summary Export")
    if slice_df.empty:
        st.info("No data in the selected range.")
    else:
        k = kpis_for_slice(slice_df)
        e = estimate_elasticity(slice_df)
        elasticity_text = f"{e['elasticity']:.2f}" if e else "N/A"
        direction = "rose" if k["wow_pct"] > 0 else "fell" if k["wow_pct"] < 0 else "held steady"
        interpretation = ("Price sensitivity is meaningful; price moves have a noticeable effect."
                          if (e and abs(e["elasticity"]) > 1) else
                          "Demand is relatively steady against price changes." if e else
                          "Not enough history to estimate price sensitivity yet.")
        summary_text = f"""
### Sales Snapshot — {store_sel} / {item_sel}

**Period:** {start_ts.date()} → {end_ts.date()}  
**Avg Units / Day:** {k['avg_units']:.1f}  
**Avg Price / Unit:** ${k['avg_price']:.2f}  
**Week-over-Week Change:** {k['wow_pct']:+.1f}%  
**Elasticity:** {elasticity_text}

**Interpretation:**  
Sales {direction} {abs(k['wow_pct']):.1f}% vs the prior week. {interpretation}
"""
        st.markdown(summary_text)
        row = {
            "Store": store_sel,
            "Item": item_sel,
            "Period Start": str(start_ts.date()),
            "Period End": str(end_ts.date()),
            "Avg Units/Day": k["avg_units"],
            "Avg Price/Unit": k["avg_price"],
            "Week-over-Week %": k["wow_pct"],
            "Elasticity": e["elasticity"] if e else None
        }
        st.download_button(
            "Download Data (CSV)",
            data=pd.DataFrame([row]).to_csv(index=False).encode(),
            file_name=f"summary_{store_sel}_{item_sel}.csv",
            mime="text/csv"
        )