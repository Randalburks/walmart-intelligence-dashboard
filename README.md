# Walmart Intelligence — Executive Sales, Pricing, Forecasting & Scenarios

This project is an **interactive Streamlit dashboard** designed to give executives, analysts, and decision-makers a **clear and data-driven overview** of sales trends, pricing performance, and demand forecasting.  
It transforms the **Walmart M5 Forecasting Dataset** into an accessible business intelligence tool capable of **anomaly detection, short-term forecasting, elasticity estimation, and pricing scenario simulation** — all without requiring technical expertise.

---

## Project Overview

**Goal:**  
Create an intuitive, executive-ready analytics platform that converts complex retail data into actionable business insights.

**Key Questions the Dashboard Answers**
- How are sales and prices trending week-to-week?
- Which products or stores are performing above or below expectations?
- How sensitive are products to price changes (elasticity)?
- What happens to revenue if prices change by ±10% or more?
- Which stores or departments are driving growth, and which need attention?

**Core Purpose:**  
To help non-technical leaders explore, forecast, and make informed pricing and stocking decisions through clean, self-explanatory visuals and live calculations.

---

## Dataset and Sources

**Dataset:**  
Walmart’s *M5 Forecasting — Accuracy* dataset (Kaggle).  
It includes:
- **calendar.csv** – daily event calendar (SNAP flags, holidays, events)
- **sales_train_validation.csv** – historical daily unit sales by item and store
- **sell_prices.csv** – item-level price data by week and store

**Optional file:**  
- `costs.csv` — custom upload to compute profit and margin metrics.

**Derived variables:**
- `snap` → merged SNAP activity indicator per state/date (marks food-aid active days)
- `event_name_1` / `event_type_1` → overlays major events or holidays on sales charts
- `elasticity` → log-log regression of %∆units vs. %∆price  
- `anomaly` → statistically flagged outlier days

---

## Preprocessing and Data Handling

Data preparation (in `src/preprocessing.py`) performs:
- Merge of calendar, price, and sales files on shared keys.
- Melt of daily columns into long format for time-series analysis.
- Forward/backfill of missing `sell_price` values.
- Derivation of `snap` and event flags.
- Parquet caching for faster reloads.
- Sampling controls to keep the app responsive (adjustable from sidebar).

---

## Features and Purpose of Each Dashboard Tab

### 1. **Overview**
Shows KPIs, daily sales, anomalies, and event overlays.  
**Purpose:** Identify short-term performance patterns and unusual sales behavior.  
**Use:** Hover over spikes/dips to read event labels; margins display if cost data is uploaded.

---

### 2. **Forecast**
Projects the next 28 days using seasonal and moving-average baselines.  
**Purpose:** Give managers a quick directional outlook without heavy modeling.  
**Use:** Select a date range; the chart updates automatically with two forecast lines.

---

### 3. **Price Sensitivity**
Estimates price elasticity (how sales volume reacts to price changes).  
**Purpose:** Determine whether to focus on volume (elastic) or margin (inelastic) strategies.  
**Use:** View product-specific elasticity and compare it with the distribution across items.

---

### 4. **Scenario Compare**
Runs up to three “what-if” simulations for pricing changes.  
**Purpose:** Compare the impact of different price adjustments on revenue in seconds.  
**Use:** Enter price deltas (− for discounts, + for increases); download the recommended plan.

---

### 5. **Compare Segments**
Ranks stores, departments, or categories by total units sold or week-over-week change.  
**Purpose:** Quickly highlight leaders and laggards across the retail footprint.  
**Use:** Switch between grouping dimensions and metrics in the dropdowns.

---

### 6. **Top Performers**
Lists the top 10 growing and declining items in the selected store.  
**Purpose:** Identify which products are driving performance shifts.  
**Use:** Review “Week-over-week increases” vs “declines” tables for category action points.

---

### 7. **Summary Export**
Generates a one-page markdown and CSV summary of the selected store/product.  
**Purpose:** Share a clean, ready-to-email performance summary with team members.  
**Use:** Click **Download Markdown Summary** or **Download Data (CSV)**.

---

## Technical Stack

| Component | Purpose |
|------------|----------|
| **Python 3.9+** | Core language |
| **Streamlit** | Web app framework |
| **Plotly Express** | Interactive charts |
| **pandas / numpy** | Data manipulation |
| **scikit-learn** | Linear regression for elasticity |
| **Statsmodels (optional)** | Statistical tests |
| **Parquet (pyarrow)** | Efficient local caching |

---

## How to Run

```bash
# 1. Clone the repository
git clone https://github.com/your-username/executive-sales-forecasting-and-pricing-dashboard.git
cd executive-sales-forecasting-and-pricing-dashboard

# 2. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your data folder
mkdir data
# Place calendar.csv, sell_prices.csv, sales_train_validation.csv inside /data

# 5. Launch the dashboard
streamlit run app.py
