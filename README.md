# Walmart Intelligence — Executive Sales, Pricing, Forecasting & Scenario Dashboard

### Michigan State University | CMSE 830 Midterm Project  
**Author:** Randal Burks  
**Purpose:** This project builds an interactive, executive-ready dashboard that summarizes key sales insights from Walmart’s M5 competition dataset. It connects descriptive, diagnostic, and prescriptive analytics — showing what happened, why it happened, and what actions to take.

## Dataset and Source
The raw datasets used for this project are part of the **Walmart M5 Forecasting Competition** on Kaggle:
https://www.kaggle.com/competitions/m5-forecasting-accuracy/data

You’ll need to download the following files and place them in your local `data/` folder:
- `calendar.csv`
- `sell_prices.csv`
- `sales_train_validation.csv`

These files are not included in this repository due to their size.
Link to kaggle dataset: https://www.kaggle.com/competitions/m5-forecasting-accuracy
---

## Project Overview

This dashboard was designed for **executives and decision-makers** who need quick, actionable insights without diving into raw data or code.  
It unifies time-series, pricing, and forecasting analytics in one interactive Streamlit app.

The goal is to make complex data — sales patterns, price sensitivity, and event effects — **visually intuitive** and **strategically meaningful**.

The dashboard:
- Displays historical sales and pricing trends.
- Estimates price elasticity to show how unit demand responds to price changes.
- Compares multiple pricing scenarios side-by-side for revenue planning.
- Highlights top-performing stores, categories, and products.
- Provides exportable summaries for reports or meetings.

---

## Dataset and Source

The data used comes from the **Walmart M5 Forecasting competition** on Kaggle.  
It combines:
- **calendar.csv** — dates, events, and SNAP (Supplemental Nutrition Assistance Program) indicators  
- **sell_prices.csv** — historical price data by store and product  
- **sales_train_validation.csv** — daily sales by store, product, and department  

These datasets are merged, cleaned, and preprocessed before visualization.

> Optional: Upload your own `costs.csv` (columns: `store_id, item_id, unit_cost`) to add margin and profit metrics.

---
walmart-intelligence-dashboard/  

├── app.py  

├── requirements.txt  

├── .gitignore  

├── README.md  

├── .streamlit/  
│   └── secrets.toml  

├── src/  
│   ├── __init__.py  
│   ├── preprocessing.py  
│   └── utils.py  

├── notebooks/  
│   └── eda_ida.ipynb  

├── data/  
│   ├── calendar.csv  
│   ├── sell_prices.csv  
│   ├── sales_train_validation.csv  
│   ├── _merged_m5.parquet  
│   └── costs.csv  
├── exports/  
│   ├── summaries/  
│   └── figures/  
└── docs/  
    └── slides/  
---
## Features

### 1. **Overview**
Shows the overall performance of a selected store and product:
- KPIs for average units, price, and week-over-week change.
- Profit and margin tiles if cost data is provided.
- Time-series chart highlighting anomalies and events.

**Purpose:**  
Provides a quick readout of business health — what’s trending, what’s normal, and what’s unusual.

---

### 2. **Forecast (Next 4 Weeks)**
Uses two lightweight models:
- **Seasonal-naive forecast:** repeats last week’s pattern.
- **Moving average forecast:** smooths recent trends.

**Purpose:**  
Gives short-term directional guidance for planning inventory, staffing, and marketing activities.

---

### 3. **Price Sensitivity**
Estimates price elasticity using log–log regression and visualizes:
- Scatterplots of price vs sales.
- Histograms showing how sensitive this product is relative to others.

**Purpose:**  
Quantifies how pricing changes might affect unit demand and revenue — crucial for promotions and margin strategy.

---

### 4. **Scenario Compare**
Allows users to define three price-change scenarios (e.g., -10%, 0%, +10%) and compare projected revenue.

**Purpose:**  
Lets decision-makers test “what if we discount by 10%?” or “what if we raise prices by 5%?” before taking action.  
Also allows CSV download of the recommended price plan.

---

### 5. **Compare Segments**
Ranks stores, departments, or categories by total units sold or week-over-week % change.

**Purpose:**  
Identifies which locations or departments are leading growth or need attention.

---

### 6. **Top Performers**
Lists products with the highest and lowest week-over-week sales change for the selected store.

**Purpose:**  
Helps managers quickly see which SKUs are driving growth or underperforming without digging through reports.

---

### 7. **Summary Export**
Generates a one-page executive summary with:
- Period metrics
- Elasticity and interpretation
- Week-over-week change summary
- Downloadable CSV and Markdown files for reports or emails

**Purpose:**  
Creates ready-to-share, human-readable summaries for presentations or meetings.

---

## Repository Structure and File Descriptions

### `app.py`
The main Streamlit application.  
It integrates data, analytics, and visualization logic — defining tabs, charts, and metrics displayed in the dashboard.

Key responsibilities:
- Loads and filters data.
- Displays visuals and metrics per tab.
- Handles user input and dynamic updates.
- Generates exportable files (CSV, Markdown summaries).

---

### `src/preprocessing.py`
Handles all **data preparation** steps.

Functions:
- `load_merge(use_cache=True)`:  
  Reads the three M5 datasets, merges them, fills missing prices, derives SNAP and event flags, and saves a tidy `.parquet` cache for performance.

- `sample_panel(df, n_stores, n_items)`:  
  Subsamples the dataset for faster local exploration.

**Why this matters:**  
The dashboard needs consistent, clean data — this script ensures integrity and speed.

---

### `src/utils.py`
Contains **analytical helper functions** used in `app.py`.

Functions:
- `kpis_for_slice(df)` → computes KPIs (avg units, avg price, week-over-week %).  
- `anomaly_flags(df)` → identifies unusually high or low days using rolling z-scores.  
- `seasonal_naive()` and `moving_avg()` → baseline forecasting methods.  
- `safe_num()` → ensures clean numeric calculations (avoids division-by-zero errors).

**Why this matters:**  
Keeps your business logic modular and reusable — you can test or upgrade each component independently.

---

### `notebooks/eda_ida.ipynb`
Jupyter notebook for your **in-depth data exploration** (EDA/IDA).

Includes:
- Data type validation
- Missingness analysis
- Trend and seasonality analysis
- Price–sales relationships
- Event and SNAP effects
- Category, department, and store comparisons
- Outlier detection

**Why this matters:**  
This notebook documents your reasoning for every dashboard design decision — it’s the “why” behind your visuals.

---

### `data/`
Contains all input and cached data files.

- `calendar.csv`
- `sell_prices.csv`
- `sales_train_validation.csv`
- `_merged_m5.parquet` (auto-generated)
- `costs.csv` (optional, for profit calculations)

> You should **not** push raw M5 data to GitHub unless it’s allowed by license. Use `.gitignore` to exclude large files.

---

### `requirements.txt`
Lists all Python dependencies required to run the dashboard.


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
