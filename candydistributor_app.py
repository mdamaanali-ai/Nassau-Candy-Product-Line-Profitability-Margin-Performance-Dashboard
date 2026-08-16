# candydistributor_app.py
"""
Nassau Candy Distributor — Product Line Profitability & Margin Performance
Ready-to-run Streamlit app (corrected data-loading and fixed function calls).

Usage:
    1) (Optional) Set DATA_FILE_ID env var to a public Google Drive file id:
       - Linux/macOS: export DATA_FILE_ID=1c4VDb0Pf7RCgps4aLMiSuLtdaUpU_X49
       - Windows PowerShell: $env:DATA_FILE_ID="1c4VDb0Pf7RCgps4aLMiSuLtdaUpU_X49"

    2) Install dependencies:
       pip install -r requirements.txt
       (requirements.txt should include: streamlit, pandas, numpy, plotly, folium, requests)

    3) Run:
       streamlit run candydistributor_app.py
"""
import io
import os
import tempfile
from typing import Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
import folium
import streamlit.components.v1 as components

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(
    page_title="Nassau Candy — Profitability Dashboard",
    page_icon="🍫",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------
# Factory & product mapping (from your input)
# ---------------------------
FACTORIES = {
    "Lot's O' Nuts": {"lat": 32.881893, "lon": -111.768036},
    "Wicked Choccy's": {"lat": 32.076176, "lon": -81.088371},
    "Sugar Shack": {"lat": 48.11914, "lon": -96.18115},
    "Secret Factory": {"lat": 41.446333, "lon": -90.565487},
    "The Other Factory": {"lat": 35.1175, "lon": -89.971107},
}

PRODUCT_FACTORY = {
    "Wonka Bar - Nutty Crunch Surprise": "Lot's O' Nuts",
    "Wonka Bar - Fudge Mallows": "Lot's O' Nuts",
    "Wonka Bar -Scrumdiddlyumptious": "Lot's O' Nuts",
    "Wonka Bar - Milk Chocolate": "Wicked Choccy's",
    "Wonka Bar - Triple Dazzle Caramel": "Wicked Choccy's",
    "Laffy Taffy": "Sugar Shack",
    "SweeTARTS": "Sugar Shack",
    "Nerds": "Sugar Shack",
    "Fun Dip": "Sugar Shack",
    "Fizzy Lifting Drinks": "Sugar Shack",
    "Everlasting Gobstopper": "Secret Factory",
    "Hair Toffee": "The Other Factory",
    "Lickable Wallpaper": "Secret Factory",
    "Wonka Gum": "Secret Factory",
    "Kazookles": "The Other Factory",
}


# ---------------------------
# Helpers
# ---------------------------
def fmt_dollar(x, decimals=0):
    if pd.isna(x):
        return "-"
    return f"${x:,.{decimals}f}"


def download_public_gdrive(file_id: str, dest_path: str) -> None:
    """
    Download a public Google Drive file via the uc?export=download link.
    The file must be shared publicly ("Anyone with the link").
    """
    url = f"https://docs.google.com/uc?export=download&id={file_id}"
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=32768):
                if chunk:
                    f.write(chunk)


@st.cache_data(show_spinner=False)
def load_data_from_path(path: str) -> pd.DataFrame:
    """
    Load CSV or Excel into DataFrame, parsing typical date columns if present.
    """
    try:
        if path.lower().endswith(".csv"):
            return pd.read_csv(path, parse_dates=["Order Date", "Ship Date"], dayfirst=False)
        else:
            return pd.read_excel(path, parse_dates=["Order Date", "Ship Date"])
    except Exception:
        # fallback: read without parse_args, then try to coerce date-like columns
        if path.lower().endswith(".csv"):
            df = pd.read_csv(path)
        else:
            df = pd.read_excel(path)
        for c in df.columns:
            if "order" in c.lower() and "date" in c.lower():
                try:
                    df[c] = pd.to_datetime(df[c], errors="coerce")
                except Exception:
                    pass
        return df


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"[\/\-\s]+", "_", regex=True)
        .str.replace(r"[^0-9a-zA-Z_]", "", regex=True)
    )
    map_candidates = {
        "rowid": "row_id",
        "orderid": "order_id",
        "orderdate": "order_date",
        "shipdate": "ship_date",
        "shipmode": "ship_mode",
        "customerid": "customer_id",
        "countryregion": "country_region",
        "stateprovince": "state_province",
        "postalcode": "postal_code",
        "productid": "product_id",
        "productname": "product_name",
        "grossprofit": "gross_profit",
    }
    rename_map = {c: map_candidates[c] for c in df.columns if c in map_candidates}
    df = df.rename(columns=rename_map)
    return df


def clean_and_validate(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()

    # Parse dates if present
    for dcol in ["order_date", "ship_date"]:
        if dcol in df.columns:
            df[dcol] = pd.to_datetime(df[dcol], errors="coerce")

    # Normalize string columns
    for tcol in ["division", "product_name", "ship_mode", "city", "state_province", "country_region"]:
        if tcol in df.columns:
            df[tcol] = df[tcol].astype(str).str.strip()

    # Clean numeric columns: remove $ and comma
    for ncol in ["sales", "cost", "gross_profit", "units"]:
        if ncol in df.columns:
            df[ncol] = (
                df[ncol]
                .astype(str)
                .str.replace(r"[\$,]", "", regex=True)
                .str.strip()
                .replace({"nan": ""})
            )
            df[ncol] = pd.to_numeric(df[ncol], errors="coerce")

    # Recompute gross_profit where missing or mismatched
    if {"sales", "cost"}.issubset(df.columns):
        gp_calc = df["sales"] - df["cost"]
        if "gross_profit" in df.columns:
            mismatch_mask = df["gross_profit"].isna() | (np.abs(df["gross_profit"] - gp_calc) > 1e-2)
            df.loc[mismatch_mask, "gross_profit"] = gp_calc.loc[mismatch_mask]
            df["gp_mismatch_flag"] = mismatch_mask
        else:
            df["gross_profit"] = gp_calc
            df["gp_mismatch_flag"] = True
    else:
        if "gross_profit" not in df.columns:
            df["gross_profit"] = np.nan
            df["gp_mismatch_flag"] = True
        else:
            df["gp_mismatch_flag"] = False

    # Units: set 0 and negative to NaN (so profit_per_unit is calculated only when meaningful)
    if "units" in df.columns:
        df.loc[df["units"] <= 0, "units"] = np.nan
    else:
        df["units"] = np.nan

    # Drop rows missing critical fields
    critical = [c for c in ("order_date", "product_name", "sales", "gross_profit") if c in df.columns]
    before = len(df)
    df = df.dropna(subset=critical)
    after = len(df)

    # Remove non-positive sales rows (keep a removed sample for report)
    positive_mask = df["sales"] > 0
    removed_nonpositive = df[~positive_mask].copy()
    df = df[positive_mask].copy()

    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    report = pd.DataFrame(
        [
            {"check": "original_rows", "value": before},
            {"check": "rows_after_dropping_missing_critical", "value": after},
            {"check": "removed_nonpositive_sales", "value": len(removed_nonpositive)},
            {"check": "rows_remaining", "value": len(df)},
        ]
    )

    return df, report


def compute_product_metrics(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["product_name"]
    if "product_id" in df.columns:
        group_cols = ["product_name", "product_id"]

    if "order_id" in df.columns:
        agg = (
            df.groupby(group_cols, dropna=False)
            .agg(sales=("sales", "sum"), units=("units", "sum"), gross_profit=("gross_profit", "sum"), cost=("cost", "sum"), orders=("order_id", "nunique"))
            .reset_index()
        )
    else:
        agg = (
            df.groupby(group_cols, dropna=False)
            .agg(sales=("sales", "sum"), units=("units", "sum"), gross_profit=("gross_profit", "sum"), cost=("cost", "sum"), orders=("gross_profit", "count"))
            .reset_index()
        )

    agg["gross_margin_pct"] = np.where(agg["sales"] != 0, agg["gross_profit"] / agg["sales"] * 100, np.nan)
    agg["profit_per_unit"] = np.where(agg["units"].notna() & (agg["units"] != 0), agg["gross_profit"] / agg["units"], np.nan)
    total_sales = agg["sales"].sum() if agg["sales"].sum() != 0 else np.nan
    total_profit = agg["gross_profit"].sum() if agg["gross_profit"].sum() != 0 else np.nan
    agg["revenue_contribution_pct"] = np.where(total_sales and not np.isnan(total_sales), agg["sales"] / total_sales * 100, 0)
    agg["profit_contribution_pct"] = np.where(total_profit and not np.isnan(total_profit), agg["gross_profit"] / total_profit * 100, 0)
    agg["cost_ratio_pct"] = np.where(agg["sales"] != 0, agg["cost"] / agg["sales"] * 100, np.nan)

    # map factories
    def map_factory(pn):
        if pn in PRODUCT_FACTORY:
            return PRODUCT_FACTORY[pn]
        for key in PRODUCT_FACTORY:
            if key.lower() in pn.lower():
                return PRODUCT_FACTORY[key]
        return np.nan

    agg["factory"] = agg["product_name"].apply(map_factory)
    agg["factory_lat"] = agg["factory"].map(lambda f: FACTORIES.get(f, {}).get("lat") if isinstance(f, str) else np.nan)
    agg["factory_lon"] = agg["factory"].map(lambda f: FACTORIES.get(f, {}).get("lon") if isinstance(f, str) else np.nan)

    agg = agg.sort_values("gross_profit", ascending=False).reset_index(drop=True)
    return agg


def compute_division_metrics(df: pd.DataFrame) -> pd.DataFrame:
    agg = (
        df.groupby("division", dropna=False)
        .agg(sales=("sales", "sum"), gross_profit=("gross_profit", "sum"), cost=("cost", "sum"), units=("units", "sum"))
        .reset_index()
    )
    agg["gross_margin_pct"] = np.where(agg["sales"] != 0, agg["gross_profit"] / agg["sales"] * 100, np.nan)
    total_sales = agg["sales"].sum() if agg["sales"].sum() != 0 else np.nan
    total_profit = agg["gross_profit"].sum() if agg["gross_profit"].sum() != 0 else np.nan
    agg["revenue_contribution_pct"] = np.where(total_sales and not np.isnan(total_sales), agg["sales"] / total_sales * 100, 0)
    agg["profit_contribution_pct"] = np.where(total_profit and not np.isnan(total_profit), agg["gross_profit"] / total_profit * 100, 0)
    agg = agg.sort_values("gross_profit", ascending=False).reset_index(drop=True)
    return agg


def pareto_counts(df_agg: pd.DataFrame, value_col: str = "sales", threshold: float = 0.8) -> Tuple[pd.DataFrame, int]:
    df = df_agg.sort_values(value_col, ascending=False).copy().reset_index(drop=True)
    total = df[value_col].sum()
    if total == 0:
        df["cum_pct"] = 0
        return df, 0
    df["cum"] = df[value_col].cumsum()
    df["cum_pct"] = df["cum"] / total
    mask = df["cum_pct"] >= threshold
    count = int(mask.idxmax() + 1) if mask.any() else len(df)
    return df, count


def monthly_margin_volatility(df: pd.DataFrame, product: str = None) -> pd.DataFrame:
    tmp = df.copy()
    if product:
        tmp = tmp[tmp["product_name"] == product]
    tmp["month"] = tmp["order_date"].dt.to_period("M").dt.to_timestamp()
    monthly = tmp.groupby("month").agg(sales=("sales", "sum"), gross_profit=("gross_profit", "sum")).reset_index()
    monthly["gross_margin_pct"] = np.where(monthly["sales"] != 0, monthly["gross_profit"] / monthly["sales"] * 100, np.nan)
    monthly["rolling_margin_std"] = monthly["gross_margin_pct"].rolling(window=3, min_periods=1).std()
    return monthly


# ---------------------------
# UI & Flow
# ---------------------------
st.title("🍫 Nassau Candy Distributor — Product Line Profitability & Margin Performance")
st.markdown("Interactive dashboard to find high/low margin products, division performance, cost risks, Pareto concentration, and factory mapping.")

# Sidebar: dataset source
st.sidebar.header("Data Source")
st.sidebar.write("Upload a dataset or set DATA_FILE_ID env var for automatic Google Drive download (file must be public).")
data_file_id_env = os.environ.get("DATA_FILE_ID", None)

uploaded_file = st.sidebar.file_uploader("Upload dataset (CSV or Excel)", type=["csv", "xlsx", "xls"])

auto_loaded_path = None
if data_file_id_env and uploaded_file is None:
    st.sidebar.write("DATA_FILE_ID detected — attempting to download Google Drive file.")
    try:
        tmpdir = tempfile.mkdtemp()
        tmp_base = os.path.join(tmpdir, "drive_download")
        for ext in [".csv", ".xlsx", ".xls"]:
            candidate = tmp_base + ext
            try:
                download_public_gdrive(data_file_id_env, candidate)
                auto_loaded_path = candidate
                break
            except Exception:
                auto_loaded_path = None
        if auto_loaded_path:
            st.sidebar.success(f"Downloaded dataset to temporary path: {auto_loaded_path}")
        else:
            st.sidebar.error("Failed to download dataset from Google Drive — check DATA_FILE_ID and sharing settings.")
    except Exception as e:
        st.sidebar.error(f"Error downloading Drive file: {e}")
        auto_loaded_path = None

# choose data source: uploaded_file preferred, else auto_loaded_path
df_raw = None

if uploaded_file is not None:
    try:
        file_bytes = uploaded_file.read()
        try:
            df_raw = pd.read_csv(io.BytesIO(file_bytes))
        except Exception:
            df_raw = pd.read_excel(io.BytesIO(file_bytes))
    except Exception as e:
        st.error(f"Error reading uploaded file: {e}")
        st.stop()
elif auto_loaded_path is not None:
    try:
        df_raw = load_data_from_path(auto_loaded_path)
    except Exception as e:
        st.error(f"Failed to load downloaded file: {e}")
        st.stop()
else:
    st.info("No dataset selected yet. Upload a CSV/Excel via the sidebar to begin.")
    st.stop()

# Ensure df_raw is a DataFrame
if df_raw is None:
    st.error("Failed to load dataset (df_raw is None).")
    st.stop()

if not isinstance(df_raw, pd.DataFrame):
    try:
        df_raw = pd.DataFrame(df_raw)
    except Exception as e:
        st.error(f"Loaded dataset is not a DataFrame and cannot be coerced: {e}")
        st.stop()

# Standardize and clean
df_raw = standardize_columns(df_raw)
df, validation_report = clean_and_validate(df_raw)

# Check required columns
required_cols = ["order_date", "division", "product_name", "sales", "gross_profit", "cost"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    st.error(f"Required columns missing after normalization: {missing}. Detected columns: {list(df.columns)}")
    st.stop()

# Sidebar filters
st.sidebar.header("Filters & Settings")
min_date = df["order_date"].min().date()
max_date = df["order_date"].max().date()
date_range = st.sidebar.date_input("Order date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
divisions = sorted(df["division"].dropna().unique())
selected_divisions = st.sidebar.multiselect("Division", options=divisions, default=divisions)
margin_threshold = st.sidebar.slider("Margin Risk Threshold (%)", min_value=-100.0, max_value=100.0, value=20.0, step=1.0)
product_search = st.sidebar.text_input("Search product (substring)", "")

# Apply filters
fdf = df.copy()
if isinstance(date_range, tuple) and len(date_range) == 2:
    start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    fdf = fdf[(fdf["order_date"] >= start) & (fdf["order_date"] <= end)]
if selected_divisions:
    fdf = fdf[fdf["division"].isin(selected_divisions)]
if product_search:
    fdf = fdf[fdf["product_name"].str.contains(product_search, case=False, na=False)]

if fdf.empty:
    st.warning("No records match the selected filters.")
    st.stop()

# Metrics
product_metrics = compute_product_metrics(fdf)
division_metrics = compute_division_metrics(fdf)

total_sales = fdf["sales"].sum()
total_profit = fdf["gross_profit"].sum()
overall_margin = (total_profit / total_sales * 100) if total_sales != 0 else np.nan
total_units = fdf["units"].sum()
profit_per_unit_overall = (total_profit / total_units) if total_units and not np.isnan(total_units) else np.nan

# Executive KPIs
st.markdown("### Executive Summary")
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total Revenue", fmt_dollar(total_sales, 0))
k2.metric("Gross Profit", fmt_dollar(total_profit, 0))
k3.metric("Gross Margin", f"{overall_margin:.2f}%" if not np.isnan(overall_margin) else "N/A")
k4.metric("Profit/Unit (avg)", fmt_dollar(profit_per_unit_overall, 2))
k5.metric("Product count", f"{product_metrics['product_name'].nunique():,}")

# Product Leaderboard
st.markdown("### Product Profitability Leaderboard")
leaderboard = product_metrics.copy()
leaderboard["margin_status"] = np.where(leaderboard["gross_margin_pct"] < margin_threshold, "⚠️ Margin Risk", "✅ Healthy")
display_cols = ["product_name", "sales", "gross_profit", "gross_margin_pct", "profit_per_unit", "cost_ratio_pct", "factory", "margin_status"]
st.dataframe(
    leaderboard[display_cols].rename(
        columns={
            "product_name": "Product",
            "sales": "Sales",
            "gross_profit": "Gross Profit",
            "gross_margin_pct": "Gross Margin (%)",
            "profit_per_unit": "Profit / Unit",
            "cost_ratio_pct": "Cost / Sales (%)",
            "factory": "Factory",
            "margin_status": "Margin Status",
        }
    ),
    use_container_width=True,
    height=360,
)

# Top charts
top_n = st.sidebar.slider("Top N products to chart", 5, 50, 10)
col1, col2 = st.columns(2)
with col1:
    top_profit = product_metrics.sort_values("gross_profit", ascending=False).head(top_n).sort_values("gross_profit")
    fig = px.bar(top_profit, x="gross_profit", y="product_name", orientation="h", labels={"gross_profit": "Gross Profit ($)", "product_name": "Product"}, title=f"Top {top_n} Products by Gross Profit")
    st.plotly_chart(fig, use_container_width=True)
with col2:
    top_margin = product_metrics.dropna(subset=["gross_margin_pct"]).sort_values("gross_margin_pct", ascending=False).head(top_n).sort_values("gross_margin_pct")
    fig2 = px.bar(top_margin, x="gross_margin_pct", y="product_name", orientation="h", labels={"gross_margin_pct": "Gross Margin (%)", "product_name": "Product"}, title=f"Top {top_n} Products by Gross Margin")
    st.plotly_chart(fig2, use_container_width=True)

# Cost vs Margin scatter
st.markdown("### Cost vs Sales (colored by margin)")
fig_scatter = px.scatter(product_metrics, x="cost", y="sales", size="gross_profit", color="gross_margin_pct", hover_name="product_name", color_continuous_scale="RdYlGn", labels={"cost": "Cost ($)", "sales": "Sales ($)"})
fig_scatter.update_layout(height=520)
fig_scatter.add_hline(y=0)
st.plotly_chart(fig_scatter, use_container_width=True)

# Cost-heavy & risk flags
st.markdown("### Cost-Heavy & Margin Risk Products")
cost_risk = product_metrics[(product_metrics["cost_ratio_pct"] > 70) | (product_metrics["gross_margin_pct"] < margin_threshold)].copy()
if cost_risk.empty:
    st.success("No cost-heavy or margin-risk products found under current filters.")
else:
    conditions = [
        cost_risk["gross_margin_pct"] < 0,
        (cost_risk["gross_margin_pct"] < margin_threshold) & (cost_risk["gross_margin_pct"] >= 0),
        cost_risk["cost_ratio_pct"] > 70,
    ]
    choices = ["🔴 Critical", "🟠 High", "🟡 Medium"]
    cost_risk["risk_level"] = np.select(conditions, choices, default="🟢 Low")
    st.dataframe(
        cost_risk[["product_name", "sales", "cost", "gross_profit", "gross_margin_pct", "cost_ratio_pct", "risk_level", "factory"]].rename(
            columns={
                "product_name": "Product",
                "sales": "Sales",
                "cost": "Cost",
                "gross_profit": "Gross Profit",
                "gross_margin_pct": "Gross Margin (%)",
                "cost_ratio_pct": "Cost / Sales (%)",
                "risk_level": "Risk Level",
                "factory": "Factory",
            }
        ),
        use_container_width=True,
    )

# Pareto analysis
st.markdown("### Pareto / Concentration Analysis")
rev_pareto, rev_count = pareto_counts(product_metrics, "sales", 0.8)
profit_pareto, profit_count = pareto_counts(product_metrics, "gross_profit", 0.8)
c1, c2 = st.columns(2)
c1.metric("Products for 80% Revenue", rev_count)
c2.metric("Products for 80% Profit", profit_count)

col1, col2 = st.columns(2)
with col1:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=rev_pareto["product_name"].head(50), y=rev_pareto["sales"].head(50), name="Revenue"))
    fig.add_trace(go.Scatter(x=rev_pareto["product_name"].head(50), y=rev_pareto["cum_pct"].head(50) * 100, name="Cumulative %", yaxis="y2", mode="lines+markers"))
    fig.update_layout(title="Revenue Pareto (top 50)", yaxis=dict(title="Revenue ($)"), yaxis2=dict(title="Cumulative %", overlaying="y", side="right", range=[0, 110]))
    st.plotly_chart(fig, use_container_width=True)
with col2:
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(x=profit_pareto["product_name"].head(50), y=profit_pareto["gross_profit"].head(50), name="Profit"))
    fig2.add_trace(go.Scatter(x=profit_pareto["product_name"].head(50), y=profit_pareto["cum_pct"].head(50) * 100, name="Cumulative %", yaxis="y2", mode="lines+markers"))
    fig2.update_layout(title="Profit Pareto (top 50)", yaxis=dict(title="Gross Profit ($)"), yaxis2=dict(title="Cumulative %", overlaying="y", side="right", range=[0, 110]))
    st.plotly_chart(fig2, use_container_width=True)

# Division analysis
st.markdown("### Division Performance")
fig_div = px.bar(division_metrics.sort_values("sales", ascending=False), x="division", y=["sales", "gross_profit"], barmode="group", title="Revenue vs Gross Profit by Division")
st.plotly_chart(fig_div, use_container_width=True)
fig_div_margin = px.bar(division_metrics.sort_values("gross_margin_pct"), x="gross_margin_pct", y="division", orientation="h", title="Gross Margin by Division", labels={"gross_margin_pct": "Gross Margin (%)"})
st.plotly_chart(fig_div_margin, use_container_width=True)
st.dataframe(division_metrics.rename(columns={"division": "Division", "sales": "Sales", "gross_profit": "Gross Profit", "gross_margin_pct": "Gross Margin (%)"}), use_container_width=True)

# Monthly margin trend & volatility (corrected call)
st.markdown("### Monthly Margin Trend")
monthly = monthly_margin_volatility(fdf)  # correct positional call using filtered dataframe
fig_month = px.line(monthly, x="month", y="gross_margin_pct", title="Monthly Gross Margin", markers=True)
fig_month.add_hline(y=margin_threshold, line_dash="dash", annotation_text="Margin Threshold")
st.plotly_chart(fig_month, use_container_width=True)

# Optional per-product volatility
st.sidebar.header("Product margin volatility")
pv_product = st.sidebar.selectbox("Select product for margin volatility", options=[None] + product_metrics["product_name"].tolist())
if pv_product:
    monthly_prod = monthly_margin_volatility(fdf, pv_product)
    fig_mp = px.line(monthly_prod, x="month", y="gross_margin_pct", title=f"Monthly Margin — {pv_product}", markers=True)
    fig_mp.add_trace(go.Scatter(x=monthly_prod["month"], y=monthly_prod["rolling_margin_std"], mode="lines", name="Rolling std (3mo)", yaxis="y2"))
    fig_mp.update_layout(yaxis2=dict(overlaying="y", side="right", title="Rolling Std"))
    st.plotly_chart(fig_mp, use_container_width=True)

# Factory map (folium) — embedded as HTML (no streamlit_folium dependency required)
st.markdown("### Factory Locations & Product Mapping")
m = folium.Map(location=[39.5, -98.35], zoom_start=4, tiles="CartoDB positron")
for name, coords in FACTORIES.items():
    folium.Marker([coords["lat"], coords["lon"]], popup=name, tooltip=name).add_to(m)

mapped = product_metrics.dropna(subset=["factory"])[["product_name", "factory", "factory_lat", "factory_lon", "sales", "gross_profit"]].copy()
for _, r in mapped.iterrows():
    try:
        if pd.notna(r["factory_lat"]) and pd.notna(r["factory_lon"]):
            folium.CircleMarker([float(r["factory_lat"]), float(r["factory_lon"])], radius=6, popup=f"{r['product_name']} ({fmt_dollar(r['gross_profit'],2)})", color="blue", fill=True).add_to(m)
    except Exception:
        pass

map_html = m.get_root().render()
components.html(map_html, height=360, scrolling=True)

# Automated insights and recommendations
st.markdown("### Automated Insights & Recommendations")
insights = []
if overall_margin < margin_threshold:
    insights.append(f"Overall gross margin ({overall_margin:.2f}%) is below threshold ({margin_threshold}%). Review pricing/sourcing.")
else:
    insights.append(f"Overall gross margin ({overall_margin:.2f}%) is above threshold ({margin_threshold}%).")

if not product_metrics.empty:
    best_profit = product_metrics.loc[product_metrics["gross_profit"].idxmax()]
    insights.append(f"Top profit product: {best_profit['product_name']} — {fmt_dollar(best_profit['gross_profit'],2)} ({best_profit['gross_margin_pct']:.2f}%).")
    tmp_margin = product_metrics.dropna(subset=["gross_margin_pct"])
    if not tmp_margin.empty:
        best_margin = tmp_margin.loc[tmp_margin["gross_margin_pct"].idxmax()]
        worst_margin = tmp_margin.loc[tmp_margin["gross_margin_pct"].idxmin()]
        insights.append(f"Highest margin product: {best_margin['product_name']} ({best_margin['gross_margin_pct']:.2f}%). Lowest margin product: {worst_margin['product_name']} ({worst_margin['gross_margin_pct']:.2f}%).")
if not division_metrics.empty:
    worst_div = division_metrics.loc[division_metrics["gross_margin_pct"].idxmin()]
    insights.append(f"Lowest margin division: {worst_div['division']} ({worst_div['gross_margin_pct']:.2f}%).")

for i in insights:
    st.write("•", i)

# Downloads
st.markdown("### Export Reports")
st.download_button("Download cleaned dataset (CSV)", data=df.to_csv(index=False).encode("utf-8"), file_name="nassau_cleaned_orders.csv", mime="text/csv")
st.download_button("Download product metrics (CSV)", data=product_metrics.to_csv(index=False).encode("utf-8"), file_name="nassau_product_metrics.csv", mime="text/csv")
st.download_button("Download division metrics (CSV)", data=division_metrics.to_csv(index=False).encode("utf-8"), file_name="nassau_division_metrics.csv", mime="text/csv")

# Validation report
st.markdown("---")
st.markdown("Data validation summary:")
st.table(validation_report.set_index("check").T)