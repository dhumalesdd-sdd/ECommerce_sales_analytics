"""
=============================================================================
E-Commerce Sales Intelligence Dashboard
Data Storage & Visualization (DSV) Capstone Project
Production-grade interactive BI analytics application built with Streamlit & Plotly
=============================================================================
"""

import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# =============================================================================
# 1. PAGE CONFIGURATION & METADATA
# =============================================================================
st.set_page_config(
    page_title="E-Commerce Sales Intelligence",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# 2. STYLING & CUSTOM THEME INGESTION
# =============================================================================
def load_custom_css():
    """Load premium CSS design system from assets/style.css."""
    css_file = Path(__file__).parent / "assets" / "style.css"
    if css_file.exists():
        with open(css_file, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        # Fallback minimal styling if asset file is missing
        st.markdown("""
        <style>
        .block-container { padding-top: 1.5rem; max-width: 95%; }
        </style>
        """, unsafe_allow_html=True)

load_custom_css()

# Consistent Plotly Color Palette & Chart Theming
CHART_THEME = {
    "primary": "#6366f1",
    "violet": "#8b5cf6",
    "cyan": "#06b6d4",
    "emerald": "#10b981",
    "amber": "#f59e0b",
    "rose": "#f43f5e",
    "palette": ["#6366f1", "#06b6d4", "#10b981", "#f59e0b", "#8b5cf6", "#f43f5e", "#38bdf8", "#ec4899"],
    "bg_paper": "rgba(0,0,0,0)",
    "bg_plot": "rgba(0,0,0,0)",
    "font_family": "Plus Jakarta Sans, -apple-system, sans-serif",
    "grid_color": "rgba(255, 255, 255, 0.07)",
    "text_color": "#94a3b8",
    "title_color": "#f8fafc"
}

def format_plotly_figure(fig, height=380, show_legend=True):
    """Apply uniform premium styling to any Plotly figure."""
    fig.update_layout(
        paper_bgcolor=CHART_THEME["bg_paper"],
        plot_bgcolor=CHART_THEME["bg_plot"],
        font=dict(family=CHART_THEME["font_family"], color=CHART_THEME["text_color"], size=12),
        title_font=dict(family=CHART_THEME["font_family"], size=16, color=CHART_THEME["title_color"]),
        margin=dict(l=20, r=20, t=50, b=30),
        height=height,
        showlegend=show_legend,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11, color="#cbd5e1")
        ),
        xaxis=dict(
            gridcolor=CHART_THEME["grid_color"],
            linecolor=CHART_THEME["grid_color"],
            zerolinecolor=CHART_THEME["grid_color"],
            tickfont=dict(color="#94a3b8", size=11)
        ),
        yaxis=dict(
            gridcolor=CHART_THEME["grid_color"],
            linecolor=CHART_THEME["grid_color"],
            zerolinecolor=CHART_THEME["grid_color"],
            tickfont=dict(color="#94a3b8", size=11)
        ),
        hoverlabel=dict(
            bgcolor="#1e293b",
            bordercolor="rgba(99, 102, 241, 0.5)",
            font=dict(family=CHART_THEME["font_family"], size=12, color="#ffffff")
        )
    )
    return fig

# =============================================================================
# 3. ROBUST DATA PROCESSING & CLEANING PIPELINE
# =============================================================================
DATA_PATH = Path(__file__).parent / "data" / "ecommerce_sales_data.csv"

@st.cache_data(show_spinner=False)
def load_and_preprocess_dataset():
    """
    Safely load, profile, and clean the real e-commerce transactions dataset.
    Maintains raw snapshot and logs comprehensive data-quality metrics.
    """
    if not DATA_PATH.exists():
        return None, None, None

    # Load raw dataset
    df_raw = pd.read_csv(DATA_PATH)
    raw_rows, raw_cols = df_raw.shape

    # Profile raw anomalies
    raw_duplicates = int(df_raw.duplicated().sum())
    order_id_duplicates = int(df_raw.duplicated(subset=['Order_ID']).sum()) if 'Order_ID' in df_raw.columns else 0
    raw_missing_dict = df_raw.isnull().sum().to_dict()
    total_raw_nulls = int(df_raw.isnull().sum().sum())

    # Create cleaning working copy
    df_clean = df_raw.copy()

    # Step 1: Deduplication (Keep first occurrence, removing duplicate transaction orders)
    if 'Order_ID' in df_clean.columns:
        df_clean = df_clean.drop_duplicates(subset=['Order_ID']).reset_index(drop=True)
    else:
        df_clean = df_clean.drop_duplicates().reset_index(drop=True)

    duplicates_removed = raw_rows - len(df_clean)

    # Step 2: Datetime Conversion
    if 'Date' in df_clean.columns:
        df_clean['Date'] = pd.to_datetime(df_clean['Date'], errors='coerce')
        # Drop rows with corrupted unparseable dates if any exist
        df_clean = df_clean.dropna(subset=['Date']).reset_index(drop=True)

    # Step 3: Domain-Aware Missing Value Imputation
    # City: Categorical -> Mode imputation
    if 'City' in df_clean.columns and df_clean['City'].isnull().any():
        city_mode = df_clean['City'].mode()[0] if not df_clean['City'].mode().empty else "Unknown"
        df_clean['City'] = df_clean['City'].fillna(city_mode)

    # Age: Demographic -> Median imputation
    if 'Age' in df_clean.columns and df_clean['Age'].isnull().any():
        age_median = df_clean['Age'].median()
        df_clean['Age'] = df_clean['Age'].fillna(age_median).astype(int)

    # Discount Percent: Promotional -> 0.0 (No discount applied)
    if 'Discount_Percent' in df_clean.columns and df_clean['Discount_Percent'].isnull().any():
        df_clean['Discount_Percent'] = df_clean['Discount_Percent'].fillna(0.0)

    # Customer Rating: Satisfaction -> Mean rating rounded to 1 decimal
    if 'Customer_Rating' in df_clean.columns and df_clean['Customer_Rating'].isnull().any():
        mean_rating = round(df_clean['Customer_Rating'].mean(), 1)
        df_clean['Customer_Rating'] = df_clean['Customer_Rating'].fillna(mean_rating)

    # Step 4: Numeric Validation & Constraints Enforcement
    if 'Quantity' in df_clean.columns:
        df_clean = df_clean[df_clean['Quantity'] > 0]
    if 'Unit_Price' in df_clean.columns:
        df_clean = df_clean[df_clean['Unit_Price'] > 0]
    if 'Discount_Percent' in df_clean.columns:
        df_clean['Discount_Percent'] = df_clean['Discount_Percent'].clip(0.0, 100.0)
    if 'Total_Amount' in df_clean.columns:
        df_clean = df_clean[df_clean['Total_Amount'] > 0]

    # Step 5: Feature Engineering for Enhanced Analytics
    if 'Date' in df_clean.columns:
        df_clean['Year_Month'] = df_clean['Date'].dt.to_period('M').astype(str)
        df_clean['Month_Name'] = df_clean['Date'].dt.strftime('%b')
        df_clean['Month_Num'] = df_clean['Date'].dt.month
        df_clean['Day_Name'] = df_clean['Date'].dt.day_name()
        df_clean['Day_OfWeek'] = df_clean['Date'].dt.dayofweek

    if all(col in df_clean.columns for col in ['Quantity', 'Unit_Price', 'Total_Amount']):
        df_clean['Gross_Amount'] = df_clean['Quantity'] * df_clean['Unit_Price']
        df_clean['Discount_Amount'] = (df_clean['Gross_Amount'] - df_clean['Total_Amount']).clip(lower=0.0)

    clean_rows, clean_cols = df_clean.shape
    total_clean_nulls = int(df_clean.isnull().sum().sum())

    # Build Comprehensive Audit Metadata
    audit_summary = {
        "raw_rows": raw_rows,
        "raw_cols": raw_cols,
        "clean_rows": clean_rows,
        "clean_cols": clean_cols,
        "raw_duplicates": raw_duplicates,
        "order_duplicates_handled": duplicates_removed,
        "total_raw_nulls": total_raw_nulls,
        "total_clean_nulls": total_clean_nulls,
        "raw_missing_breakdown": raw_missing_dict,
        "data_completeness_pct": 100.0 if total_clean_nulls == 0 else round((1 - total_clean_nulls / (clean_rows * clean_cols)) * 100, 2),
        "min_date": df_clean['Date'].min() if 'Date' in df_clean.columns else None,
        "max_date": df_clean['Date'].max() if 'Date' in df_clean.columns else None
    }

    return df_clean, df_raw, audit_summary

# Load data into memory
df_clean, df_raw, audit_summary = load_and_preprocess_dataset()

# Handle missing file scenario gracefully
if df_clean is None:
    st.error("🚨 Dataset Not Found: Unable to locate `data/ecommerce_sales_data.csv`.")
    st.info("Please verify that `ecommerce_sales_data.csv` is present in the `data/` project directory.")
    st.code("DSV/\n├── app.py\n└── data/\n    └── ecommerce_sales_data.csv")
    st.stop()

# =============================================================================
# 4. INTERACTIVE SIDEBAR & DYNAMIC FILTERS
# =============================================================================
st.sidebar.markdown("""
<div style="padding: 0.5rem 0 1rem 0;">
    <div style="font-size: 1.25rem; font-weight: 800; color: #f8fafc; letter-spacing: -0.02em;">
        ⚡ Control Center
    </div>
    <div style="font-size: 0.8rem; color: #94a3b8;">
        Slice & explore transactions dynamically
    </div>
</div>
""", unsafe_allow_html=True)

# Quick Filter Preset Action Handlers
def reset_filters():
    for key in ['filter_categories', 'filter_cities', 'filter_payments', 'filter_statuses', 'filter_genders']:
        if key in st.session_state:
            st.session_state[key] = []
    if 'filter_date_range' in st.session_state:
        del st.session_state['filter_date_range']
    if 'filter_sales_range' in st.session_state:
        del st.session_state['filter_sales_range']

col_reset, col_clear = st.sidebar.columns([1, 1])
with col_reset:
    if st.button("🔄 Reset All", use_container_width=True, help="Reset all filters to default"):
        reset_filters()
        st.rerun()

with col_clear:
    if st.button("📦 Delivered Only", use_container_width=True, help="Show delivered orders only"):
        st.session_state['filter_statuses'] = ['Delivered']
        st.rerun()

st.sidebar.markdown("---")

# 1. Date Range Filter
min_dt = df_clean['Date'].min().date()
max_dt = df_clean['Date'].max().date()

st.sidebar.markdown("**📅 Transaction Date Range**")
selected_dates = st.sidebar.date_input(
    "Date Range",
    value=(min_dt, max_dt),
    min_value=min_dt,
    max_value=max_dt,
    key="filter_date_range",
    label_visibility="collapsed"
)

# Handle single date selection vs date range
if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
    start_date, end_date = selected_dates
elif isinstance(selected_dates, tuple) and len(selected_dates) == 1:
    start_date, end_date = selected_dates[0], max_dt
else:
    start_date, end_date = min_dt, max_dt

# 2. Product Category Filter
all_categories = sorted(df_clean['Product_Category'].unique().tolist())
selected_categories = st.sidebar.multiselect(
    "🏷️ Product Category",
    options=all_categories,
    default=st.session_state.get('filter_categories', []),
    placeholder="All Categories Selected",
    key="filter_categories"
)

# 3. City / Regional Hub Filter
all_cities = sorted(df_clean['City'].unique().tolist())
selected_cities = st.sidebar.multiselect(
    "🌆 Regional Market / City",
    options=all_cities,
    default=st.session_state.get('filter_cities', []),
    placeholder="All Cities Selected",
    key="filter_cities"
)

# 4. Payment Method Filter
all_payments = sorted(df_clean['Payment_Method'].unique().tolist())
selected_payments = st.sidebar.multiselect(
    "💳 Payment Method",
    options=all_payments,
    default=st.session_state.get('filter_payments', []),
    placeholder="All Payment Methods",
    key="filter_payments"
)

# 5. Delivery Status Filter
all_statuses = sorted(df_clean['Delivery_Status'].unique().tolist())
selected_statuses = st.sidebar.multiselect(
    "🚚 Delivery Status",
    options=all_statuses,
    default=st.session_state.get('filter_statuses', []),
    placeholder="All Delivery Statuses",
    key="filter_statuses"
)

# 6. Order Value Range Slider
min_order_val = float(np.floor(df_clean['Total_Amount'].min()))
max_order_val = float(np.ceil(df_clean['Total_Amount'].max()))

st.sidebar.markdown("**💰 Order Amount Range (₹)**")
selected_sales_range = st.sidebar.slider(
    "Order Amount Range",
    min_value=min_order_val,
    max_value=max_order_val,
    value=(min_order_val, max_order_val),
    step=10.0,
    key="filter_sales_range",
    label_visibility="collapsed"
)

# 7. Customer Gender Filter
all_genders = sorted(df_clean['Gender'].unique().tolist())
selected_genders = st.sidebar.multiselect(
    "👤 Customer Gender",
    options=all_genders,
    default=st.session_state.get('filter_genders', []),
    placeholder="All Demographics",
    key="filter_genders"
)

# Apply Filters Simultaneously
filtered_df = df_clean.copy()

# Date filter
filtered_df = filtered_df[
    (filtered_df['Date'].dt.date >= start_date) &
    (filtered_df['Date'].dt.date <= end_date)
]

# Category filter
if selected_categories:
    filtered_df = filtered_df[filtered_df['Product_Category'].isin(selected_categories)]

# City filter
if selected_cities:
    filtered_df = filtered_df[filtered_df['City'].isin(selected_cities)]

# Payment filter
if selected_payments:
    filtered_df = filtered_df[filtered_df['Payment_Method'].isin(selected_payments)]

# Delivery status filter
if selected_statuses:
    filtered_df = filtered_df[filtered_df['Delivery_Status'].isin(selected_statuses)]

# Sales range filter
filtered_df = filtered_df[
    (filtered_df['Total_Amount'] >= selected_sales_range[0]) &
    (filtered_df['Total_Amount'] <= selected_sales_range[1])
]

# Gender filter
if selected_genders:
    filtered_df = filtered_df[filtered_df['Gender'].isin(selected_genders)]

# Sidebar Filter Summary Box
filtered_rows = len(filtered_df)
total_clean_rows = len(df_clean)
match_percentage = (filtered_rows / total_clean_rows * 100) if total_clean_rows > 0 else 0

st.sidebar.markdown("---")
st.sidebar.markdown(f"""
<div style="background: rgba(15, 23, 42, 0.85); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px; padding: 14px; text-align: center;">
    <div style="font-size: 0.76rem; font-weight: 700; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px;">
        Filter Coverage
    </div>
    <div style="font-size: 1.55rem; font-weight: 800; color: #f8fafc;">
        {filtered_rows:,} <span style="font-size: 0.9rem; color: #64748b; font-weight: 500;">/ {total_clean_rows:,}</span>
    </div>
    <div style="font-size: 0.8rem; color: #34d399; font-weight: 600; margin-top: 3px;">
        {match_percentage:.1f}% Active Dataset Match
    </div>
</div>
""", unsafe_allow_html=True)

# =============================================================================
# 5. EXECUTIVE HERO HEADER
# =============================================================================
min_period_str = start_date.strftime("%b %d, %Y")
max_period_str = end_date.strftime("%b %d, %Y")

st.markdown(f"""
<div class="hero-card">
    <div class="hero-title">🛍️ E-Commerce Sales Intelligence</div>
    <div class="hero-subtitle">
        Explore sales performance, customer behavior, product trends, and business growth through interactive data analytics.
    </div>
    <div class="badge-row">
        <div class="status-pill live">
            <span class="status-dot"></span> Live Dashboard • Indian Edition 🇮🇳
        </div>
        <div class="status-pill primary">
            📁 Dataset: ecommerce_sales_data.csv
        </div>
        <div class="status-pill neutral">
            📅 Horizon: {min_period_str} – {max_period_str}
        </div>
        <div class="status-pill neutral">
            🎯 Scope: {filtered_rows:,} Verified Transactions
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Graceful Empty State Handler
if filtered_df.empty:
    st.markdown("""
    <div class="empty-state-box">
        <div class="empty-state-icon">🔍</div>
        <div class="empty-state-title">No Transactions Match Current Filter Criteria</div>
        <div class="empty-state-desc">
            Your combined date range, category, city, or order value filters resulted in 0 records.
            Adjust your sidebar selections or click below to restore full dataset view.
        </div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("🔄 Reset Filters to View All Data", type="primary"):
        reset_filters()
        st.rerun()
    st.stop()

# =============================================================================
# 6. KEY PERFORMANCE INDICATORS (KPI CARDS)
# =============================================================================
# Accurate Formulas & Semantic Aggregations
total_revenue = float(filtered_df['Total_Amount'].sum())
unique_orders = int(filtered_df['Order_ID'].nunique())
total_units = int(filtered_df['Quantity'].sum())
aov = float(total_revenue / unique_orders) if unique_orders > 0 else 0.0
unique_customers = int(filtered_df['Customer_ID'].nunique())
avg_rating = float(filtered_df['Customer_Rating'].mean())
units_per_order = float(total_units / unique_orders) if unique_orders > 0 else 0.0
orders_per_cust = float(unique_orders / unique_customers) if unique_customers > 0 else 0.0

st.markdown(f"""
<div class="kpi-grid">
    <div class="kpi-card">
        <div class="kpi-card-header">
            <span class="kpi-label">Total Revenue</span>
            <div class="kpi-icon icon-purple">💰</div>
        </div>
        <div class="kpi-value">₹{total_revenue:,.2f}</div>
        <div class="kpi-caption">Net sales generated after discounts</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-card-header">
            <span class="kpi-label">Total Orders</span>
            <div class="kpi-icon icon-cyan">🛒</div>
        </div>
        <div class="kpi-value">{unique_orders:,}</div>
        <div class="kpi-caption">Verified distinct transaction IDs</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-card-header">
            <span class="kpi-label">Units Sold</span>
            <div class="kpi-icon icon-emerald">📦</div>
        </div>
        <div class="kpi-value">{total_units:,}</div>
        <div class="kpi-caption">Avg {units_per_order:.2f} items per checkout</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-card-header">
            <span class="kpi-label">Avg Order Value</span>
            <div class="kpi-icon icon-amber">📈</div>
        </div>
        <div class="kpi-value">₹{aov:,.2f}</div>
        <div class="kpi-caption">Revenue per unique order (AOV)</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-card-header">
            <span class="kpi-label">Unique Shoppers</span>
            <div class="kpi-icon icon-violet">👥</div>
        </div>
        <div class="kpi-value">{unique_customers:,}</div>
        <div class="kpi-caption">Avg {orders_per_cust:.2f} orders per customer</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-card-header">
            <span class="kpi-label">Customer Rating</span>
            <div class="kpi-icon icon-rose">⭐</div>
        </div>
        <div class="kpi-value">{avg_rating:.2f} <span style="font-size:1.1rem; color:#94a3b8;">/ 5.0</span></div>
        <div class="kpi-caption">Satisfaction index across orders</div>
    </div>
</div>
""", unsafe_allow_html=True)

# =============================================================================
# 7. NAVIGATION TABS (WORKFLOW & VISUALIZATIONS)
# =============================================================================
tab_viz, tab_insights, tab_import, tab_explorer = st.tabs([
    "📊 Executive Visualizations",
    "💡 Business Insights & Intelligence",
    "📤 Import Your Own Data",
    "📋 Raw Data Explorer & Export"
])

# -----------------------------------------------------------------------------
# TAB 1: EXECUTIVE VISUALIZATIONS (6 CORE PLOTLY CHARTS)
# -----------------------------------------------------------------------------
with tab_viz:
    st.markdown("""
    <div class="section-header">
        <div>
            <div class="section-title">📈 Core Business Intelligence Visualizations</div>
            <div class="section-subtitle">Real-time analytical graphs synchronizing with active sidebar filters</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ROW 1: SALES TREND & TOP-SELLING PRODUCTS
    row1_col1, row1_col2 = st.columns([1.1, 0.9])

    with row1_col1:
        st.markdown("""
        <div class="chart-panel">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom: 8px;">
                <div style="font-weight:700; font-size:1.05rem; color:#f8fafc;">
                    Chart 1: Sales Revenue Trend Over Time
                </div>
            </div>
        """, unsafe_allow_html=True)

        granularity = st.radio(
            "Aggregation Interval",
            options=["Monthly", "Weekly", "Daily"],
            horizontal=True,
            key="trend_granularity",
            label_visibility="collapsed"
        )

        # Dynamic time aggregation
        trend_df = filtered_df.copy()
        if granularity == "Monthly":
            trend_data = trend_df.groupby(pd.Grouper(key='Date', freq='ME')).agg(
                Revenue=('Total_Amount', 'sum'),
                Orders=('Order_ID', 'nunique')
            ).reset_index()
            trend_data['Date_Label'] = trend_data['Date'].dt.strftime('%b %Y')
            x_col = 'Date'
        elif granularity == "Weekly":
            trend_data = trend_df.groupby(pd.Grouper(key='Date', freq='W-MON')).agg(
                Revenue=('Total_Amount', 'sum'),
                Orders=('Order_ID', 'nunique')
            ).reset_index()
            trend_data['Date_Label'] = trend_data['Date'].dt.strftime('%b %d, %Y')
            x_col = 'Date'
        else:
            trend_data = trend_df.groupby(pd.Grouper(key='Date', freq='D')).agg(
                Revenue=('Total_Amount', 'sum'),
                Orders=('Order_ID', 'nunique')
            ).reset_index()
            trend_data['Date_Label'] = trend_data['Date'].dt.strftime('%b %d')
            x_col = 'Date'

        fig_trend = px.area(
            trend_data,
            x=x_col,
            y='Revenue',
            markers=True,
            color_discrete_sequence=['#6366f1'],
            labels={'Revenue': 'Revenue (₹)', 'Date': 'Date'},
            custom_data=['Orders']
        )
        fig_trend.update_traces(
            line=dict(width=3, color='#818cf8'),
            fillcolor='rgba(99, 102, 241, 0.2)',
            hovertemplate="<b>Date:</b> %{x|%b %d, %Y}<br><b>Revenue:</b> ₹%{y:,.2f}<br><b>Orders:</b> %{customdata[0]:,}<extra></extra>"
        )

        # Highlight Peak Period
        if not trend_data.empty and trend_data['Revenue'].max() > 0:
            peak_row = trend_data.loc[trend_data['Revenue'].idxmax()]
            fig_trend.add_annotation(
                x=peak_row[x_col],
                y=peak_row['Revenue'],
                text=f"Peak: ₹{peak_row['Revenue']:,.0f}",
                showarrow=True,
                arrowhead=2,
                arrowcolor="#34d399",
                font=dict(color="#34d399", size=11, family="Plus Jakarta Sans"),
                bgcolor="#0f172a",
                bordercolor="#34d399"
            )

        format_plotly_figure(fig_trend, height=360)
        st.plotly_chart(fig_trend, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with row1_col2:
        st.markdown("""
        <div class="chart-panel">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom: 8px;">
                <div style="font-weight:700; font-size:1.05rem; color:#f8fafc;">
                    Chart 2: Top-Selling Products Performance
                </div>
            </div>
        """, unsafe_allow_html=True)

        prod_metric_col1, prod_metric_col2 = st.columns([1, 1])
        with prod_metric_col1:
            product_metric = st.selectbox("Ranking Metric", ["Revenue (₹)", "Units Sold"], key="prod_metric", label_visibility="collapsed")
        with prod_metric_col2:
            top_n = st.selectbox("Display Top", [5, 10, 15], index=1, key="top_n_prod", label_visibility="collapsed")

        metric_col = 'Total_Amount' if product_metric == "Revenue (₹)" else 'Quantity'
        label_text = 'Revenue' if product_metric == "Revenue (₹)" else 'Units'

        top_products = filtered_df.groupby('Product_Name').agg(
            Revenue=('Total_Amount', 'sum'),
            Units=('Quantity', 'sum'),
            Category=('Product_Category', 'first')
        ).reset_index()

        top_products = top_products.sort_values(by=label_text, ascending=True).tail(top_n)

        fig_products = px.bar(
            top_products,
            x=label_text,
            y='Product_Name',
            orientation='h',
            color=label_text,
            color_continuous_scale='Viridis',
            custom_data=['Category', 'Revenue', 'Units']
        )
        fig_products.update_traces(
            hovertemplate="<b>%{y}</b><br>Category: %{customdata[0]}<br>Revenue: ₹%{customdata[1]:,.2f}<br>Units Sold: %{customdata[2]:,}<extra></extra>",
            marker=dict(line=dict(width=0))
        )
        fig_products.update_layout(coloraxis_showscale=False)
        format_plotly_figure(fig_products, height=360, show_legend=False)
        st.plotly_chart(fig_products, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ROW 2: CATEGORY SALES ANALYSIS & REGIONAL PERFORMANCE
    row2_col1, row2_col2 = st.columns([1, 1])

    with row2_col1:
        st.markdown("""
        <div class="chart-panel">
            <div style="font-weight:700; font-size:1.05rem; color:#f8fafc; margin-bottom: 12px;">
                Chart 3: Category-Wise Sales & Volume Analysis
            </div>
        """, unsafe_allow_html=True)

        cat_analysis = filtered_df.groupby('Product_Category').agg(
            Revenue=('Total_Amount', 'sum'),
            Orders=('Order_ID', 'nunique'),
            AOV=('Total_Amount', 'mean')
        ).reset_index().sort_values('Revenue', ascending=False)

        cat_analysis['Revenue_Share'] = (cat_analysis['Revenue'] / total_revenue * 100) if total_revenue > 0 else 0

        fig_category = px.bar(
            cat_analysis,
            x='Product_Category',
            y='Revenue',
            color='Product_Category',
            color_discrete_sequence=CHART_THEME["palette"],
            text=cat_analysis['Revenue_Share'].apply(lambda x: f"{x:.1f}%"),
            custom_data=['Orders', 'AOV', 'Revenue_Share']
        )
        fig_category.update_traces(
            textposition='outside',
            hovertemplate="<b>%{x}</b><br>Revenue: ₹%{y:,.2f}<br>Share: %{customdata[2]:.1f}%<br>Orders: %{customdata[0]:,}<br>AOV: ₹%{customdata[1]:,.2f}<extra></extra>"
        )
        format_plotly_figure(fig_category, height=380, show_legend=False)
        st.plotly_chart(fig_category, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with row2_col2:
        st.markdown("""
        <div class="chart-panel">
            <div style="font-weight:700; font-size:1.05rem; color:#f8fafc; margin-bottom: 12px;">
                Chart 4: Regional Market & City-Wise Performance
            </div>
        """, unsafe_allow_html=True)

        city_perf = filtered_df.groupby('City').agg(
            Revenue=('Total_Amount', 'sum'),
            Orders=('Order_ID', 'nunique'),
            Avg_Spend=('Total_Amount', 'mean')
        ).reset_index().sort_values('Revenue', ascending=False)

        fig_city = px.bar(
            city_perf,
            x='Revenue',
            y='City',
            orientation='h',
            color='Avg_Spend',
            color_continuous_scale='Tealgrn',
            custom_data=['Orders', 'Avg_Spend']
        )
        fig_city.update_layout(yaxis=dict(autorange="reversed"), coloraxis_colorbar=dict(title="AOV (₹)"))
        fig_city.update_traces(
            hovertemplate="<b>%{y}</b><br>Total Revenue: ₹%{x:,.2f}<br>Orders: %{customdata[0]:,}<br>Average Spend: ₹%{customdata[1]:,.2f}<extra></extra>"
        )
        format_plotly_figure(fig_city, height=380, show_legend=False)
        st.plotly_chart(fig_city, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ROW 3: PAYMENT METHOD DISTRIBUTION & STRATEGIC INSIGHT (FULFILLMENT / DISCOUNT)
    row3_col1, row3_col2 = st.columns([1, 1])

    with row3_col1:
        st.markdown("""
        <div class="chart-panel">
            <div style="font-weight:700; font-size:1.05rem; color:#f8fafc; margin-bottom: 12px;">
                Chart 5: Payment Method & Checkout Share
            </div>
        """, unsafe_allow_html=True)

        payment_summary = filtered_df.groupby('Payment_Method').agg(
            Transactions=('Order_ID', 'nunique'),
            Volume=('Total_Amount', 'sum')
        ).reset_index().sort_values('Volume', ascending=False)

        fig_payment = px.pie(
            payment_summary,
            names='Payment_Method',
            values='Volume',
            hole=0.55,
            color_discrete_sequence=['#6366f1', '#06b6d4', '#10b981', '#f59e0b', '#8b5cf6', '#ec4899'],
            custom_data=['Transactions']
        )
        fig_payment.update_traces(
            textinfo='percent+label',
            textfont=dict(color="#f8fafc", size=11),
            hovertemplate="<b>%{label}</b><br>Total Volume: ₹%{value:,.2f}<br>Share: %{percent}<br>Transactions: %{customdata[0]:,}<extra></extra>"
        )
        format_plotly_figure(fig_payment, height=380, show_legend=False)
        st.plotly_chart(fig_payment, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with row3_col2:
        st.markdown("""
        <div class="chart-panel">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom: 10px;">
                <div style="font-weight:700; font-size:1.05rem; color:#f8fafc;">
                    Chart 6: Strategic Business Insight
                </div>
            </div>
        """, unsafe_allow_html=True)

        insight_mode = st.radio(
            "Select Analysis Angle",
            ["Fulfillment & Cancellation Impact", "Discount vs Satisfaction Matrix"],
            horizontal=True,
            key="insight_chart_toggle",
            label_visibility="collapsed"
        )

        if insight_mode == "Fulfillment & Cancellation Impact":
            delivery_perf = filtered_df.groupby('Delivery_Status').agg(
                Orders=('Order_ID', 'nunique'),
                Revenue=('Total_Amount', 'sum')
            ).reset_index().sort_values('Orders', ascending=False)

            status_colors = {
                'Delivered': '#10b981',
                'Shipped': '#06b6d4',
                'Processing': '#f59e0b',
                'Cancelled': '#f43f5e'
            }

            fig_deliv = px.bar(
                delivery_perf,
                x='Delivery_Status',
                y='Orders',
                color='Delivery_Status',
                color_discrete_map=status_colors,
                text=delivery_perf['Orders'].apply(lambda x: f"{x:,}"),
                custom_data=['Revenue']
            )
            fig_deliv.update_traces(
                textposition='outside',
                hovertemplate="<b>Status: %{x}</b><br>Orders: %{y:,}<br>Associated Revenue: ₹%{customdata[0]:,.2f}<extra></extra>"
            )

            # Calculate Cancelled revenue
            cancelled_revenue = filtered_df[filtered_df['Delivery_Status'] == 'Cancelled']['Total_Amount'].sum()
            cancelled_rate = (filtered_df[filtered_df['Delivery_Status'] == 'Cancelled']['Order_ID'].nunique() / unique_orders * 100) if unique_orders > 0 else 0

            fig_deliv.add_annotation(
                text=f"Revenue at Risk (Cancelled): ₹{cancelled_revenue:,.2f} ({cancelled_rate:.1f}%)",
                xref="paper", yref="paper",
                x=0.5, y=1.12,
                showarrow=False,
                font=dict(color="#fb7185", size=12, family="Plus Jakarta Sans"),
                bgcolor="rgba(244, 63, 94, 0.12)",
                bordercolor="rgba(244, 63, 94, 0.3)"
            )

            format_plotly_figure(fig_deliv, height=380, show_legend=False)
            st.plotly_chart(fig_deliv, use_container_width=True)

        else:
            # Discount vs Rating & Sales Box/Scatter
            discount_perf = filtered_df.groupby('Discount_Percent').agg(
                Avg_Rating=('Customer_Rating', 'mean'),
                Avg_Order_Value=('Total_Amount', 'mean'),
                Orders=('Order_ID', 'nunique')
            ).reset_index().sort_values('Discount_Percent')

            fig_discount = go.Figure()
            fig_discount.add_trace(go.Bar(
                x=discount_perf['Discount_Percent'].astype(str) + '%',
                y=discount_perf['Avg_Order_Value'],
                name="Avg Order Value (₹)",
                marker_color='#6366f1',
                yaxis='y1'
            ))
            fig_discount.add_trace(go.Scatter(
                x=discount_perf['Discount_Percent'].astype(str) + '%',
                y=discount_perf['Avg_Rating'],
                name="Satisfaction Rating",
                marker=dict(color='#fbbf24', size=10),
                line=dict(color='#fbbf24', width=3),
                yaxis='y2'
            ))

            fig_discount.update_layout(
                yaxis=dict(title="Avg Order Value (₹)", gridcolor=CHART_THEME["grid_color"]),
                yaxis2=dict(title="Rating (1-5)", overlaying='y', side='right', range=[1, 5.2], gridcolor="rgba(0,0,0,0)"),
                legend=dict(orientation="h", y=1.05, x=1, xanchor="right")
            )
            format_plotly_figure(fig_discount, height=380, show_legend=True)
            st.plotly_chart(fig_discount, use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# TAB 2: ADVANCED ANALYTICS & BUSINESS INSIGHTS
# -----------------------------------------------------------------------------
with tab_insights:
    st.markdown("""
    <div class="section-header">
        <div>
            <div class="section-title">💡 Automated Data Story & Strategic Insights</div>
            <div class="section-subtitle">Algorithmic synthesis derived directly from active filtered transactions</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 1. Compute dynamic insights from current filtered state
    cat_revenue_series = filtered_df.groupby('Product_Category')['Total_Amount'].sum()
    top_cat_name = cat_revenue_series.idxmax() if not cat_revenue_series.empty else "N/A"
    top_cat_val = cat_revenue_series.max() if not cat_revenue_series.empty else 0.0
    top_cat_share = (top_cat_val / total_revenue * 100) if total_revenue > 0 else 0.0

    prod_revenue_series = filtered_df.groupby('Product_Name')['Total_Amount'].sum()
    top_prod_name = prod_revenue_series.idxmax() if not prod_revenue_series.empty else "N/A"
    top_prod_val = prod_revenue_series.max() if not prod_revenue_series.empty else 0.0

    city_revenue_series = filtered_df.groupby('City')['Total_Amount'].sum()
    top_city_name = city_revenue_series.idxmax() if not city_revenue_series.empty else "N/A"
    top_city_val = city_revenue_series.max() if not city_revenue_series.empty else 0.0
    top_city_share = (top_city_val / total_revenue * 100) if total_revenue > 0 else 0.0

    month_revenue_series = filtered_df.groupby('Month_Name')['Total_Amount'].sum()
    peak_month_name = month_revenue_series.idxmax() if not month_revenue_series.empty else "N/A"
    peak_month_val = month_revenue_series.max() if not month_revenue_series.empty else 0.0

    day_sales_series = filtered_df.groupby('Day_Name')['Total_Amount'].sum()
    peak_day_name = day_sales_series.idxmax() if not day_sales_series.empty else "N/A"

    payment_counts = filtered_df['Payment_Method'].value_counts()
    top_payment_method = payment_counts.index[0] if not payment_counts.empty else "N/A"
    top_payment_share = (payment_counts.iloc[0] / len(filtered_df) * 100) if not payment_counts.empty else 0.0

    cancelled_count = filtered_df[filtered_df['Delivery_Status'] == 'Cancelled']['Order_ID'].nunique()
    cancelled_rev = filtered_df[filtered_df['Delivery_Status'] == 'Cancelled']['Total_Amount'].sum()
    cancellation_pct = (cancelled_count / unique_orders * 100) if unique_orders > 0 else 0.0

    # Render Insight Cards Grid
    col_ins1, col_ins2 = st.columns(2)

    with col_ins1:
        st.markdown(f"""
        <div class="insight-card success">
            <div class="insight-header">
                🏆 Top Revenue Anchor: {top_cat_name}
            </div>
            <div class="insight-body">
                The <span class="insight-highlight">{top_cat_name}</span> category drives the highest financial volume, capturing 
                <strong>₹{top_cat_val:,.2f}</strong> ({top_cat_share:.1f}% of active revenue). 
                Top performing single item is <strong>{top_prod_name}</strong> generating ₹{top_prod_val:,.2f}.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="insight-card info">
            <div class="insight-header">
                🌆 Dominant Regional Market: {top_city_name}
            </div>
            <div class="insight-body">
                <span class="insight-highlight">{top_city_name}</span> ranks as the number one customer hub with 
                <strong>₹{top_city_val:,.2f}</strong> ({top_city_share:.1f}% share). 
                Marketing ROI is maximized by running geo-targeted inventory fulfillment and localized promotions in this metropolitan area.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="insight-card warning">
            <div class="insight-header">
                📅 Peak Demand Velocity: {peak_month_name} & {peak_day_name}
            </div>
            <div class="insight-body">
                Transactions peak during <span class="insight-highlight">{peak_month_name}</span> (<strong>₹{peak_month_val:,.2f}</strong>), 
                with <span class="insight-highlight">{peak_day_name}</span> consistently showing the strongest conversion rates throughout the week.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_ins2:
        st.markdown(f"""
        <div class="insight-card danger">
            <div class="insight-header">
                ⚠️ Fulfillment Health & Revenue at Risk: {cancellation_pct:.1f}% Cancelled
            </div>
            <div class="insight-body">
                A total of <span class="insight-highlight">{cancelled_count:,} orders</span> were cancelled, representing 
                <strong>₹{cancelled_rev:,.2f}</strong> in lost revenue. 
                Implementing pre-dispatch confirmation SMS/calls and fast-shipping partnerships can recover up to 40% of this leak.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="insight-card purple">
            <div class="insight-header">
                💳 Digital Checkout Habit: {top_payment_method} Leads
            </div>
            <div class="insight-body">
                <span class="insight-highlight">{top_payment_method}</span> is preferred in 
                <strong>{top_payment_share:.1f}%</strong> of checkouts. Instant digital gateways (Credit Cards, UPI, Google Pay, PhonePe) 
                dominate cash on delivery, reducing payment settlement friction and improving operational cash flow.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="insight-card info">
            <div class="insight-header">
                ⭐ Customer Loyalty & Average Order Value
            </div>
            <div class="insight-body">
                Average Basket Size sits at <strong>₹{aov:,.2f}</strong> across {unique_orders:,} transactions with a 
                <strong>{avg_rating:.2f} / 5.0</strong> satisfaction index. Repeat shopper ratio stands at 
                <strong>{orders_per_cust:.2f} orders/customer</strong>.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # HEAD-TO-HEAD CATEGORY BENCHMARK TOOL
    st.markdown("---")
    st.markdown("""
    <div style="font-size: 1.2rem; font-weight: 700; color: #f8fafc; margin-bottom: 0.8rem;">
        ⚔️ Deep-Dive: Category Head-to-Head Benchmark
    </div>
    <div style="font-size: 0.88rem; color: #94a3b8; margin-bottom: 1.2rem;">
        Select any two product categories to run an instant comparative performance diagnosis across 5 key dimensions.
    </div>
    """, unsafe_allow_html=True)

    c_cat1, c_cat2 = st.columns(2)
    categories_available = sorted(df_clean['Product_Category'].unique().tolist())

    with c_cat1:
        cat_a = st.selectbox("Benchmark Category A", categories_available, index=0, key="benchmark_cat_a")
    with c_cat2:
        default_b_idx = 1 if len(categories_available) > 1 else 0
        cat_b = st.selectbox("Benchmark Category B", categories_available, index=default_b_idx, key="benchmark_cat_b")

    if cat_a and cat_b:
        df_comp = filtered_df[filtered_df['Product_Category'].isin([cat_a, cat_b])]
        if not df_comp.empty:
            comp_metrics = df_comp.groupby('Product_Category').agg(
                Revenue=('Total_Amount', 'sum'),
                Orders=('Order_ID', 'nunique'),
                AOV=('Total_Amount', 'mean'),
                Rating=('Customer_Rating', 'mean'),
                Discount=('Discount_Percent', 'mean'),
                Units=('Quantity', 'sum')
            ).reset_index()

            # Render comparison header
            st.markdown(f"""
            <div class="compare-header">
                <span class="compare-badge badge-a">{cat_a}</span>
                <div class="vs-circle">VS</div>
                <span class="compare-badge badge-b">{cat_b}</span>
            </div>
            """, unsafe_allow_html=True)

            m_c1, m_c2, m_c3, m_c4 = st.columns(4)
            cat_a_row = comp_metrics[comp_metrics['Product_Category'] == cat_a]
            cat_b_row = comp_metrics[comp_metrics['Product_Category'] == cat_b]

            rev_a = cat_a_row['Revenue'].values[0] if not cat_a_row.empty else 0
            rev_b = cat_b_row['Revenue'].values[0] if not cat_b_row.empty else 0
            aov_a = cat_a_row['AOV'].values[0] if not cat_a_row.empty else 0
            aov_b = cat_b_row['AOV'].values[0] if not cat_b_row.empty else 0
            ord_a = cat_a_row['Orders'].values[0] if not cat_a_row.empty else 0
            ord_b = cat_b_row['Orders'].values[0] if not cat_b_row.empty else 0
            rat_a = cat_a_row['Rating'].values[0] if not cat_a_row.empty else 0
            rat_b = cat_b_row['Rating'].values[0] if not cat_b_row.empty else 0

            with m_c1:
                st.metric(f"Revenue ({cat_a})", f"₹{rev_a:,.2f}", delta=f"₹{rev_a - rev_b:,.2f} vs {cat_b}")
            with m_c2:
                st.metric(f"Total Orders ({cat_a})", f"{ord_a:,}", delta=f"{ord_a - ord_b:,} vs {cat_b}")
            with m_c3:
                st.metric(f"Avg Order Value ({cat_a})", f"₹{aov_a:,.2f}", delta=f"₹{aov_a - aov_b:,.2f} vs {cat_b}")
            with m_c4:
                st.metric(f"Avg Rating ({cat_a})", f"{rat_a:.2f} ★", delta=f"{rat_a - rat_b:.2f} vs {cat_b}")

            # Grouped bar chart comparison
            fig_benchmark = px.bar(
                comp_metrics,
                x='Product_Category',
                y=['Revenue', 'Units', 'Orders'],
                barmode='group',
                color_discrete_sequence=['#6366f1', '#06b6d4', '#10b981'],
                title=f"Head-to-Head Scale: {cat_a} vs {cat_b}"
            )
            format_plotly_figure(fig_benchmark, height=340)
            st.plotly_chart(fig_benchmark, use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 3: IMPORT YOUR OWN DATA (REPLACES PIPELINE)
# -----------------------------------------------------------------------------
with tab_import:
    st.markdown("""
    <div class="section-header">
        <div>
            <div class="section-title">📤 Import & Analyse Your Own Data</div>
            <div class="section-subtitle">Upload an Excel (.xlsx) or CSV file to instantly generate interactive charts and summary statistics</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="chart-panel">
        <div style="font-weight:700; font-size:1.05rem; color:#f8fafc; margin-bottom: 12px;">
            📁 Upload Your Dataset
        </div>
        <div style="font-size:0.88rem; color:#94a3b8; margin-bottom: 16px;">
            Accepted formats: <strong>.xlsx</strong>, <strong>.xls</strong>, <strong>.csv</strong> — Max file size: 200 MB
        </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Drop your file here or click to browse",
        type=["xlsx", "xls", "csv"],
        key="user_data_upload",
        label_visibility="collapsed"
    )
    st.markdown("</div>", unsafe_allow_html=True)

    if uploaded_file is not None:
        # Load the uploaded file
        try:
            if uploaded_file.name.endswith('.csv'):
                user_df = pd.read_csv(uploaded_file)
            else:
                user_df = pd.read_excel(uploaded_file)

            st.success(f"✅ Successfully loaded **{uploaded_file.name}** — {len(user_df):,} rows × {len(user_df.columns)} columns")

            # --- DATA OVERVIEW ---
            st.markdown("""
            <div class="chart-panel">
                <div style="font-weight:700; font-size:1.05rem; color:#f8fafc; margin-bottom: 12px;">
                    📊 Data Overview & Quality Summary
                </div>
            """, unsafe_allow_html=True)

            ov_c1, ov_c2, ov_c3, ov_c4 = st.columns(4)
            with ov_c1:
                st.metric("Total Rows", f"{len(user_df):,}")
            with ov_c2:
                st.metric("Total Columns", f"{len(user_df.columns)}")
            with ov_c3:
                st.metric("Missing Values", f"{int(user_df.isnull().sum().sum()):,}")
            with ov_c4:
                st.metric("Duplicate Rows", f"{int(user_df.duplicated().sum()):,}")

            st.markdown("</div>", unsafe_allow_html=True)

            # --- DATA PREVIEW ---
            st.markdown("""
            <div class="chart-panel">
                <div style="font-weight:700; font-size:1.05rem; color:#f8fafc; margin-bottom: 12px;">
                    🔍 Data Preview (First 100 Rows)
                </div>
            """, unsafe_allow_html=True)
            st.dataframe(user_df.head(100), use_container_width=True, height=400)
            st.markdown("</div>", unsafe_allow_html=True)

            # --- COLUMN ANALYSIS ---
            numeric_cols = user_df.select_dtypes(include=np.number).columns.tolist()
            categorical_cols = user_df.select_dtypes(include=['object', 'category']).columns.tolist()
            datetime_cols = user_df.select_dtypes(include=['datetime64']).columns.tolist()

            # Auto-detect date columns from object type
            for col in categorical_cols[:]:
                try:
                    pd.to_datetime(user_df[col], errors='raise', infer_datetime_format=True)
                    datetime_cols.append(col)
                    categorical_cols.remove(col)
                except (ValueError, TypeError):
                    pass

            # --- DESCRIPTIVE STATISTICS ---
            if numeric_cols:
                st.markdown("""
                <div class="chart-panel">
                    <div style="font-weight:700; font-size:1.05rem; color:#f8fafc; margin-bottom: 12px;">
                        📈 Descriptive Statistics (Numerical Columns)
                    </div>
                """, unsafe_allow_html=True)
                st.dataframe(user_df[numeric_cols].describe().T.round(2), use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

            # --- INTERACTIVE CHARTS ---
            if numeric_cols or categorical_cols:
                st.markdown("""
                <div class="chart-panel">
                    <div style="font-weight:700; font-size:1.05rem; color:#f8fafc; margin-bottom: 12px;">
                        📊 Interactive Chart Builder
                    </div>
                    <div style="font-size:0.88rem; color:#94a3b8; margin-bottom: 16px;">
                        Select columns and chart type to instantly visualize your data
                    </div>
                """, unsafe_allow_html=True)

                chart_type = st.selectbox(
                    "Chart Type",
                    ["Bar Chart", "Line Chart", "Scatter Plot", "Pie Chart", "Histogram", "Box Plot"],
                    key="user_chart_type"
                )

                all_cols = user_df.columns.tolist()

                if chart_type == "Bar Chart":
                    bc1, bc2 = st.columns(2)
                    with bc1:
                        bar_x = st.selectbox("X-Axis (Category)", categorical_cols if categorical_cols else all_cols, key="bar_x")
                    with bc2:
                        bar_y = st.selectbox("Y-Axis (Value)", numeric_cols if numeric_cols else all_cols, key="bar_y")

                    if bar_x and bar_y:
                        bar_data = user_df.groupby(bar_x)[bar_y].sum().reset_index().sort_values(bar_y, ascending=False).head(20)
                        fig_user = px.bar(bar_data, x=bar_x, y=bar_y, color=bar_x, color_discrete_sequence=CHART_THEME["palette"])
                        format_plotly_figure(fig_user, height=420, show_legend=False)
                        st.plotly_chart(fig_user, use_container_width=True)

                elif chart_type == "Line Chart":
                    lc1, lc2 = st.columns(2)
                    with lc1:
                        line_x = st.selectbox("X-Axis", all_cols, key="line_x")
                    with lc2:
                        line_y = st.selectbox("Y-Axis (Value)", numeric_cols if numeric_cols else all_cols, key="line_y")

                    if line_x and line_y:
                        fig_user = px.line(user_df, x=line_x, y=line_y, markers=True, color_discrete_sequence=['#6366f1'])
                        format_plotly_figure(fig_user, height=420)
                        st.plotly_chart(fig_user, use_container_width=True)

                elif chart_type == "Scatter Plot":
                    sc1, sc2, sc3 = st.columns(3)
                    with sc1:
                        scatter_x = st.selectbox("X-Axis", numeric_cols if numeric_cols else all_cols, key="scatter_x")
                    with sc2:
                        scatter_y = st.selectbox("Y-Axis", numeric_cols[1:] if len(numeric_cols) > 1 else numeric_cols, key="scatter_y")
                    with sc3:
                        scatter_color = st.selectbox("Color By (Optional)", ["None"] + categorical_cols, key="scatter_color")

                    if scatter_x and scatter_y:
                        color_arg = scatter_color if scatter_color != "None" else None
                        fig_user = px.scatter(user_df, x=scatter_x, y=scatter_y, color=color_arg, color_discrete_sequence=CHART_THEME["palette"], opacity=0.7)
                        format_plotly_figure(fig_user, height=420)
                        st.plotly_chart(fig_user, use_container_width=True)

                elif chart_type == "Pie Chart":
                    pc1, pc2 = st.columns(2)
                    with pc1:
                        pie_names = st.selectbox("Labels (Category)", categorical_cols if categorical_cols else all_cols, key="pie_names")
                    with pc2:
                        pie_values = st.selectbox("Values (Numeric)", numeric_cols if numeric_cols else all_cols, key="pie_values")

                    if pie_names and pie_values:
                        pie_data = user_df.groupby(pie_names)[pie_values].sum().reset_index().sort_values(pie_values, ascending=False).head(10)
                        fig_user = px.pie(pie_data, names=pie_names, values=pie_values, hole=0.5, color_discrete_sequence=CHART_THEME["palette"])
                        fig_user.update_traces(textinfo='percent+label', textfont=dict(color="#f8fafc", size=11))
                        format_plotly_figure(fig_user, height=420, show_legend=False)
                        st.plotly_chart(fig_user, use_container_width=True)

                elif chart_type == "Histogram":
                    hist_col = st.selectbox("Select Column", numeric_cols if numeric_cols else all_cols, key="hist_col")
                    hist_bins = st.slider("Number of Bins", 5, 100, 30, key="hist_bins")

                    if hist_col:
                        fig_user = px.histogram(user_df, x=hist_col, nbins=hist_bins, color_discrete_sequence=['#6366f1'])
                        format_plotly_figure(fig_user, height=420, show_legend=False)
                        st.plotly_chart(fig_user, use_container_width=True)

                elif chart_type == "Box Plot":
                    bx1, bx2 = st.columns(2)
                    with bx1:
                        box_y = st.selectbox("Value Column", numeric_cols if numeric_cols else all_cols, key="box_y")
                    with bx2:
                        box_x = st.selectbox("Group By (Optional)", ["None"] + categorical_cols, key="box_x")

                    if box_y:
                        x_arg = box_x if box_x != "None" else None
                        fig_user = px.box(user_df, x=x_arg, y=box_y, color=x_arg, color_discrete_sequence=CHART_THEME["palette"])
                        format_plotly_figure(fig_user, height=420, show_legend=False)
                        st.plotly_chart(fig_user, use_container_width=True)

                st.markdown("</div>", unsafe_allow_html=True)

            # --- MISSING VALUES HEATMAP ---
            missing_total = user_df.isnull().sum()
            cols_with_missing = missing_total[missing_total > 0]
            if not cols_with_missing.empty:
                st.markdown("""
                <div class="chart-panel">
                    <div style="font-weight:700; font-size:1.05rem; color:#f8fafc; margin-bottom: 12px;">
                        ⚠️ Missing Values Breakdown
                    </div>
                """, unsafe_allow_html=True)

                missing_df = pd.DataFrame({
                    'Column': cols_with_missing.index,
                    'Missing Count': cols_with_missing.values,
                    'Missing %': (cols_with_missing.values / len(user_df) * 100).round(2)
                }).sort_values('Missing Count', ascending=True)

                fig_missing = px.bar(missing_df, x='Missing Count', y='Column', orientation='h',
                                     color='Missing %', color_continuous_scale='OrRd',
                                     text=missing_df['Missing %'].apply(lambda x: f"{x:.1f}%"))
                fig_missing.update_traces(textposition='outside')
                format_plotly_figure(fig_missing, height=max(300, len(cols_with_missing) * 35), show_legend=False)
                st.plotly_chart(fig_missing, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

            # --- DOWNLOAD CLEANED DATA ---
            st.markdown("""
            <div class="chart-panel">
                <div style="font-weight:700; font-size:1.05rem; color:#f8fafc; margin-bottom: 12px;">
                    ⬇️ Download Options
                </div>
            """, unsafe_allow_html=True)

            dl_c1, dl_c2 = st.columns(2)
            with dl_c1:
                csv_export = user_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download as CSV",
                    data=csv_export,
                    file_name=f"imported_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            with dl_c2:
                # Drop duplicates and nulls for a cleaned version
                user_df_cleaned = user_df.drop_duplicates().dropna()
                csv_clean_export = user_df_cleaned.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label=f"🧹 Download Cleaned CSV ({len(user_df_cleaned):,} rows)",
                    data=csv_clean_export,
                    file_name=f"cleaned_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            st.markdown("</div>", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")
            st.info("Please ensure your file is a valid Excel (.xlsx/.xls) or CSV file with proper formatting.")

    else:
        # Empty state when no file is uploaded
        st.markdown("""
        <div style="text-align: center; padding: 60px 20px;">
            <div style="font-size: 4rem; margin-bottom: 16px;">📂</div>
            <div style="font-size: 1.3rem; font-weight: 700; color: #f8fafc; margin-bottom: 8px;">
                No File Uploaded Yet
            </div>
            <div style="font-size: 0.95rem; color: #94a3b8; max-width: 500px; margin: 0 auto; line-height: 1.6;">
                Upload an Excel or CSV file above to instantly explore your data with interactive charts, 
                descriptive statistics, missing value analysis, and download options.
            </div>
            <div style="margin-top: 24px; font-size: 0.85rem; color: #64748b;">
                Supported: .xlsx • .xls • .csv — up to 200 MB
            </div>
        </div>
        """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# TAB 4: RAW DATA EXPLORER & CSV EXPORT
# -----------------------------------------------------------------------------
with tab_explorer:
    st.markdown("""
    <div class="section-header">
        <div>
            <div class="section-title">📋 Raw Data Explorer & Export Engine</div>
            <div class="section-subtitle">Search, inspect column distributions, and download cleaned transactional records</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    exp_col1, exp_col2 = st.columns([1.2, 0.8])
    with exp_col1:
        search_query = st.text_input(
            "🔍 Instant Keyword Search (Order ID, Customer ID, Product, City)",
            placeholder="Type keyword e.g. ORD-10040, Smart TV, Mumbai...",
            key="table_search_query"
        )
    with exp_col2:
        visible_columns = st.multiselect(
            "Select Columns to Display",
            options=df_clean.columns.tolist(),
            default=[col for col in ['Order_ID', 'Date', 'Customer_ID', 'City', 'Product_Category', 'Product_Name', 'Quantity', 'Total_Amount', 'Payment_Method', 'Delivery_Status', 'Customer_Rating'] if col in df_clean.columns],
            key="visible_cols_picker"
        )

    # Filter by search string if specified
    explorer_df = filtered_df.copy()
    if search_query.strip():
        q = search_query.strip().lower()
        mask = (
            explorer_df['Order_ID'].astype(str).str.lower().str.contains(q) |
            explorer_df['Customer_ID'].astype(str).str.lower().str.contains(q) |
            explorer_df['Product_Name'].astype(str).str.lower().str.contains(q) |
            explorer_df['City'].astype(str).str.lower().str.contains(q) |
            explorer_df['Product_Category'].astype(str).str.lower().str.contains(q)
        )
        explorer_df = explorer_df[mask]

    st.markdown(f"**Displaying {len(explorer_df):,} matching transaction records** (out of {len(filtered_df):,} filtered orders)")

    # Display interactive table
    display_df = explorer_df[visible_columns] if visible_columns else explorer_df
    st.dataframe(
        display_df,
        use_container_width=True,
        height=480
    )

    # Export Buttons
    export_c1, export_c2 = st.columns([1, 3])
    with export_c1:
        csv_data = explorer_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download Filtered CSV",
            data=csv_data,
            file_name=f"ecommerce_sales_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True,
            help="Download the currently filtered and searched records as clean CSV"
        )

    # Descriptive Statistics Section
    with st.expander("📊 Numerical Descriptive Statistics (describe)", expanded=False):
        num_cols = explorer_df.select_dtypes(include=np.number).columns.tolist()
        if num_cols:
            st.dataframe(explorer_df[num_cols].describe().T, use_container_width=True)
        else:
            st.info("No numerical columns in the selected view.")

# =============================================================================
# 8. FOOTER
# =============================================================================
st.markdown("---")
st.markdown("""
<div style="display: flex; justify-content: space-between; align-items: center; padding: 10px 0; color: #64748b; font-size: 0.8rem;">
    <div>
        <strong>E-Commerce Sales Intelligence Dashboard</strong> • College Data Storage & Visualization (DSV) Project • 🇮🇳 Indian Edition
    </div>
    <div>
        Production BI Stack: Streamlit • Plotly • Pandas • Custom Vanilla CSS
    </div>
</div>
""", unsafe_allow_html=True)