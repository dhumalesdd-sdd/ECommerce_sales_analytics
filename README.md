# 🛍️ E-Commerce Sales Intelligence & Analytics Dashboard

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io)
[![Python](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Plotly](https://img.shields.io/badge/Plotly-5.20+-3F4F75?logo=plotly&logoColor=white)](https://plotly.com/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0+-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An enterprise-grade, interactive Business Intelligence (BI) analytics application and data pipeline engineered with **Python**, **Streamlit**, and **Plotly**. Tailored for modern e-commerce commerce intelligence with localized Indian financial paradigms (**₹ INR**, UPI, Net Banking, Credit/Debit cards, Cash on Delivery).

---

## 🌟 Key Features

### 1. 📊 Executive Visualizations & KPI Dashboard
- **Dynamic KPI Scorecards**: Real-time revenue tracking (₹), Total Unit Sales, Order Count, Average Order Value (AOV), Unique Customer Headcount, and Customer Satisfaction Rating.
- **Multi-Granular Revenue Trends**: Interactive temporal revenue charts supporting Daily, Weekly, and Monthly rollups with automatic smoothing.
- **Top Product & Category Analysis**: Revenue vs. Volume metrics, Pareto breakdowns, and margin-level product ranking.
- **Geographic & Demographic Segmentation**: Order distribution across Tier-1 and metropolitan hubs, gender cohort preferences, and age distributions.
- **Payment Method Intelligence**: Share of wallet analysis across UPI, Cards, Net Banking, and COD.

### 2. 💡 Business Insights & Intelligence Engine
- **Automated Strategic Takeaways**: Executive observations highlighting peak sales windows, top category revenue anchors, and customer satisfaction benchmarks.
- **Customer Lifetime & Frequency Dynamics**: Repeat purchase rate, frequency metrics, and customer retention insights.
- **Actionable Optimization Recommendations**: Data-driven strategies for inventory stocking, promotional discounts, and payment gateway optimizations.

### 3. 📤 Self-Service "Import Your Own Data" Hub
- **Universal Spreadsheet Support**: Upload your own **Excel (`.xlsx`, `.xls`)** or **CSV** files seamlessly.
- **Instant Schema Profiling**: Automatic column type detection, dataset shape reporting, and missing value breakdown.
- **No-Code Interactive Chart Builder**: Create custom visualizations on the fly with 6 chart styles:
  - 📊 Bar Chart
  - 📈 Line Chart
  - 🔵 Scatter Plot
  - 🥧 Pie Chart
  - 📉 Histogram
  - 📦 Box Plot
- **One-Click Clean & Export**: Export raw data or download an automated cleaned version with duplicates and null values resolved.

### 4. 📋 Interactive Data Explorer & Export
- **Search & Filter Table**: Full tabular view with multi-column filtering by category, city, date range, and payment method.
- **Direct CSV Download**: Export active filtered datasets directly to your local workstation.

---

## 📁 Repository Structure

```text
ECommerce_sales_analytics/
├── .gitignore                      # Git ignore patterns for Python, venv & Streamlit
├── README.md                       # Comprehensive documentation & project overview
├── requirements.txt                # Python package dependencies
├── app.py                          # Main Streamlit web application & BI dashboard
├── create_notebook.py              # Script to build data_analysis.ipynb pipeline
├── generate_sample_data.py         # Synthetic e-commerce transactional data generator
├── assets/
│   └── style.css                   # Custom dark-theme glassmorphism CSS design system
├── data/
│   └── ecommerce_sales_data.csv    # Benchmark e-commerce transaction dataset
└── notebooks/
    └── data_analysis.ipynb         # End-to-end 10-step exploratory data analysis notebook
```

---

## 🚀 Quick Start Guide

### Prerequisites
- Python 3.9+ installed on your machine.
- Git installed.

### 1. Clone the Repository
```bash
git clone https://github.com/dhumalesdd-sdd/ECommerce_sales_analytics.git
cd ECommerce_sales_analytics
```

### 2. Create and Activate a Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch the Dashboard
```bash
streamlit run app.py
```
Open your browser and navigate to **`http://localhost:8501`**.

---

## 📓 Jupyter Notebook Analysis Pipeline

The repository includes a dedicated 10-step analytical notebook located at `notebooks/data_analysis.ipynb`:
1. **Library Ingestion**: Initialization of `pandas`, `numpy`, `matplotlib`, `seaborn`, and `plotly`.
2. **Dataset Loading & Understanding**: Structural inspection, shape, columns, and data types.
3. **Data Profiling & Quality Audit**: Identifying duplicates, missing data points, and schema constraints.
4. **Data Cleaning & Transformation**: Handling null values (median/mode), type conversions, and numeric constraint clamping.
5. **Feature Engineering**: Deriving temporal metrics (Year-Month, Day of Week) and Gross vs. Discount margins.
6. **Descriptive Statistics & Business KPIs**: Revenue aggregates, AOV, and distribution percentiles.
7. **Exploratory Visualizations**: 6 production-grade figures analyzing trends, categories, and channels.
8. **Cohort & Customer Behavioral Analysis**: Basket sizes, repeat rates, and payment channel preferences.
9. **Discount Sensitivity vs. Revenue**: Elasticity analysis of discount rates against order volumes.
10. **Executive Summary & Actionable Recommendations**: Synthesized strategic insights for business decision-makers.

To launch the notebook:
```bash
jupyter notebook notebooks/data_analysis.ipynb
```

---

## 🛠️ Tech Stack

| Component | Technology |
| :--- | :--- |
| **Frontend & App Framework** | [Streamlit](https://streamlit.io/) |
| **Interactive Plotting** | [Plotly Graph Objects & Express](https://plotly.com/) |
| **Data Manipulation** | [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/) |
| **Statistical Visuals** | [Matplotlib](https://matplotlib.org/), [Seaborn](https://seaborn.pydata.org/) |
| **Spreadsheet Ingestion** | [openpyxl](https://openpyxl.readthedocs.io/) |
| **Design System** | Custom Vanilla CSS (Glassmorphism & Dark Mode) |

---

## 👤 Author

- **Shravani Dhumale**
- GitHub: [@dhumalesdd-sdd](https://github.com/dhumalesdd-sdd)

---

## 📄 License

This project is licensed under the [MIT License](LICENSE) - feel free to use and modify for learning, research, or commercial analytics.
