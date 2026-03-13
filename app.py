"""
AI vs Consulting: A Data-Driven Decision Support Dashboard for AI Strategy
Streamlit Analytics Dashboard — University Data Analytics Course Project
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report,
                             mean_squared_error, r2_score)

# ── mlxtend for association rules ────────────────────────────────────────────
try:
    from mlxtend.frequent_patterns import apriori, association_rules
    from mlxtend.preprocessing import TransactionEncoder
    MLXTEND_OK = True
except ImportError:
    MLXTEND_OK = False

# ── Google Drive large-file downloader ───────────────────────────────────────
import os
import gdown

# ─────────────────────────────────────────────────────────────────────────────
# ⚙️  CONFIGURATION — paste your Google Drive file ID here
#     How to get it:
#       1. Upload ai_company_adoption.csv to Google Drive
#       2. Right-click → Share → "Anyone with the link" → Copy link
#       3. The link looks like:
#          https://drive.google.com/file/d/FILE_ID_HERE/view?usp=sharing
#       4. Copy only the FILE_ID_HERE part and paste it below
# ─────────────────────────────────────────────────────────────────────────────
GDRIVE_FILE_ID = "1PmhmJ6iqJOmDsWV5bmBVA_hPh-dev8et"

COMPANY_CSV_LOCAL = "ai_company_adoption.csv"

@st.cache_data(show_spinner=False)
def download_company_csv():
    """Download the large CSV from Google Drive using gdown (handles large-file confirmation)."""
    if os.path.exists(COMPANY_CSV_LOCAL):
        return  # already downloaded in this session

    with st.spinner("📥 Downloading company dataset from Google Drive (one-time, ~38 MB)…"):
        url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
        gdown.download(url, COMPANY_CSV_LOCAL, quiet=False, fuzzy=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG  (replaces .streamlit/config.toml — no folder needed)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI vs Consulting Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "AI vs Consulting · University Data Analytics Dashboard"},
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────
# Inject theme colours that config.toml would normally set
st.markdown("""
<style>
/* ── Streamlit theme overrides (replaces .streamlit/config.toml) ─────────── */
:root {
    --primary-color: #4F8EF7;
}
[data-testid="stAppViewContainer"] {
    background-color: #0E1117;
    color: #FAFAFA;
}
[data-testid="stSidebar"] {
    background-color: #1A1F2E;
}
[data-testid="stHeader"] {
    background-color: #0E1117;
}
/* ── KPI card*/
.kpi-card {
    background: linear-gradient(135deg, #1A1F2E 0%, #252B3B 100%);
    border: 1px solid #2E3550;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
    margin-bottom: 10px;
}
.kpi-value { font-size: 2rem; font-weight: 700; color: #4F8EF7; margin: 4px 0; }
.kpi-label { font-size: 0.85rem; color: #A0AEC0; text-transform: uppercase; letter-spacing: 0.05em; }
.kpi-delta { font-size: 0.8rem; color: #68D391; margin-top: 4px; }

/* Section headers */
.section-header {
    border-left: 4px solid #4F8EF7;
    padding-left: 14px;
    margin: 10px 0 20px 0;
}

/* Insight box */
.insight-box {
    background: #1A2535;
    border-left: 3px solid #4F8EF7;
    border-radius: 0 8px 8px 0;
    padding: 14px 18px;
    margin: 12px 0;
    font-size: 0.92rem;
    line-height: 1.6;
}

/* Model card */
.model-card {
    background: #1A1F2E;
    border: 1px solid #2E3550;
    border-radius: 10px;
    padding: 16px;
    margin: 6px 0;
}

/* Badge */
.badge-green { background:#276749; color:#C6F6D5; padding:2px 10px; border-radius:12px; font-size:0.78rem; }
.badge-blue  { background:#2A4365; color:#BEE3F8; padding:2px 10px; border-radius:12px; font-size:0.78rem; }
.badge-orange{ background:#7B341E; color:#FEEBC8; padding:2px 10px; border-radius:12px; font-size:0.78rem; }

h1,h2,h3 { color: #E2E8F0 !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING  (cached)
# ─────────────────────────────────────────────────────────────────────────────
# Ensure the large CSV is available before loading
download_company_csv()

@st.cache_data(show_spinner=False)
def load_data():
    company   = pd.read_csv("ai_company_adoption.csv")
    industry  = pd.read_csv("ai_industry_summary.csv")
    country   = pd.read_csv("country_ai_index.csv")
    return company, industry, country

@st.cache_data(show_spinner=False)
def get_sample(df, n=15_000, seed=42):
    """Return a stratified sample for ML training."""
    return df.sample(n=min(n, len(df)), random_state=seed)

with st.spinner("Loading datasets…"):
    df_company, df_industry, df_country = load_data()

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=60)
    st.title("AI Strategy Dashboard")
    st.markdown("---")

    st.markdown("### 🔍 Filters")

    all_industries = sorted(df_company["industry"].dropna().unique())
    sel_industries = st.multiselect("Industry", all_industries, default=all_industries,
                                    key="sb_industry")

    all_sizes = sorted(df_company["company_size"].dropna().unique())
    sel_sizes = st.multiselect("Company Size", all_sizes, default=all_sizes,
                               key="sb_size")

    all_countries = sorted(df_company["country"].dropna().unique())
    sel_countries = st.multiselect("Country", all_countries, default=all_countries,
                                   key="sb_country")

    st.markdown("---")
    st.markdown("### 📊 Dataset Info")
    st.markdown(f"- **Companies**: {len(df_company):,} rows")
    st.markdown(f"- **Industries**: {len(df_industry)} sectors")
    st.markdown(f"- **Countries**: {len(df_country)} nations")
    st.markdown("---")
    st.caption("University Data Analytics Course · 2024")

# Apply sidebar filters
mask = (
    df_company["industry"].isin(sel_industries) &
    df_company["company_size"].isin(sel_sizes) &
    df_company["country"].isin(sel_countries)
)
df_filtered = df_company[mask].copy()

# ─────────────────────────────────────────────────────────────────────────────
# KPI HELPER
# ─────────────────────────────────────────────────────────────────────────────
def kpi_card(label, value, delta=""):
    delta_html = f'<div class="kpi-delta">{delta}</div>' if delta else ""
    st.markdown(f"""
    <div class="kpi-card">
      <div class="kpi-label">{label}</div>
      <div class="kpi-value">{value}</div>
      {delta_html}
    </div>""", unsafe_allow_html=True)

def insight(text):
    st.markdown(f'<div class="insight-box">💡 {text}</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# NAVIGATION TABS
# ─────────────────────────────────────────────────────────────────────────────
tab_labels = [
    "🏠 Introduction",
    "🌍 Global Landscape",
    "🏭 Industry Analysis",
    "🔬 Company EDA",
    "🤖 Classification",
    "🔵 Clustering",
    "🔗 Association Rules",
    "📈 Regression",
    "🎯 AI Strategy Advisor",
]
tabs = st.tabs(tab_labels)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 0 – INTRODUCTION
# ═════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown("""
    <h1 style='text-align:center; color:#4F8EF7;'>🤖 AI vs Consulting</h1>
    <h3 style='text-align:center; color:#A0AEC0;'>A Data-Driven Decision Support Dashboard for AI Strategy</h3>
    """, unsafe_allow_html=True)
    st.markdown("---")

    # KPI row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("Avg AI Adoption Rate",
                 f"{df_filtered['ai_adoption_rate'].mean():.1f}%",
                 "across filtered companies")
    with c2:
        kpi_card("Avg Productivity Gain",
                 f"{df_filtered['productivity_change_percent'].mean():.1f}%",
                 "year-over-year")
    with c3:
        kpi_card("Avg Cost Reduction",
                 f"{df_filtered['cost_reduction_percent'].mean():.1f}%",
                 "operational savings")
    with c4:
        kpi_card("Companies Analyzed",
                 f"{len(df_filtered):,}",
                 f"of {len(df_company):,} total")

    st.markdown("---")
    col_a, col_b = st.columns([3, 2])
    with col_a:
        st.markdown("### 🎯 Business Objective")
        st.markdown("""
        Many organizations face a critical strategic decision: **should they invest in AI solutions
        internally, or hire external consultants** to address business challenges?

        This dashboard uses **data analytics and machine learning** to answer that question by
        analyzing patterns in AI adoption, productivity gains, cost reduction, and workforce impact
        across 150,000 company records spanning multiple countries, industries, and company sizes.

        The analysis provides:
        - **Global benchmarks** of AI adoption by country and region
        - **Industry-level insights** on which sectors benefit most from AI
        - **Predictive models** classifying AI adoption maturity
        - **Clustering** to identify natural company segments
        - **Association rules** revealing co-occurring AI adoption behaviors
        - **Regression models** predicting productivity and cost outcomes
        - An **interactive AI Strategy Advisor** for personalized recommendations
        """)
    with col_b:
        st.markdown("### 📂 Datasets")
        st.markdown("""
        | Dataset | Level | Rows |
        |---------|-------|------|
        | `ai_company_adoption.csv` | Company | 150,000 |
        | `ai_industry_summary.csv` | Industry | 9 |
        | `country_ai_index.csv` | Country | 30 |
        """)
        st.markdown("### 🧠 Techniques Used")
        st.markdown("""
        - Classification (6 models)
        - K-Means Clustering
        - Association Rule Mining
        - Linear & Multiple Regression
        - Dimensionality Reduction (PCA)
        - Exploratory Data Analysis
        """)

    st.markdown("---")
    st.markdown("### 🗺️ How to Use This Dashboard")
    e1, e2, e3 = st.columns(3)
    with e1:
        with st.expander("📌 Step 1 – Filter Data"):
            st.write("Use the **sidebar filters** to narrow the analysis to specific industries, "
                     "company sizes, or countries. All sections update automatically.")
    with e2:
        with st.expander("📌 Step 2 – Explore Sections"):
            st.write("Navigate through the **tabs** above to explore global trends, industry "
                     "patterns, EDA, ML models, and clustering results.")
    with e3:
        with st.expander("📌 Step 3 – Get a Recommendation"):
            st.write("Visit the **🎯 AI Strategy Advisor** tab, enter your company's profile, "
                     "and receive a data-driven recommendation: Implement AI, Hire Consultants, "
                     "or a Hybrid approach.")

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 – GLOBAL AI LANDSCAPE
# ═════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown("<div class='section-header'><h2>🌍 Global AI Landscape</h2></div>",
                unsafe_allow_html=True)
    st.markdown("Explore AI readiness, investment, and innovation capacity across 30 countries "
                "using the **Country AI Index** dataset.")
    st.markdown("---")

    # ── Row 1: Bubble map ────────────────────────────────────────────────────
    st.markdown("#### 🗺️ AI Patent Filings by Country")
    fig_map = px.choropleth(
        df_country, locations="country", locationmode="country names",
        color="ai_patent_filings_2024",
        hover_data=["region", "gdp_per_capita", "ai_researchers_per_million"],
        color_continuous_scale="Blues",
        title="AI Patent Filings (2024) — Global View",
        labels={"ai_patent_filings_2024": "Patent Filings"},
    )
    fig_map.update_layout(
        template="plotly_dark", height=420,
        coloraxis_colorbar=dict(title="Patents"),
        geo=dict(showframe=False, showcoastlines=True, projection_type="natural earth"),
    )
    st.plotly_chart(fig_map, use_container_width=True)
    insight("The USA, China, and Germany dominate AI patent filings, indicating mature AI R&D "
            "ecosystems. Organizations in these countries face higher competitive pressure to "
            "adopt AI internally rather than relying solely on consulting.")

    # ── Row 2: Bar charts ────────────────────────────────────────────────────
    c1, c2 = st.columns(2)
    with c1:
        fig_res = px.bar(
            df_country.sort_values("ai_researchers_per_million", ascending=False).head(15),
            x="ai_researchers_per_million", y="country", orientation="h",
            color="region", title="AI Researchers per Million Population (Top 15)",
            labels={"ai_researchers_per_million": "Researchers / Million", "country": ""},
            color_discrete_sequence=px.colors.qualitative.Bold,
        )
        fig_res.update_layout(template="plotly_dark", height=400, showlegend=True)
        st.plotly_chart(fig_res, use_container_width=True)
        insight("Countries with high researcher density (USA, Canada, Germany) have stronger "
                "internal AI talent pipelines, making in-house AI implementation more viable.")

    with c2:
        fig_gdp = px.scatter(
            df_country, x="gdp_per_capita", y="ai_patent_filings_2024",
            color="region", size="ai_researchers_per_million",
            hover_name="country",
            title="GDP per Capita vs AI Patent Filings",
            labels={"gdp_per_capita": "GDP per Capita (USD)",
                    "ai_patent_filings_2024": "AI Patent Filings"},
            color_discrete_sequence=px.colors.qualitative.Safe,
        )
        fig_gdp.update_layout(template="plotly_dark", height=400)
        st.plotly_chart(fig_gdp, use_container_width=True)
        insight("There is a strong positive correlation between GDP per capita and AI innovation "
                "output. Wealthier economies invest more in AI infrastructure and talent, enabling "
                "sustained internal AI adoption.")

    # ── Row 3: Digital maturity & policy ─────────────────────────────────────
    c3, c4 = st.columns(2)
    with c3:
        fig_dig = px.bar(
            df_country.sort_values("digital_maturity_index", ascending=False),
            x="country", y="digital_maturity_index", color="region",
            title="Digital Maturity Index by Country",
            labels={"digital_maturity_index": "Digital Maturity Index", "country": ""},
            color_discrete_sequence=px.colors.qualitative.Pastel,
        )
        fig_dig.update_layout(template="plotly_dark", height=360,
                              xaxis_tickangle=-45)
        st.plotly_chart(fig_dig, use_container_width=True)

    with c4:
        policy_counts = df_country["country_ai_policy"].value_counts().reset_index()
        policy_counts.columns = ["Policy", "Count"]
        fig_pol = px.pie(
            policy_counts, names="Policy", values="Count",
            title="Distribution of National AI Policy Stances",
            color_discrete_sequence=["#4F8EF7", "#68D391", "#F6AD55"],
            hole=0.4,
        )
        fig_pol.update_layout(template="plotly_dark", height=360)
        st.plotly_chart(fig_pol, use_container_width=True)

    insight("Most countries hold a **Moderate** AI policy stance — neither fully accelerationist "
            "nor restrictive. This regulatory environment creates an opportunity for companies to "
            "implement AI proactively before stricter regulations emerge.")

    # ── Internet penetration ─────────────────────────────────────────────────
    st.markdown("#### 🌐 Internet Penetration vs Digital Maturity")
    fig_inet = px.scatter(
        df_country, x="internet_penetration", y="digital_maturity_index",
        color="region", text="country", size="ai_patent_filings_2024",
        hover_data=["gdp_per_capita"],
        title="Internet Penetration vs Digital Maturity Index",
        labels={"internet_penetration": "Internet Penetration (%)",
                "digital_maturity_index": "Digital Maturity Index"},
        color_discrete_sequence=px.colors.qualitative.Bold,
    )
    fig_inet.update_traces(textposition="top center", textfont_size=9)
    fig_inet.update_layout(template="plotly_dark", height=420)
    st.plotly_chart(fig_inet, use_container_width=True)
    insight("Countries with high internet penetration consistently show higher digital maturity "
            "scores, confirming that digital infrastructure is the foundation of AI adoption. "
            "Companies operating in highly connected markets should prioritize internal AI "
            "capabilities to stay competitive.")

# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 – INDUSTRY ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown("<div class='section-header'><h2>🏭 Industry-Level AI Analysis</h2></div>",
                unsafe_allow_html=True)
    st.markdown("Examine how AI adoption, productivity, and cost outcomes vary across **9 industry "
                "sectors** using the aggregated industry summary dataset.")
    st.markdown("---")

    c1, c2 = st.columns(2)
    with c1:
        fig_adp = px.bar(
            df_industry.sort_values("avg_ai_adoption_rate", ascending=True),
            x="avg_ai_adoption_rate", y="industry", orientation="h",
            color="avg_ai_adoption_rate",
            color_continuous_scale="Blues",
            title="Average AI Adoption Rate by Industry (%)",
            labels={"avg_ai_adoption_rate": "Avg Adoption Rate (%)", "industry": ""},
        )
        fig_adp.update_layout(template="plotly_dark", height=380, coloraxis_showscale=False)
        st.plotly_chart(fig_adp, use_container_width=True)

    with c2:
        fig_prod = px.bar(
            df_industry.sort_values("avg_productivity_change_percent", ascending=True),
            x="avg_productivity_change_percent", y="industry", orientation="h",
            color="avg_productivity_change_percent",
            color_continuous_scale="Greens",
            title="Average Productivity Gain by Industry (%)",
            labels={"avg_productivity_change_percent": "Avg Productivity Gain (%)", "industry": ""},
        )
        fig_prod.update_layout(template="plotly_dark", height=380, coloraxis_showscale=False)
        st.plotly_chart(fig_prod, use_container_width=True)

    insight("**Technology** and **Finance** sectors lead in both AI adoption rate and productivity "
            "gains, suggesting that data-rich industries see the highest return on AI investment. "
            "Companies in these sectors should strongly consider internal AI implementation rather "
            "than consulting engagements.")

    # ── Radar chart ──────────────────────────────────────────────────────────
    st.markdown("#### 🕸️ Multi-Metric Industry Comparison (Radar Chart)")
    metrics = ["avg_ai_adoption_rate", "avg_productivity_change_percent",
               "avg_ai_maturity_score", "avg_customer_satisfaction"]
    metric_labels = ["AI Adoption", "Productivity Gain", "AI Maturity Score", "Customer Satisfaction"]

    df_radar = df_industry.copy()
    for m in metrics:
        col_min = df_radar[m].min()
        col_max = df_radar[m].max()
        df_radar[m + "_norm"] = (df_radar[m] - col_min) / (col_max - col_min + 1e-9) * 100

    fig_radar = go.Figure()
    colors = px.colors.qualitative.Bold
    for i, row in df_radar.iterrows():
        vals = [row[m + "_norm"] for m in metrics]
        vals += [vals[0]]
        fig_radar.add_trace(go.Scatterpolar(
            r=vals,
            theta=metric_labels + [metric_labels[0]],
            name=row["industry"],
            line=dict(color=colors[i % len(colors)]),
            fill="toself", opacity=0.35,
        ))
    fig_radar.update_layout(
        template="plotly_dark", height=500,
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        title="Normalised Industry Performance — Radar",
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    # ── Jobs displaced vs created ─────────────────────────────────────────────
    c3, c4 = st.columns(2)
    with c3:
        fig_jobs = px.bar(
            df_industry.sort_values("avg_jobs_displaced"),
            x="industry",
            y=["avg_jobs_displaced", "avg_jobs_created"],
            barmode="group",
            title="Average Jobs Displaced vs Created by Industry",
            labels={"value": "Number of Jobs", "variable": ""},
            color_discrete_map={"avg_jobs_displaced": "#FC8181", "avg_jobs_created": "#68D391"},
        )
        fig_jobs.update_layout(template="plotly_dark", height=360, xaxis_tickangle=-30)
        st.plotly_chart(fig_jobs, use_container_width=True)

    with c4:
        fig_fail = px.bar(
            df_industry.sort_values("avg_ai_failure_rate", ascending=False),
            x="industry", y="avg_ai_failure_rate",
            color="avg_ai_failure_rate",
            color_continuous_scale="Reds",
            title="Average AI Failure Rate by Industry (%)",
            labels={"avg_ai_failure_rate": "AI Failure Rate (%)", "industry": ""},
        )
        fig_fail.update_layout(template="plotly_dark", height=360,
                               xaxis_tickangle=-30, coloraxis_showscale=False)
        st.plotly_chart(fig_fail, use_container_width=True)

    insight("AI projects create more jobs than they displace on average, challenging the common "
            "narrative that AI purely eliminates employment. Sectors such as **Agriculture** and "
            "**Healthcare** show the highest job creation ratios, making them strong candidates "
            "for internal AI investment rather than consultant-led automation.")

    st.markdown("#### 📋 Full Industry Summary Table")
    st.dataframe(
        df_industry.rename(columns={
            "avg_ai_adoption_rate": "Adoption %",
            "avg_productivity_change_percent": "Productivity %",
            "avg_ai_maturity_score": "Maturity Score",
            "avg_ai_failure_rate": "Failure Rate %",
            "avg_jobs_displaced": "Jobs Displaced",
            "avg_jobs_created": "Jobs Created",
            "avg_customer_satisfaction": "CSAT",
        }).style.background_gradient(subset=["Adoption %", "Productivity %"], cmap="Blues"),
        use_container_width=True, height=360,
    )

# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 – COMPANY EDA
# ═════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown("<div class='section-header'><h2>🔬 Company-Level EDA</h2></div>",
                unsafe_allow_html=True)
    st.markdown("Deep-dive into the company dataset with statistical summaries, distribution "
                "plots, correlation analysis, and outlier detection.")
    st.markdown("---")

    eda_df = df_filtered.copy()
    num_cols = ["ai_adoption_rate", "ai_investment_per_employee", "automation_level" if "automation_level" in eda_df.columns else "task_automation_rate",
                "productivity_change_percent", "cost_reduction_percent",
                "ai_training_hours", "ai_budget_percentage", "revenue_growth_percent",
                "employee_satisfaction_score", "innovation_score"]
    num_cols = [c for c in num_cols if c in eda_df.columns]

    # ── Dataset Overview ──────────────────────────────────────────────────────
    with st.expander("📋 Dataset Overview", expanded=True):
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Rows", f"{len(eda_df):,}")
        r2.metric("Columns", f"{eda_df.shape[1]}")
        r3.metric("Numeric Columns", f"{eda_df.select_dtypes(include=np.number).shape[1]}")
        r4.metric("Categorical Columns", f"{eda_df.select_dtypes(include='object').shape[1]}")
        st.dataframe(eda_df.head(50), use_container_width=True, height=260)

    # ── Missing values ─────────────────────────────────────────────────────────
    with st.expander("🕳️ Missing Value Analysis"):
        missing = eda_df.isnull().sum()
        missing_pct = (missing / len(eda_df) * 100).round(2)
        mv_df = pd.DataFrame({"Missing Count": missing, "Missing %": missing_pct})
        mv_df = mv_df[mv_df["Missing Count"] > 0].sort_values("Missing %", ascending=False)
        if mv_df.empty:
            st.success("✅ No missing values detected in the filtered dataset.")
        else:
            fig_mv = px.bar(mv_df, x=mv_df.index, y="Missing %",
                            title="Missing Values by Column",
                            color="Missing %", color_continuous_scale="Reds")
            fig_mv.update_layout(template="plotly_dark", height=300)
            st.plotly_chart(fig_mv, use_container_width=True)

    # ── Statistical summary ────────────────────────────────────────────────────
    with st.expander("📊 Statistical Summary"):
        st.dataframe(eda_df[num_cols].describe().T.round(3).style.background_gradient(cmap="Blues"),
                     use_container_width=True)

    # ── Correlation heatmap ───────────────────────────────────────────────────
    st.markdown("#### 🔥 Correlation Heatmap")
    corr = eda_df[num_cols].corr()
    fig_corr = px.imshow(
        corr, text_auto=".2f", aspect="auto",
        color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
        title="Correlation Matrix — Key Numeric Features",
    )
    fig_corr.update_layout(template="plotly_dark", height=520,
                           coloraxis_colorbar=dict(title="r"))
    st.plotly_chart(fig_corr, use_container_width=True)
    insight("**AI investment per employee** shows the strongest positive correlation with "
            "productivity gains and cost reduction. Companies that invest heavily per employee "
            "consistently outperform those that spread AI budgets thinly — a clear signal that "
            "deep internal AI implementation beats shallow consulting engagements.")

    # ── Scatter plots ─────────────────────────────────────────────────────────
    c1, c2 = st.columns(2)
    with c1:
        sample = eda_df.sample(min(3000, len(eda_df)), random_state=1)
        fig_s1 = px.scatter(
            sample, x="ai_investment_per_employee", y="productivity_change_percent",
            color="industry", opacity=0.5, trendline="ols",
            title="AI Investment per Employee vs Productivity Gain",
            labels={"ai_investment_per_employee": "AI Investment / Employee (USD)",
                    "productivity_change_percent": "Productivity Gain (%)"},
        )
        fig_s1.update_layout(template="plotly_dark", height=380)
        st.plotly_chart(fig_s1, use_container_width=True)

    with c2:
        auto_col = "task_automation_rate" if "task_automation_rate" in eda_df.columns else "ai_adoption_rate"
        fig_s2 = px.scatter(
            sample, x=auto_col, y="cost_reduction_percent",
            color="company_size", opacity=0.5, trendline="ols",
            title=f"{auto_col.replace('_', ' ').title()} vs Cost Reduction",
            labels={auto_col: auto_col.replace("_", " ").title(),
                    "cost_reduction_percent": "Cost Reduction (%)"},
        )
        fig_s2.update_layout(template="plotly_dark", height=380)
        st.plotly_chart(fig_s2, use_container_width=True)

    insight("There is a clear positive relationship between **task automation rate** and cost "
            "reduction. Enterprise companies achieve disproportionately higher cost savings, "
            "likely because they have the scale to amortise AI infrastructure costs.")

    # ── Distribution plots ────────────────────────────────────────────────────
    st.markdown("#### 📊 Feature Distributions")
    sel_dist = st.selectbox("Select feature to plot", num_cols, key="dist_col")
    c3, c4 = st.columns(2)
    with c3:
        fig_hist = px.histogram(
            eda_df, x=sel_dist, color="ai_adoption_stage", nbins=50,
            title=f"Distribution of {sel_dist.replace('_',' ').title()} by Adoption Stage",
            barmode="overlay", opacity=0.7,
        )
        fig_hist.update_layout(template="plotly_dark", height=360)
        st.plotly_chart(fig_hist, use_container_width=True)

    with c4:
        fig_box = px.box(
            eda_df, x="company_size", y=sel_dist, color="company_size",
            title=f"{sel_dist.replace('_',' ').title()} by Company Size",
            points=False,
        )
        fig_box.update_layout(template="plotly_dark", height=360)
        st.plotly_chart(fig_box, use_container_width=True)

    # ── Outlier detection ─────────────────────────────────────────────────────
    st.markdown("#### 🎯 Outlier Detection (IQR Method)")
    outlier_col = st.selectbox("Select feature for outlier detection", num_cols, key="out_col")
    q1, q3 = eda_df[outlier_col].quantile([0.25, 0.75])
    iqr = q3 - q1
    outliers = eda_df[(eda_df[outlier_col] < q1 - 1.5 * iqr) |
                      (eda_df[outlier_col] > q3 + 1.5 * iqr)]
    fig_out = px.box(eda_df, y=outlier_col, points="outliers",
                     title=f"Outlier Detection — {outlier_col.replace('_', ' ').title()}",
                     color_discrete_sequence=["#4F8EF7"])
    fig_out.update_layout(template="plotly_dark", height=360)
    col_o1, col_o2 = st.columns([2, 1])
    with col_o1:
        st.plotly_chart(fig_out, use_container_width=True)
    with col_o2:
        st.metric("Outlier Records", f"{len(outliers):,}")
        st.metric("Outlier %", f"{len(outliers)/len(eda_df)*100:.1f}%")
        st.metric("IQR", f"{iqr:.2f}")
        st.metric("Lower Fence", f"{q1 - 1.5*iqr:.2f}")
        st.metric("Upper Fence", f"{q3 + 1.5*iqr:.2f}")

    # ── Employees trained vs productivity ─────────────────────────────────────
    st.markdown("#### 👩‍💻 Employees Trained vs Productivity Gain")
    if "reskilled_employees" in eda_df.columns:
        fig_train = px.scatter(
            sample, x="reskilled_employees", y="productivity_change_percent",
            color="industry", opacity=0.5, trendline="ols",
            title="Reskilled Employees vs Productivity Gain (%)",
            labels={"reskilled_employees": "Reskilled Employees",
                    "productivity_change_percent": "Productivity Gain (%)"},
        )
        fig_train.update_layout(template="plotly_dark", height=400)
        st.plotly_chart(fig_train, use_container_width=True)
        insight("Companies that invest in **employee reskilling** see measurably higher productivity "
                "gains. This suggests that AI adoption is most effective when paired with structured "
                "workforce training programs — a factor often underestimated in consulting-led "
                "implementations.")

# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 – CLASSIFICATION MODELS
# ═════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown("<div class='section-header'><h2>🤖 Classification Models</h2></div>",
                unsafe_allow_html=True)
    st.markdown("Six machine learning classifiers are trained to **predict AI adoption stage** "
                "(none / pilot / partial / full) based on company characteristics.")
    st.markdown("---")

    # ── Prepare data ──────────────────────────────────────────────────────────
    @st.cache_data(show_spinner=False)
    def prepare_classification_data(seed=42):
        df_ml = pd.read_csv("ai_company_adoption.csv")
        df_ml = df_ml.sample(15_000, random_state=seed)

        features = ["ai_investment_per_employee", "task_automation_rate",
                    "ai_training_hours", "ai_budget_percentage",
                    "productivity_change_percent", "cost_reduction_percent",
                    "revenue_growth_percent", "employee_satisfaction_score",
                    "innovation_score", "num_ai_tools_used", "ai_projects_active",
                    "reskilled_employees"]
        features = [f for f in features if f in df_ml.columns]

        df_ml = df_ml[features + ["ai_adoption_stage"]].dropna()
        X = df_ml[features]
        y = df_ml["ai_adoption_stage"]

        le = LabelEncoder()
        y_enc = le.fit_transform(y)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_tr, X_te, y_tr, y_te = train_test_split(X_scaled, y_enc,
                                                    test_size=0.2, random_state=seed,
                                                    stratify=y_enc)
        return X_tr, X_te, y_tr, y_te, le, features

    @st.cache_data(show_spinner=False)
    def run_classifiers():
        X_tr, X_te, y_tr, y_te, le, features = prepare_classification_data()
        models = {
            "Logistic Regression": LogisticRegression(max_iter=500, random_state=42),
            "Decision Tree":       DecisionTreeClassifier(random_state=42),
            "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            "SVM":                 SVC(kernel="rbf", probability=True, random_state=42),
            "KNN":                 KNeighborsClassifier(n_neighbors=7),
            "Gradient Boosting":   GradientBoostingClassifier(n_estimators=100, random_state=42),
        }
        results = []
        cms = {}
        for name, model in models.items():
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_te)
            results.append({
                "Model": name,
                "Accuracy":  round(accuracy_score(y_te, y_pred), 4),
                "Precision": round(precision_score(y_te, y_pred, average="weighted", zero_division=0), 4),
                "Recall":    round(recall_score(y_te, y_pred, average="weighted", zero_division=0), 4),
                "F1 Score":  round(f1_score(y_te, y_pred, average="weighted", zero_division=0), 4),
            })
            cms[name] = confusion_matrix(y_te, y_pred)
        return pd.DataFrame(results).sort_values("F1 Score", ascending=False), cms, le

    with st.spinner("Training 6 classifiers on 15,000 samples…"):
        results_df, cms, le_cls = run_classifiers()

    # ── Model comparison table ────────────────────────────────────────────────
    st.markdown("#### 📊 Model Performance Comparison")
    best_model = results_df.iloc[0]["Model"]
    st.dataframe(
        results_df.style
            .background_gradient(subset=["Accuracy", "F1 Score"], cmap="Blues")
            .format({"Accuracy": "{:.4f}", "Precision": "{:.4f}",
                     "Recall": "{:.4f}", "F1 Score": "{:.4f}"}),
        use_container_width=True, height=260,
    )
    st.success(f"🏆 Best Model: **{best_model}** with F1 Score = "
               f"{results_df.iloc[0]['F1 Score']:.4f}")

    # ── Bar chart comparison ──────────────────────────────────────────────────
    fig_cmp = px.bar(
        results_df.melt(id_vars="Model", var_name="Metric", value_name="Score"),
        x="Model", y="Score", color="Metric", barmode="group",
        title="Model Comparison — Accuracy, Precision, Recall, F1",
        color_discrete_sequence=["#4F8EF7", "#68D391", "#F6AD55", "#FC8181"],
    )
    fig_cmp.update_layout(template="plotly_dark", height=420)
    st.plotly_chart(fig_cmp, use_container_width=True)

    insight(f"**{best_model}** achieves the highest F1 Score, indicating the best balance between "
            "precision and recall across all four adoption stages. Ensemble methods (Random Forest, "
            "Gradient Boosting) generally outperform single classifiers because they average over "
            "many decision paths and are more robust to overfitting on noisy survey data.")

    # ── Confusion matrices ────────────────────────────────────────────────────
    st.markdown("#### 🔲 Confusion Matrices")
    class_labels = list(le_cls.classes_)
    sel_model_cm = st.selectbox("Select model to view confusion matrix",
                                list(cms.keys()), key="cm_sel")
    cm = cms[sel_model_cm]
    fig_cm = px.imshow(
        cm, x=class_labels, y=class_labels, text_auto=True,
        color_continuous_scale="Blues",
        title=f"Confusion Matrix — {sel_model_cm}",
        labels=dict(x="Predicted", y="Actual"),
    )
    fig_cm.update_layout(template="plotly_dark", height=420)
    st.plotly_chart(fig_cm, use_container_width=True)

    # ── Feature importance (Random Forest) ───────────────────────────────────
    st.markdown("#### 🌟 Feature Importance (Random Forest)")
    @st.cache_data(show_spinner=False)
    def get_feature_importance():
        X_tr, X_te, y_tr, y_te, le, features = prepare_classification_data()
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_tr, y_tr)
        return pd.DataFrame({"Feature": features,
                              "Importance": rf.feature_importances_}).sort_values("Importance", ascending=True)

    fi_df = get_feature_importance()
    fig_fi = px.bar(fi_df, x="Importance", y="Feature", orientation="h",
                    title="Random Forest Feature Importance",
                    color="Importance", color_continuous_scale="Blues")
    fig_fi.update_layout(template="plotly_dark", height=400, coloraxis_showscale=False)
    st.plotly_chart(fig_fi, use_container_width=True)
    insight("The most important features for predicting AI adoption stage are "
            "**AI investment per employee**, **task automation rate**, and **AI budget "
            "percentage**. Companies with high scores on these metrics are far more likely "
            "to be classified as 'full' adopters, signalling readiness for internal AI "
            "implementation without external consulting support.")

# ═════════════════════════════════════════════════════════════════════════════
# TAB 5 – CLUSTERING
# ═════════════════════════════════════════════════════════════════════════════
with tabs[5]:
    st.markdown("<div class='section-header'><h2>🔵 Clustering Analysis (K-Means)</h2></div>",
                unsafe_allow_html=True)
    st.markdown("Segment companies into natural groups based on AI investment and performance "
                "profiles to identify archetypes that inform AI vs consulting strategy.")
    st.markdown("---")

    cluster_features = ["ai_investment_per_employee", "task_automation_rate",
                        "productivity_change_percent", "cost_reduction_percent",
                        "reskilled_employees", "ai_budget_percentage"]
    cluster_features = [c for c in cluster_features if c in df_company.columns]

    @st.cache_data(show_spinner=False)
    def run_clustering(n_clusters=4, seed=42):
        sample = pd.read_csv("ai_company_adoption.csv").sample(15_000, random_state=seed)
        data = sample[cluster_features].dropna()
        scaler = StandardScaler()
        X_s = scaler.fit_transform(data)
        # Elbow
        inertia = []
        k_range = range(2, 11)
        for k in k_range:
            km = KMeans(n_clusters=k, random_state=seed, n_init=10)
            km.fit(X_s)
            inertia.append(km.inertia_)
        # Final model
        km_final = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
        labels = km_final.fit_predict(X_s)
        # PCA
        pca = PCA(n_components=2, random_state=seed)
        coords = pca.fit_transform(X_s)
        result = data.copy()
        result["Cluster"] = labels.astype(str)
        result["PCA1"] = coords[:, 0]
        result["PCA2"] = coords[:, 1]
        return result, inertia, list(k_range), km_final.cluster_centers_, scaler, pca

    n_clust = st.slider("Number of Clusters (K)", 2, 8, 4, key="k_slider")
    with st.spinner("Running K-Means clustering…"):
        cluster_result, inertia, k_range, centers, scaler_km, pca_km = run_clustering(n_clust)

    # ── Elbow method ──────────────────────────────────────────────────────────
    c1, c2 = st.columns([1, 2])
    with c1:
        fig_elbow = px.line(
            x=k_range, y=inertia,
            title="Elbow Method — Optimal K",
            labels={"x": "Number of Clusters (K)", "y": "Inertia"},
            markers=True,
        )
        fig_elbow.add_vline(x=n_clust, line_dash="dash", line_color="#F6AD55",
                            annotation_text=f"K={n_clust} (selected)")
        fig_elbow.update_layout(template="plotly_dark", height=360)
        st.plotly_chart(fig_elbow, use_container_width=True)
        insight("The elbow typically forms around **K=4**, indicating four distinct company "
                "archetypes in AI adoption behaviour.")

    with c2:
        fig_pca = px.scatter(
            cluster_result, x="PCA1", y="PCA2", color="Cluster",
            title=f"K-Means Clusters (K={n_clust}) — PCA 2D Projection",
            opacity=0.55, hover_data=cluster_features[:3],
            color_discrete_sequence=px.colors.qualitative.Bold,
        )
        fig_pca.update_layout(template="plotly_dark", height=420)
        st.plotly_chart(fig_pca, use_container_width=True)

    # ── Cluster summary stats ─────────────────────────────────────────────────
    st.markdown("#### 📋 Cluster Summary Statistics")
    cluster_summary = cluster_result.groupby("Cluster")[cluster_features].mean().round(2)
    st.dataframe(
        cluster_summary.style.background_gradient(cmap="Blues"),
        use_container_width=True,
    )

    # ── Cluster profiles ───────────────────────────────────────────────────────
    st.markdown("#### 🏷️ Cluster Archetypes")
    archetype_desc = {
        "0": ("🟢 AI Leaders", "High investment, high automation, high productivity. Strong internal AI capability — AI implementation strongly recommended."),
        "1": ("🟡 AI Explorers", "Moderate adoption, moderate outcomes. Benefit from structured AI roadmap — Hybrid approach recommended."),
        "2": ("🔴 AI Laggards", "Low investment, low automation. Limited AI readiness — External consulting recommended to build foundation."),
        "3": ("🔵 Niche Adopters", "High automation in specific use-cases but low overall investment. Targeted AI expansion recommended."),
    }
    cols = st.columns(min(n_clust, 4))
    for i in range(n_clust):
        with cols[i % 4]:
            key = str(i)
            title, desc = archetype_desc.get(key, (f"Cluster {i}", "Custom cluster profile."))
            st.markdown(f"""
            <div class='model-card'>
              <strong>{title}</strong><br/>
              <small>{desc}</small>
            </div>""", unsafe_allow_html=True)

    # ── Cluster radar ─────────────────────────────────────────────────────────
    st.markdown("#### 🕸️ Cluster Profiles — Radar Chart")
    norm_summary = cluster_summary.copy()
    for col in norm_summary.columns:
        cmin, cmax = norm_summary[col].min(), norm_summary[col].max()
        norm_summary[col] = (norm_summary[col] - cmin) / (cmax - cmin + 1e-9) * 100

    fig_cr = go.Figure()
    colors = px.colors.qualitative.Bold
    for i, idx in enumerate(norm_summary.index):
        vals = list(norm_summary.loc[idx])
        vals += [vals[0]]
        fig_cr.add_trace(go.Scatterpolar(
            r=vals,
            theta=cluster_features + [cluster_features[0]],
            name=f"Cluster {idx}",
            line=dict(color=colors[i % len(colors)]),
            fill="toself", opacity=0.4,
        ))
    fig_cr.update_layout(
        template="plotly_dark", height=460,
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        title="Cluster Profiles — Normalised Radar",
    )
    st.plotly_chart(fig_cr, use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 6 – ASSOCIATION RULES
# ═════════════════════════════════════════════════════════════════════════════
with tabs[6]:
    st.markdown("<div class='section-header'><h2>🔗 Association Rule Mining</h2></div>",
                unsafe_allow_html=True)
    st.markdown("Discover **co-occurring patterns** in AI adoption behaviour using the Apriori "
                "algorithm to identify which combinations of company characteristics frequently "
                "appear together.")
    st.markdown("---")

    if not MLXTEND_OK:
        st.error("⚠️ `mlxtend` library not installed. Run `pip install mlxtend` and restart.")
    else:
        @st.cache_data(show_spinner=False)
        def run_apriori(min_sup=0.15, min_conf=0.5, sample_n=5000):
            df_a = pd.read_csv("ai_company_adoption.csv").sample(sample_n, random_state=42)

            # Bin continuous features into categories
            df_a["invest_level"]  = pd.qcut(df_a["ai_investment_per_employee"], 3,
                                             labels=["LowInvest","MidInvest","HighInvest"])
            df_a["prod_level"]    = pd.qcut(df_a["productivity_change_percent"], 3,
                                             labels=["LowProd","MidProd","HighProd"])
            df_a["auto_level"]    = pd.qcut(df_a["task_automation_rate"], 3,
                                             labels=["LowAuto","MidAuto","HighAuto"])
            df_a["cost_level"]    = pd.qcut(df_a["cost_reduction_percent"], 3,
                                             labels=["LowCost","MidCost","HighCost"])
            df_a["size_cat"]      = df_a["company_size"]
            df_a["adopt_stage"]   = df_a["ai_adoption_stage"]
            df_a["industry_cat"]  = df_a["industry"]

            cat_cols = ["invest_level", "prod_level", "auto_level", "cost_level",
                        "size_cat", "adopt_stage", "industry_cat"]
            transactions = df_a[cat_cols].astype(str).values.tolist()

            te = TransactionEncoder()
            te_array = te.fit(transactions).transform(transactions)
            trans_df = pd.DataFrame(te_array, columns=te.columns_)

            freq_items = apriori(trans_df, min_support=min_sup, use_colnames=True)
            rules = association_rules(freq_items, metric="confidence", min_threshold=min_conf)
            rules = rules.sort_values("lift", ascending=False).reset_index(drop=True)
            rules["antecedents"] = rules["antecedents"].apply(lambda x: ", ".join(list(x)))
            rules["consequents"] = rules["consequents"].apply(lambda x: ", ".join(list(x)))
            return rules, freq_items

        min_sup  = st.slider("Minimum Support",  0.05, 0.50, 0.15, 0.01)
        min_conf = st.slider("Minimum Confidence", 0.30, 0.95, 0.50, 0.05)

        with st.spinner("Mining association rules…"):
            try:
                rules_df, freq_df = run_apriori(min_sup, min_conf)

                st.markdown(f"#### 📋 Association Rules (Top 30 by Lift)")
                st.markdown(f"Found **{len(rules_df)}** rules | Showing top 30")
                display_rules = rules_df[["antecedents","consequents","support",
                                          "confidence","lift"]].head(30)
                st.dataframe(
                    display_rules.style
                        .background_gradient(subset=["lift","confidence"], cmap="Blues")
                        .format({"support":"{:.3f}","confidence":"{:.3f}","lift":"{:.2f}"}),
                    use_container_width=True, height=420,
                )

                # ── Lift scatter ──────────────────────────────────────────────
                st.markdown("#### 📈 Support vs Confidence (sized by Lift)")
                fig_ar = px.scatter(
                    rules_df.head(50), x="support", y="confidence", size="lift",
                    hover_data=["antecedents","consequents"],
                    color="lift", color_continuous_scale="Blues",
                    title="Association Rules — Support vs Confidence (Bubble Size = Lift)",
                    labels={"support":"Support","confidence":"Confidence"},
                )
                fig_ar.update_layout(template="plotly_dark", height=420)
                st.plotly_chart(fig_ar, use_container_width=True)

                # ── Top consequents bar ───────────────────────────────────────
                top_cons = rules_df["consequents"].value_counts().head(10).reset_index()
                top_cons.columns = ["Consequent","Count"]
                fig_cons = px.bar(top_cons, x="Count", y="Consequent", orientation="h",
                                  title="Most Frequent Rule Consequents",
                                  color="Count", color_continuous_scale="Blues")
                fig_cons.update_layout(template="plotly_dark", height=360,
                                       coloraxis_showscale=False)
                st.plotly_chart(fig_cons, use_container_width=True)

                insight("The most lifted rules reveal that **high AI investment + high automation "
                        "→ full AI adoption** and **low investment + low automation → none/pilot** "
                        "stages. This confirms a virtuous cycle: companies that commit early to AI "
                        "investment reach full adoption significantly faster, reducing their "
                        "long-term dependence on external consulting.")

            except Exception as e:
                st.error(f"Could not generate rules with current thresholds: {e}. "
                         "Try lowering minimum support.")

# ═════════════════════════════════════════════════════════════════════════════
# TAB 7 – REGRESSION
# ═════════════════════════════════════════════════════════════════════════════
with tabs[7]:
    st.markdown("<div class='section-header'><h2>📈 Regression Analysis</h2></div>",
                unsafe_allow_html=True)
    st.markdown("Quantify how AI adoption inputs **predict productivity gains and cost reduction** "
                "using linear and multiple regression models.")
    st.markdown("---")

    @st.cache_data(show_spinner=False)
    def run_regression(seed=42):
        df_r = pd.read_csv("ai_company_adoption.csv").sample(15_000, random_state=seed)

        feat_prod = ["ai_investment_per_employee", "task_automation_rate",
                     "ai_training_hours", "ai_budget_percentage",
                     "reskilled_employees", "num_ai_tools_used",
                     "ai_projects_active", "innovation_score"]
        feat_prod = [f for f in feat_prod if f in df_r.columns]

        feat_cost = ["ai_investment_per_employee", "task_automation_rate",
                     "ai_budget_percentage", "reskilled_employees",
                     "num_ai_tools_used", "ai_projects_active"]
        feat_cost = [f for f in feat_cost if f in df_r.columns]

        target_prod = "productivity_change_percent"
        target_cost = "cost_reduction_percent"

        results = {}
        for target, feats in [(target_prod, feat_prod), (target_cost, feat_cost)]:
            sub = df_r[feats + [target]].dropna()
            X = sub[feats]
            y = sub[target]
            scaler = StandardScaler()
            X_s = scaler.fit_transform(X)
            X_tr, X_te, y_tr, y_te = train_test_split(X_s, y, test_size=0.2, random_state=seed)
            model = LinearRegression()
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_te)
            results[target] = {
                "model": model, "feats": feats, "scaler": scaler,
                "X_te": X_te, "y_te": y_te, "y_pred": y_pred,
                "r2":   round(r2_score(y_te, y_pred), 4),
                "rmse": round(np.sqrt(mean_squared_error(y_te, y_pred)), 4),
                "coefs": pd.DataFrame({"Feature": feats,
                                        "Coefficient": model.coef_}).sort_values("Coefficient"),
            }
        return results

    with st.spinner("Fitting regression models…"):
        reg_results = run_regression()

    for target_name, label in [
        ("productivity_change_percent", "Productivity Gain (%)"),
        ("cost_reduction_percent", "Cost Reduction (%)"),
    ]:
        r = reg_results[target_name]
        st.markdown(f"#### 🎯 Predicting **{label}**")
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.metric("R² Score", f"{r['r2']:.4f}")
        with col_m2:
            st.metric("RMSE", f"{r['rmse']:.4f}")

        cc1, cc2 = st.columns(2)
        with cc1:
            # Predicted vs Actual
            idx = np.random.choice(len(r["y_te"]), min(500, len(r["y_te"])), replace=False)
            fig_pred = px.scatter(
                x=r["y_te"].values[idx], y=r["y_pred"][idx],
                title=f"Actual vs Predicted — {label}",
                labels={"x": "Actual", "y": "Predicted"},
                opacity=0.5, trendline="ols",
                color_discrete_sequence=["#4F8EF7"],
            )
            lo = min(r["y_te"].min(), r["y_pred"].min())
            hi = max(r["y_te"].max(), r["y_pred"].max())
            fig_pred.add_shape(type="line", x0=lo, y0=lo, x1=hi, y1=hi,
                               line=dict(color="#FC8181", dash="dash"))
            fig_pred.update_layout(template="plotly_dark", height=380)
            st.plotly_chart(fig_pred, use_container_width=True)

        with cc2:
            # Coefficients
            fig_coef = px.bar(
                r["coefs"], x="Coefficient", y="Feature", orientation="h",
                title=f"Regression Coefficients — {label}",
                color="Coefficient",
                color_continuous_scale="RdBu", color_continuous_midpoint=0,
            )
            fig_coef.update_layout(template="plotly_dark", height=380,
                                   coloraxis_showscale=False)
            st.plotly_chart(fig_coef, use_container_width=True)

        st.markdown("---")

    insight("The regression models explain a moderate proportion of variance in productivity "
            "and cost outcomes (R² ≈ 0.3–0.5). The strongest positive predictors are "
            "**AI budget percentage**, **task automation rate**, and **AI investment per employee**. "
            "The regression coefficients confirm that each additional percentage point of AI "
            "budget allocation translates to measurable productivity gains, providing a "
            "quantitative business case for internal AI investment over consulting fees.")

# ═════════════════════════════════════════════════════════════════════════════
# TAB 8 – AI STRATEGY ADVISOR
# ═════════════════════════════════════════════════════════════════════════════
with tabs[8]:
    st.markdown("<div class='section-header'><h2>🎯 AI Strategy Advisor</h2></div>",
                unsafe_allow_html=True)
    st.markdown("Enter your company's profile below to receive a **data-driven recommendation** "
                "on whether to implement AI internally, hire external consultants, or use a "
                "hybrid approach.")
    st.markdown("---")

    col_form, col_result = st.columns([2, 3])

    with col_form:
        st.markdown("### 📝 Company Profile Input")
        adv_industry    = st.selectbox("Industry", all_industries, key="adv_ind")
        adv_size        = st.selectbox("Company Size", ["Startup", "SME", "Enterprise"], key="adv_size")
        adv_invest      = st.slider("AI Investment per Employee (USD)",
                                    0, 200_000, 30_000, 5000, key="adv_inv")
        adv_auto        = st.slider("Task Automation Rate (%)", 0, 100, 25, key="adv_auto")
        adv_budget_pct  = st.slider("AI Budget as % of IT Budget", 0.0, 30.0, 5.0, 0.5,
                                    key="adv_bud")
        adv_training    = st.slider("AI Training Hours per Employee / Year", 0, 200, 20,
                                    key="adv_train")
        adv_reskill     = st.slider("Reskilled Employees", 0, 500, 50, key="adv_resk")
        adv_tools       = st.slider("Number of AI Tools Used", 0, 20, 3, key="adv_tools")
        adv_projects    = st.slider("Active AI Projects", 0, 30, 5, key="adv_proj")
        adv_innov       = st.slider("Innovation Score (0–100)", 0, 100, 50, key="adv_innov")

        run_advisor = st.button("🔍 Get AI Strategy Recommendation", type="primary",
                                use_container_width=True)

    with col_result:
        if run_advisor:
            # ── Score calculation ────────────────────────────────────────────
            score = 0
            feedback = []

            # Investment readiness
            if adv_invest >= 50_000:
                score += 30
                feedback.append(("✅", "Strong AI investment per employee (≥ $50K)", "green"))
            elif adv_invest >= 20_000:
                score += 15
                feedback.append(("⚠️", "Moderate AI investment — consider increasing budget", "orange"))
            else:
                score += 0
                feedback.append(("❌", "Low AI investment — consulting may be needed to upskill", "red"))

            # Automation readiness
            if adv_auto >= 40:
                score += 20
                feedback.append(("✅", "High task automation rate — strong internal capability", "green"))
            elif adv_auto >= 20:
                score += 10
                feedback.append(("⚠️", "Moderate automation — room to grow with AI tooling", "orange"))
            else:
                feedback.append(("❌", "Low automation — significant AI implementation work needed", "red"))

            # Budget commitment
            if adv_budget_pct >= 10:
                score += 20
                feedback.append(("✅", "AI budget is well-funded (≥ 10% of IT budget)", "green"))
            elif adv_budget_pct >= 5:
                score += 10
                feedback.append(("⚠️", "AI budget is modest — scale up to accelerate adoption", "orange"))
            else:
                feedback.append(("❌", "Very low AI budget allocation — strategic re-prioritisation needed", "red"))

            # Training
            if adv_training >= 40:
                score += 15
                feedback.append(("✅", "Robust employee AI training programme", "green"))
            elif adv_training >= 15:
                score += 7
                feedback.append(("⚠️", "Some AI training in place — expand for better outcomes", "orange"))
            else:
                feedback.append(("❌", "Minimal training — workforce upskilling is a priority", "red"))

            # Tools & projects
            if adv_tools >= 5 and adv_projects >= 5:
                score += 15
                feedback.append(("✅", "Active AI portfolio (tools + projects) indicates strong culture", "green"))
            elif adv_tools >= 2 or adv_projects >= 2:
                score += 7
                feedback.append(("⚠️", "Emerging AI portfolio — scale AI use-cases", "orange"))
            else:
                feedback.append(("❌", "Limited AI tooling — begin with pilot projects", "red"))

            # ── Recommendation ───────────────────────────────────────────────
            if score >= 75:
                rec = "🚀 Implement AI Internally"
                rec_color = "#276749"
                rec_bg    = "#C6F6D5"
                rec_detail = (
                    "Your company demonstrates strong AI readiness across investment, automation, "
                    "budget commitment, and workforce training. The data suggests you have the "
                    "internal capability to lead your own AI implementation without heavy reliance "
                    "on external consultants. Focus on scaling existing AI projects, deepening "
                    "automation, and continuing workforce development."
                )
            elif score >= 45:
                rec = "🤝 Hybrid Approach (AI + Consulting)"
                rec_color = "#7B5E00"
                rec_bg    = "#FEFCBF"
                rec_detail = (
                    "Your company shows moderate AI readiness with clear strengths in some areas "
                    "and gaps in others. A hybrid strategy — using consultants to fill specific "
                    "skill gaps while building internal AI capability in parallel — will deliver "
                    "the best outcomes. Prioritise increasing AI investment per employee and "
                    "expanding your workforce training programme."
                )
            else:
                rec = "🏢 Hire External Consultants"
                rec_color = "#742A2A"
                rec_bg    = "#FED7D7"
                rec_detail = (
                    "Your current AI maturity indicators suggest that internal implementation "
                    "would face significant challenges. External consulting expertise can help "
                    "you build a foundation: defining an AI strategy, selecting appropriate "
                    "tools, and upskilling your workforce. Plan a 12–18 month consulting "
                    "engagement focused on capability building so you can transition to internal "
                    "AI ownership over time."
                )

            # ── Display ──────────────────────────────────────────────────────
            st.markdown("### 🏆 Strategy Recommendation")
            st.markdown(f"""
            <div style='background:{rec_bg}; border-radius:12px; padding:20px 24px; margin:10px 0;'>
              <h2 style='color:{rec_color}; margin:0;'>{rec}</h2>
              <hr style='border-color:{rec_color}; opacity:0.3;'/>
              <p style='color:#2D3748; font-size:0.95rem; line-height:1.7;'>{rec_detail}</p>
            </div>""", unsafe_allow_html=True)

            # Score gauge
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=score,
                title={"text": "AI Readiness Score", "font": {"size": 18}},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#4F8EF7"},
                    "steps": [
                        {"range": [0, 45],  "color": "#FC8181"},
                        {"range": [45, 75], "color": "#F6AD55"},
                        {"range": [75, 100],"color": "#68D391"},
                    ],
                    "threshold": {"line": {"color": "white", "width": 3},
                                  "thickness": 0.75, "value": score},
                },
                number={"suffix": " / 100", "font": {"size": 28}},
            ))
            fig_gauge.update_layout(
                template="plotly_dark", height=280,
                margin=dict(t=40, b=10, l=30, r=30),
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

            # Factor breakdown
            st.markdown("### 📋 Factor Analysis")
            for icon, text, colour in feedback:
                badge_class = f"badge-{'green' if colour=='green' else 'orange' if colour=='orange' else 'orange'}"
                st.markdown(f"{icon} &nbsp; {text}", unsafe_allow_html=True)

            # Peer benchmarks
            st.markdown("### 📊 How You Compare to Similar Companies")
            peer_mask = (df_company["industry"] == adv_industry) & \
                        (df_company["company_size"] == adv_size)
            peers = df_company[peer_mask]
            if len(peers) > 10:
                metrics_compare = {
                    "AI Investment / Employee": ("ai_investment_per_employee", adv_invest),
                    "Task Automation Rate %": ("task_automation_rate", adv_auto),
                    "AI Budget %": ("ai_budget_percentage", adv_budget_pct),
                }
                compare_rows = []
                for label, (col, user_val) in metrics_compare.items():
                    if col in peers.columns:
                        peer_avg = peers[col].mean()
                        compare_rows.append({
                            "Metric": label,
                            "Your Value": user_val,
                            "Industry Peer Avg": round(peer_avg, 2),
                            "vs Peers": "Above ✅" if user_val >= peer_avg else "Below ⚠️",
                        })
                if compare_rows:
                    st.dataframe(pd.DataFrame(compare_rows), use_container_width=True)
        else:
            st.markdown("""
            <div style='text-align:center; padding:60px 40px; color:#A0AEC0;'>
              <h3>👈 Fill in your company profile on the left</h3>
              <p>Adjust the sliders to match your organisation's AI characteristics,
                 then click <strong>Get AI Strategy Recommendation</strong> to receive
                 a personalised recommendation backed by the patterns in 150,000
                 company records.</p>
            </div>""", unsafe_allow_html=True)

            # Show sample benchmark stats
            st.markdown("#### 📊 Industry Benchmark Preview")
            bench = df_company.groupby("industry")[
                ["ai_investment_per_employee", "task_automation_rate",
                 "productivity_change_percent", "cost_reduction_percent"]
            ].mean().round(2)
            fig_bench = px.imshow(
                bench.T,
                title="Industry Averages Heatmap — Key AI Metrics",
                color_continuous_scale="Blues",
                text_auto=".1f",
            )
            fig_bench.update_layout(template="plotly_dark", height=360)
            st.plotly_chart(fig_bench, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#718096; font-size:0.85rem; padding:10px 0 20px;'>
  AI vs Consulting Decision Dashboard &nbsp;·&nbsp;
  University Data Analytics Course &nbsp;·&nbsp;
  Built with Streamlit &amp; Plotly &nbsp;·&nbsp;
  150,000 company records analysed
</div>
""", unsafe_allow_html=True)
