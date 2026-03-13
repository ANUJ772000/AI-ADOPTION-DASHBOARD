
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(
    page_title="AI vs Consulting Dashboard",
    page_icon="🤖",
    layout="wide"
)

# ------------------- DARK MODE STYLING -------------------
st.markdown("""
<style>
[data-testid="stAppViewContainer"]{
    background-color:#0E1117;
    color:#FAFAFA;
}
.kpi-card{
    background:#1A1F2E;
    border-radius:10px;
    padding:15px;
    text-align:center;
}
.kpi-value{
    font-size:2rem;
    color:#4F8EF7;
    font-weight:bold;
}
.kpi-label{
    font-size:0.9rem;
    color:#A0AEC0;
}
.section{
    border-left:4px solid #4F8EF7;
    padding-left:10px;
    margin-top:10px;
}
</style>
""", unsafe_allow_html=True)

# ------------------- LOAD DATA -------------------
@st.cache_data
def load_data():
    company = pd.read_csv("ai_company_adoption.csv")
    industry = pd.read_csv("ai_industry_summary.csv")
    country = pd.read_csv("country_ai_index.csv")
    return company, industry, country

df_company, df_industry, df_country = load_data()

# ------------------- SIDEBAR FILTERS -------------------
with st.sidebar:
    st.title("AI Strategy Dashboard")

    industries = sorted(df_company["industry"].dropna().unique())
    sizes = sorted(df_company["company_size"].dropna().unique())
    countries = sorted(df_company["country"].dropna().unique())

    sel_industries = st.multiselect("Industry", industries, default=industries)
    sel_sizes = st.multiselect("Company Size", sizes, default=sizes)

    country_mode = st.radio("Country Mode", ["All", "Select One"])

    if country_mode == "All":
        sel_countries = countries
    else:
        chosen = st.selectbox("Country", countries)
        sel_countries = [chosen]

    if st.button("Reset Filters"):
        st.rerun()

mask = (
    df_company["industry"].isin(sel_industries) &
    df_company["company_size"].isin(sel_sizes) &
    df_company["country"].isin(sel_countries)
)

df_filtered = df_company[mask]

# ------------------- KPI FUNCTION -------------------
def kpi(label,value):
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
    </div>
    """, unsafe_allow_html=True)

# ------------------- TABS -------------------
tabs = st.tabs([
"Overview",
"Global AI Landscape",
"Industry Insights",
"Company Analysis",
"AI Strategy Advisor",
"What‑If Simulator"
])

# ------------------- OVERVIEW -------------------
with tabs[0]:
    st.title("AI vs Consulting Decision Dashboard")

    c1,c2,c3 = st.columns(3)

    with c1:
        kpi("Companies Analysed", len(df_filtered))

    with c2:
        if "ai_adoption_rate" in df_filtered:
            kpi("Avg AI Adoption %", round(df_filtered["ai_adoption_rate"].mean(),2))

    with c3:
        if "productivity_change_percent" in df_filtered:
            kpi("Avg Productivity Gain %", round(df_filtered["productivity_change_percent"].mean(),2))

    st.markdown("""
This dashboard helps companies decide whether to **implement AI internally** or **hire consultants**.
The analysis explores global trends, industry patterns, and company‑level data to guide strategy.
""")

# ------------------- GLOBAL LANDSCAPE -------------------
with tabs[1]:
    st.markdown('<div class="section"><h2>Global AI Landscape</h2></div>', unsafe_allow_html=True)

    if "ai_patent_filings_2024" in df_country.columns:
        fig = px.bar(
            df_country,
            x="country",
            y="ai_patent_filings_2024",
            color="region",
            template="plotly_dark",
            title="AI Patent Filings by Country"
        )
        st.plotly_chart(fig, width="stretch")

# ------------------- INDUSTRY INSIGHTS -------------------
with tabs[2]:
    st.markdown('<div class="section"><h2>Industry AI Adoption</h2></div>', unsafe_allow_html=True)

    if "avg_ai_adoption_rate" in df_industry.columns:
        fig = px.bar(
            df_industry,
            x="industry",
            y="avg_ai_adoption_rate",
            color="avg_ai_adoption_rate",
            template="plotly_dark",
            title="AI Adoption by Industry"
        )
        st.plotly_chart(fig, width="stretch")

# ------------------- COMPANY ANALYSIS -------------------
with tabs[3]:
    st.markdown('<div class="section"><h2>Company Data Exploration</h2></div>', unsafe_allow_html=True)

    if len(df_filtered) > 0:
        sample = df_filtered.sample(min(3000,len(df_filtered)))

        fig = px.scatter(
            sample,
            x="ai_investment_per_employee",
            y="productivity_change_percent",
            color="industry",
            opacity=0.6,
            template="plotly_dark",
            title="AI Investment vs Productivity Gain"
        )

        st.plotly_chart(fig, width="stretch")

# ------------------- TRAIN MODEL -------------------
@st.cache_data
def train_model():
    df = df_company.dropna()
    features = ["ai_investment_per_employee","task_automation_rate","ai_budget_percentage"]
    target = "ai_adoption_stage"

    df = df[features+[target]]

    X = df[features]
    y = df[target]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestClassifier()
    model.fit(X_scaled,y)

    return model, scaler

model, scaler = train_model()

# ------------------- AI STRATEGY ADVISOR -------------------
with tabs[4]:
    st.markdown('<div class="section"><h2>AI Strategy Advisor</h2></div>', unsafe_allow_html=True)

    industry = st.selectbox("Industry", industries)
    size = st.selectbox("Company Size", sizes)
    country = st.selectbox("Country", countries)

    invest = st.slider("AI Investment per Employee",0,200000,30000)
    automation = st.slider("Automation Level %",0,100,30)
    budget = st.slider("AI Budget %",0,30,5)

    if st.button("Get Recommendation"):

        X = np.array([[invest,automation,budget]])
        X_scaled = scaler.transform(X)

        prediction = model.predict(X_scaled)[0]
        confidence = model.predict_proba(X_scaled).max()

        st.subheader(f"Recommendation: {prediction}")
        st.metric("Confidence", f"{confidence*100:.1f}%")

# ------------------- WHAT IF SIMULATOR -------------------
with tabs[5]:
    st.markdown('<div class="section"><h2>AI Investment Simulator</h2></div>', unsafe_allow_html=True)

    invest = st.slider("AI Investment",0,200000,20000)
    automation = st.slider("Automation Level",0,100,20)
    training = st.slider("Training Hours",0,200,20)

    productivity = (invest*0.0002)+(automation*0.3)+(training*0.05)
    cost = (invest*0.00015)+(automation*0.25)+(training*0.04)

    c1,c2 = st.columns(2)

    c1.metric("Predicted Productivity Gain %", round(productivity,2))
    c2.metric("Predicted Cost Reduction %", round(cost,2))

    fig = px.scatter(
        x=[invest],
        y=[productivity],
        size=[automation],
        template="plotly_dark",
        title="Investment vs Productivity Simulation"
    )

    st.plotly_chart(fig, width="stretch")

st.markdown("---")
st.caption("AI Strategy Dashboard • Data Analytics Project")
