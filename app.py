
# Improved AI Strategy Dashboard (enhanced UI + advisor + what‑if simulator)
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression

st.set_page_config(
    page_title="AI vs Consulting Dashboard",
    page_icon="🤖",
    layout="wide"
)

# -------------------- THEME / DARK MODE STYLE --------------------
st.markdown("""
<style>
[data-testid="stAppViewContainer"]{
 background-color:#0E1117;
 color:#FAFAFA;
}
.kpi-card{
 background:#1A1F2E;
 border-radius:12px;
 padding:20px;
 text-align:center;
 margin-bottom:10px;
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
.section-title{
 border-left:4px solid #4F8EF7;
 padding-left:10px;
 margin-top:10px;
}
</style>
""", unsafe_allow_html=True)

# -------------------- LOAD DATA --------------------
@st.cache_data
def load_data():
    company = pd.read_csv("ai_company_adoption.csv")
    industry = pd.read_csv("ai_industry_summary.csv")
    country = pd.read_csv("country_ai_index.csv")
    return company, industry, country

df_company, df_industry, df_country = load_data()

# -------------------- SIDEBAR FILTERS --------------------
with st.sidebar:
    st.title("AI Strategy Dashboard")
    st.markdown("### Filters")

    industries = sorted(df_company["industry"].dropna().unique())
    sizes = sorted(df_company["company_size"].dropna().unique())
    countries = sorted(df_company["country"].dropna().unique())

    sel_industries = st.multiselect("Industry", industries, default=industries)

    sel_sizes = st.multiselect("Company Size", sizes, default=sizes)

    country_mode = st.radio("Country Selection", ["All", "Select One"])

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

# -------------------- KPI CARDS --------------------
def kpi(label,value):
    st.markdown(f"""
    <div class="kpi-card">
    <div class="kpi-label">{label}</div>
    <div class="kpi-value">{value}</div>
    </div>
    """,unsafe_allow_html=True)

# -------------------- TABS --------------------
tabs = st.tabs([
"Overview",
"Global Landscape",
"Industry Insights",
"Company EDA",
"AI Strategy Advisor",
"What‑If Simulator"
])

# -------------------- OVERVIEW --------------------
with tabs[0]:
    st.title("AI vs Consulting: Decision Dashboard")

    c1,c2,c3 = st.columns(3)
    with c1:
        kpi("Companies",len(df_filtered))
    with c2:
        kpi("Avg AI Adoption %",round(df_filtered["ai_adoption_rate"].mean(),2))
    with c3:
        kpi("Avg Productivity Gain %",round(df_filtered["productivity_change_percent"].mean(),2))

    st.markdown("""
This dashboard helps companies decide whether they should **implement AI internally or hire consultants**.
Data analytics and machine learning techniques analyze patterns across industries and countries to guide strategy.
""")

# -------------------- GLOBAL LANDSCAPE --------------------
with tabs[1]:
    st.markdown('<div class="section-title"><h2>Global AI Landscape</h2></div>',unsafe_allow_html=True)

    fig = px.bar(df_country,x="country",y="ai_patent_filings_2024",
                 color="region",template="plotly_dark",
                 title="AI Patent Filings by Country")
    st.plotly_chart(fig,use_container_width=True)

# -------------------- INDUSTRY --------------------
with tabs[2]:
    st.markdown('<div class="section-title"><h2>Industry AI Adoption</h2></div>',unsafe_allow_html=True)

    fig = px.bar(df_industry,
                 x="industry",
                 y="avg_ai_adoption_rate",
                 color="avg_ai_adoption_rate",
                 template="plotly_dark")
    st.plotly_chart(fig,use_container_width=True)

# -------------------- COMPANY EDA --------------------
with tabs[3]:
    st.markdown('<div class="section-title"><h2>Company Data Exploration</h2></div>',unsafe_allow_html=True)

    sample = df_filtered.sample(min(3000,len(df_filtered)))

    fig = px.scatter(sample,
                     x="ai_investment_per_employee",
                     y="productivity_change_percent",
                     color="industry",
                     template="plotly_dark")
    st.plotly_chart(fig,use_container_width=True)

# -------------------- TRAIN CLASSIFIER FOR ADVISOR --------------------
@st.cache_data
def train_model():
    df = df_company.sample(10000)
    features=["ai_investment_per_employee","task_automation_rate","ai_budget_percentage"]
    df=df.dropna(subset=features+["ai_adoption_stage"])
    X=df[features]
    y=df["ai_adoption_stage"]
    scaler=StandardScaler()
    Xs=scaler.fit_transform(X)
    model=RandomForestClassifier()
    model.fit(Xs,y)
    return model,scaler,features

model,scaler,features=train_model()

# -------------------- AI STRATEGY ADVISOR --------------------
with tabs[4]:
    st.markdown('<div class="section-title"><h2>AI Strategy Advisor</h2></div>',unsafe_allow_html=True)

    industry=st.selectbox("Industry",industries)
    size=st.selectbox("Company Size",sizes)
    country=st.selectbox("Country",countries)

    invest=st.slider("AI Investment per Employee",0,200000,30000)
    auto=st.slider("Automation Level",0,100,30)
    budget=st.slider("AI Budget %",0,30,5)

    if st.button("Get Recommendation"):

        X=np.array([[invest,auto,budget]])
        X=scaler.transform(X)

        pred=model.predict(X)[0]
        prob=max(model.predict_proba(X)[0])

        st.subheader(f"Prediction: {pred}")
        st.metric("Confidence",f"{prob*100:.1f}%")

        if prob>0.7:
            st.success("High confidence AI strategy recommendation.")
        else:
            st.warning("Moderate confidence prediction.")

# -------------------- WHAT IF SIMULATOR --------------------
with tabs[5]:
    st.markdown('<div class="section-title"><h2>What‑If AI Investment Simulator</h2></div>',unsafe_allow_html=True)

    invest=st.slider("AI Investment",0,200000,20000)
    auto=st.slider("Automation Level",0,100,20)
    train=st.slider("Training Hours",0,200,20)

    predicted_productivity = (invest*0.0002)+(auto*0.3)+(train*0.05)
    predicted_cost=(invest*0.00015)+(auto*0.25)+(train*0.04)

    col1,col2=st.columns(2)
    col1.metric("Predicted Productivity Gain %",round(predicted_productivity,2))
    col2.metric("Predicted Cost Reduction %",round(predicted_cost,2))

    fig=px.scatter(x=[invest],y=[predicted_productivity],
                   size=[auto],template="plotly_dark",
                   title="Investment vs Productivity Simulation")
    st.plotly_chart(fig,use_container_width=True)

st.markdown("---")
st.caption("AI Strategy Dashboard • University Data Analytics Project")
