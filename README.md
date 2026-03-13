# AI vs Consulting: A Data-Driven Decision Support Dashboard for AI Strategy

## Overview
An interactive Streamlit analytics dashboard that helps organizations decide whether to implement AI solutions internally or hire external consultants. Built for a university Data Analytics course, the dashboard demonstrates classification, clustering, association rule mining, and regression techniques on real-world AI adoption data.

## Live Demo
Deploy to [Streamlit Cloud](https://streamlit.io/cloud) for instant access.

## Dataset
Three datasets power this dashboard:
| File | Description | Rows |
|------|-------------|------|
| `ai_company_adoption.csv` | Company-level AI adoption survey data | 150,000 |
| `ai_industry_summary.csv` | Industry-level aggregated AI metrics | 9 |
| `country_ai_index.csv` | Country-level AI ecosystem indicators | 30 |

## Dashboard Sections
1. **Introduction** – Project objectives and dataset overview
2. **Global AI Landscape** – Country-level AI investment and innovation
3. **Industry Analysis** – Sector-by-sector AI adoption patterns
4. **Company EDA** – Exploratory data analysis with interactive charts
5. **Classification Models** – Six ML models predicting AI adoption stage
6. **Clustering Analysis** – K-Means segmentation of company profiles
7. **Association Rule Mining** – Frequent pattern discovery
8. **Regression Analysis** – Predicting productivity and cost outcomes
9. **AI Strategy Advisor** – Interactive decision-support tool

## Setup & Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deployment (Streamlit Cloud)
1. Push this folder to a public GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo and select `app.py` as the entry point
4. Click **Deploy**

## Tech Stack
- **Frontend**: Streamlit
- **Visualizations**: Plotly
- **Machine Learning**: scikit-learn, mlxtend
- **Data**: pandas, numpy

## Course Context
This project was developed as part of a university Data Analytics course to demonstrate the application of data science techniques to real-world business strategy decisions.
