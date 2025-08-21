import streamlit as st
import pandas as pd

st.set_page_config(page_title="Telco Churn Dashboard", layout="wide")

df = pd.read_csv("data/processed/telco_churn_scored.csv")

# Reconstruct contract for filters (works whether dummies are bool or 0/1)
df["Contract"] = "Month-to-month"
if "Contract_One year" in df.columns:
    one = df["Contract_One year"]
    df.loc[(one==1) | (one==True), "Contract"] = "One year"
if "Contract_Two year" in df.columns:
    two = df["Contract_Two year"]
    df.loc[(two==1) | (two==True), "Contract"] = "Two year"

st.title("📉 Telco Customer Churn")
col1, col2, col3 = st.columns(3)
col1.metric("Avg Churn Prob", f"{df['Churn_Prob'].mean():.2%}")
col2.metric("Customers", f"{len(df):,}")
col3.metric("Observed Churn Rate", f"{df['Churn'].mean():.2%}" if "Churn" in df.columns else "—")

# Filters
with st.sidebar:
    st.header("Filters")
    contract_opts = sorted(df["Contract"].dropna().unique().tolist())
    selected_contracts = st.multiselect("Contract", options=contract_opts, default=contract_opts)
    min_prob, max_prob = st.slider("Churn probability range", 0.0, 1.0, (0.0, 1.0), 0.01)

df_f = df[df["Contract"].isin(selected_contracts)]
df_f = df_f[(df_f["Churn_Prob"] >= min_prob) & (df_f["Churn_Prob"] <= max_prob)]

st.subheader("Churn Probability by Contract")
st.bar_chart(df_f.groupby("Contract")["Churn_Prob"].mean().rename("Avg churn prob"))

st.subheader("Top 20 Highest-Risk Customers")
show_cols = [c for c in ["customerID","Contract","MonthlyCharges","tenure","Churn_Prob"] if c in df_f.columns]
st.dataframe(df_f.sort_values("Churn_Prob", ascending=False)[show_cols].head(20), use_container_width=True)
