import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
from sklearn.metrics import r2_score

st.set_page_config(page_title="Bank Analytics Dashboard", layout="wide")
st.title("📊 Bank Profitability & ESG Analytics Dashboard")

df = pd.read_csv("Indian_Banks.csv")
df.columns = df.columns.str.strip()

model = pickle.load(open("model.pkl", "rb"))
esg_model = pickle.load(open("esg_model.pkl", "rb"))

features = [
    "Net NPA\nRatio (%)",
    "CAR / CRAR\n(%)",
    "Credit Growth\n(%)",
    "Cost-to-\nIncome\nRatio (%)",
    "Bank Size\n[Log(Assets)]",
    "NPA × CAR\n[Interaction]",
    "NPA × Size\n[Interaction]"
]

X = df[features]
y = df["ROA (%)\n[DV]"]

y_pred = model.predict(X)
r2 = r2_score(y, y_pred)

tab1, tab2 = st.tabs(["📊 Dashboard", "🤖 Prediction"])

with tab1:
    banks = st.multiselect("Select Banks", df["Bank Name"].unique(),
                           default=df["Bank Name"].unique()[:2])

    filtered_df = df[df["Bank Name"].isin(banks)]

    col1, col2, col3 = st.columns(3)
    col1.metric("Avg ROA", round(filtered_df["ROA (%)\n[DV]"].mean(), 2))
    col2.metric("Avg NPA", round(filtered_df["Net NPA\nRatio (%)"].mean(), 2))
    col3.metric("R² Score", round(r2, 3))

    fig = px.line(filtered_df, x="Year", y="ROA (%)\n[DV]",
                  color="Bank Name", markers=True)
    st.plotly_chart(fig)

with tab2:
    npa = st.number_input("NPA")
    car = st.number_input("CAR")
    credit = st.number_input("Credit Growth")
    cost_income = st.number_input("Cost Income")
    size = st.number_input("Size")

    npa_car = npa * car
    npa_size = npa * size

    if st.button("Predict ROA"):
        pred = model.predict([[npa, car, credit, cost_income, size, npa_car, npa_size]])
        st.success(f"ROA: {pred[0]}")

    esg = st.slider("ESG Score", 0, 100)
    gov = st.slider("Governance", 0, 100)
    env = st.slider("Environmental", 0, 100)
    soc = st.slider("Social", 0, 100)

    if st.button("Predict ESG"):
        esg_pred = esg_model.predict([[esg, gov, env, soc]])
        st.success(esg_pred[0])
