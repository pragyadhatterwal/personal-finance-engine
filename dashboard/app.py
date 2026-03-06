# dashboard/app.py

import os
import io
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import requests
import json

DATA_PATH = os.path.join("data", "transactions_sample.csv")
API_BASE_URL = "http://localhost:8000"

st.set_page_config(page_title="Personal Finance Insight Engine", layout="wide")

st.title("💸 Personal Finance Insight Engine")
st.caption("MVP dashboard — upload transactions, view spend insights, and spot patterns.")

# File uploader to append more transactions
st.sidebar.header("📤 Upload Transactions CSV")
uploaded = st.sidebar.file_uploader("Upload CSV with the same columns", type=["csv"])
if uploaded is not None:
    try:
        new_df = pd.read_csv(uploaded)
        st.sidebar.success(f"Loaded {len(new_df)} new rows.")
        # Append to disk
        if os.path.exists(DATA_PATH):
            base_df = pd.read_csv(DATA_PATH)
            merged = pd.concat([base_df, new_df], ignore_index=True)
        else:
            merged = new_df
        merged.to_csv(DATA_PATH, index=False)
        st.sidebar.success("Appended to data/transactions_sample.csv")
    except Exception as e:
        st.sidebar.error(f"Failed to read CSV: {e}")

# Load dataset
if not os.path.exists(DATA_PATH):
    st.warning("No data found yet. Upload a CSV to get started.")
    st.stop()

df = pd.read_csv(DATA_PATH)
if df.empty:
    st.warning("Your dataset is empty.")
    st.stop()

# Basic cleaning
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
df = df.dropna(subset=["date", "amount"])

# Filters
col1, col2, col3 = st.columns(3)
with col1:
    min_date = df["date"].min()
    max_date = df["date"].max()
    date_range = st.date_input("Date range", (min_date, max_date))
with col2:
    tx_type = st.multiselect("Type", sorted(df["type"].dropna().unique().tolist()), default=None)
with col3:
    categories = st.multiselect("Category", sorted(df.get("category", pd.Series([])).dropna().unique().tolist()), default=None)

mask = (df["date"].dt.date >= date_range[0]) & (df["date"].dt.date <= date_range[1])
if tx_type:
    mask &= df["type"].isin(tx_type)
if categories:
    mask &= df["category"].isin(categories)

vdf = df[mask].copy()

st.subheader("📊 Summary")
c1, c2, c3, c4 = st.columns(4)
total_spend = vdf.loc[vdf["type"].str.lower() == "debit", "amount"].sum()
total_in = vdf.loc[vdf["type"].str.lower() == "credit", "amount"].sum()
net = total_in - total_spend
n_tx = len(vdf)

c1.metric("Total Debits (₹)", f"{total_spend:,.2f}")
c2.metric("Total Credits (₹)", f"{total_in:,.2f}")
c3.metric("Net (₹)", f"{net:,.2f}")
c4.metric("Transactions", f"{n_tx}")

st.divider()

# Category chart
if "category" in vdf.columns and vdf["category"].notna().any():
    st.subheader("🍱 Spend by Category")
    cat = vdf[vdf["type"].str.lower() == "debit"].groupby("category")["amount"].sum().sort_values(ascending=False)
    st.bar_chart(cat)
else:
    st.info("No categories yet. Train a model and annotate some rows to enable category charts.")

# Monthly trend
st.subheader("📈 Monthly Trend")
vdf["month"] = vdf["date"].dt.to_period("M").astype(str)
trend = vdf.groupby(["month", "type"])["amount"].sum().reset_index()
pivot = trend.pivot(index="month", columns="type", values="amount").fillna(0)
st.line_chart(pivot)

# Top merchants
st.subheader("🏪 Top Merchants (Debits)")
top_merchants = (
    vdf[vdf["type"].str.lower() == "debit"]
    .groupby("merchant")["amount"].sum().sort_values(ascending=False).head(10)
)
st.dataframe(top_merchants.reset_index().rename(columns={"amount": "Total Spent (₹)"}))

# API Testing Section
st.divider()
st.subheader("🤖 Test Transaction Classification API")

# API Health Check
try:
    health_response = requests.get(f"{API_BASE_URL}/health", timeout=2)
    if health_response.status_code == 200:
        st.success("✅ API Server is running!")
    else:
        st.error("❌ API Server error")
except Exception:
    st.error("❌ API Server is not accessible. Make sure it's running on port 8000.")

# Transaction Classification Test
st.write("**Test Transaction Classification:**")
col1, col2, col3 = st.columns(3)

with col1:
    test_merchant = st.text_input("Merchant", value="Starbucks Coffee", key="test_merchant")
with col2:
    test_raw = st.text_input("Raw Description", value="STARBUCKS STORE #1234", key="test_raw") 
with col3:
    test_type = st.selectbox("Type", ["debit", "credit"], key="test_type")

if st.button("🔍 Classify Transaction"):
    try:
        classification_data = {
            "merchant": test_merchant,
            "raw": test_raw,
            "type": test_type
        }
        
        response = requests.post(
            f"{API_BASE_URL}/classify", 
            json=classification_data,
            timeout=5
        )
        
        if response.status_code == 200:
            result = response.json()
            
            col1, col2 = st.columns(2)
            with col1:
                if result.get("category"):
                    st.info(f"**Category:** {result['category']}")
                else:
                    st.warning("No category predicted")
            
            with col2:
                if result.get("confidence"):
                    confidence_pct = result['confidence'] * 100
                    st.info(f"**Confidence:** {confidence_pct:.1f}%")
                
            if result.get("message"):
                st.warning(result["message"])
            if result.get("error"):
                st.error(f"API Error: {result['error']}")
        else:
            st.error(f"API request failed: {response.status_code}")
            
    except Exception as e:
        st.error(f"Failed to connect to API: {str(e)}")

# Show raw data
st.divider()
st.subheader("📄 Raw Transaction Data")
if st.checkbox("Show all transaction data"):
    st.dataframe(vdf)
