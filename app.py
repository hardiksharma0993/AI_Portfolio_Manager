import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf
import time

from portfolio import (
    load_portfolio,
    get_current_prices,
    get_portfolio_history,
    get_total_value,
    get_metrics,
    get_sector_map
)

from llm import ask_llm


# ================================
# CONFIG
# ================================
st.set_page_config(page_title="Portfolio Intelligence", layout="wide")
st.title("📊 Pro Portfolio Intelligence Dashboard")


# ================================
# FORCE REFRESH (LIVE MODE CONTROL)
# ================================
st.sidebar.header("Controls")

force_refresh = st.sidebar.button("🔄 Refresh Live Prices")


# ================================
# LOAD DATA (LIVE PRICES ALWAYS REFRESHED)
# ================================
df = load_portfolio()

# ALWAYS REFRESH PRICES (NO CACHE ISSUES)
prices = get_current_prices(df)

sector_map = get_sector_map()

df["CurrentPrice"] = df["Ticker"].map(prices)
df["Value"] = df["CurrentPrice"] * df["Shares"]
df["Sector"] = df["Ticker"].map(sector_map).fillna("Other")
df["Weight"] = df["Value"] / df["Value"].sum()

total_value = df["Value"].sum()


# ================================
# METRICS
# ================================
metrics = get_metrics()

c1, c2, c3, c4 = st.columns(4)

c1.metric("Portfolio Value", f"₹{total_value:,.0f}")
c2.metric("Annual Return", f"{metrics['annual_return']:.2%}")
c3.metric("Sharpe Ratio", f"{metrics['sharpe']:.2f}")
c4.metric("Beta", f"{metrics['beta']:.2f}")

st.divider()


# ================================
# PORTFOLIO VS BENCHMARK (FIXED)
# ================================
portfolio = get_portfolio_history()

returns = portfolio.pct_change().dropna()

benchmark = yf.download("^NSEI", period="1y")["Close"]

aligned = pd.concat([portfolio, benchmark], axis=1).dropna()
aligned.columns = ["Portfolio", "NIFTY"]

portfolio_norm = aligned["Portfolio"] / aligned["Portfolio"].iloc[0]
nifty_norm = aligned["NIFTY"] / aligned["NIFTY"].iloc[0]

fig1 = go.Figure()

fig1.add_trace(go.Scatter(
    x=portfolio_norm.index,
    y=portfolio_norm.values,
    name="Portfolio"
))

fig1.add_trace(go.Scatter(
    x=nifty_norm.index,
    y=nifty_norm.values,
    name="NIFTY 50"
))

fig1.update_layout(
    title="Portfolio vs Benchmark (Normalized)",
    xaxis_title="Date",
    yaxis_title="Growth of ₹1",
)

st.plotly_chart(fig1, use_container_width=True)


# ================================
# ALLOCATION PIE
# ================================
fig2 = px.pie(df, names="Ticker", values="Value", title="Allocation")
st.plotly_chart(fig2, use_container_width=True)


# ================================
# SECTOR EXPOSURE
# ================================
sector_alloc = df.groupby("Sector")["Value"].sum()

fig3 = px.bar(
    x=sector_alloc.index,
    y=sector_alloc.values,
    labels={"x": "Sector", "y": "Exposure"},
    title="Sector Exposure"
)

st.plotly_chart(fig3, use_container_width=True)


# ================================
# HOLDINGS RANKING
# ================================
st.subheader("Holdings Ranking")

ranked = df.sort_values("Weight", ascending=False)

fig4 = px.bar(
    ranked,
    x="Ticker",
    y="Weight",
    text="Weight",
    title="Portfolio Weight Ranking"
)

fig4.update_traces(texttemplate='%{text:.2%}', textposition='outside')

fig4.update_layout(
    xaxis_title="Stock",
    yaxis_title="Weight in Portfolio"
)

st.plotly_chart(fig4, use_container_width=True)


# ================================
# HOLDINGS TABLE
# ================================
st.dataframe(df)


# ================================
# SMART INSIGHTS (FIXED — NO FREEZE, NO SILENT FAIL)
# ================================
st.subheader("Smart Insights")

if st.button("Generate Insights"):

    with st.spinner("Analyzing portfolio using AI..."):

        try:
            prompt = f"""
Portfolio Value: {total_value}
Sharpe: {metrics['sharpe']}
Beta: {metrics['beta']}
Annual Return: {metrics['annual_return']}

Holdings:
{df[['Ticker','Weight','Value','Sector']].to_string(index=False)}

Provide:
- Risk analysis
- Concentration risk
- Sector exposure issues
- Actionable improvements
"""

            # HARD TIME PROTECTION (prevents infinite hang)
            start = time.time()

            response = ask_llm(prompt)

            elapsed = time.time() - start

            if elapsed > 10:
                st.warning("⚠️ Response took longer than expected.")

            st.success(response)

        except Exception as e:
            st.error(f"Insights failed: {str(e)}")


# ================================
# FOOTNOTE
# ================================
st.caption("Live portfolio updates based on latest available market prices via Yahoo Finance.")