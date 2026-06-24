import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf
import time

from portfolio import (
    load_portfolio,
    get_current_prices,
    get_portfolio_history,
    get_metrics,
    get_sector_map,
    get_drawdown_series,
    get_capture_ratios
)

from llm import ask_llm


# ================================
# CONFIG
# ================================
st.set_page_config(page_title="Portfolio Intelligence", layout="wide")
st.title("📊 Pro Portfolio Intelligence Dashboard")


# ================================
# SAFE LOAD PORTFOLIO
# ================================
try:
    df = load_portfolio()
except Exception as e:
    st.error(f"❌ Failed to load portfolio.csv: {e}")
    st.stop()


# ================================
# SAFE PRICE FETCH
# ================================
sector_map = get_sector_map()

try:
    with st.spinner("Fetching live prices..."):
        prices = get_current_prices(df)
except Exception as e:
    st.warning(f"⚠️ Price fetch failed: {e}")
    prices = {}

df["CurrentPrice"] = df["Ticker"].map(prices)
df["Value"] = df["CurrentPrice"] * df["Shares"]
df["Sector"] = df["Ticker"].map(sector_map).fillna("Other")
df["Weight"] = df["Value"] / df["Value"].sum()

total_value = df["Value"].sum()

missing = df[df["CurrentPrice"].isna() | (df["CurrentPrice"] == 0)]["Ticker"].tolist()
if missing:
    st.warning(f"⚠️ Missing prices: {', '.join(missing)}")


# ================================
# SAFE METRICS
# ================================
try:
    metrics = get_metrics()
except Exception as e:
    st.error(f"❌ Metrics calculation failed: {e}")
    st.stop()

capture = get_capture_ratios(
    portfolio_returns=metrics["returns"],
    benchmark_returns=metrics["benchmark_returns"]
)


# ================================
# DASHBOARD METRICS
# ================================
c1, c2, c3, c4 = st.columns(4)
c1.metric("Portfolio Value", f"₹{total_value:,.0f}")
c2.metric("Annual Return", f"{metrics['annual_return']:.2%}")
c3.metric("Sharpe Ratio", f"{metrics['sharpe']:.2f}")
c4.metric("Beta", f"{metrics['beta']:.2f}")

c5, c6, c7, c8 = st.columns(4)
c5.metric("Max Drawdown", f"{metrics['max_drawdown']:.2%}", delta_color="inverse")
c6.metric("Volatility", f"{metrics['volatility']:.2%}")
c7.metric("Upside Capture", f"{capture['upside_capture']:.1f}%")
c8.metric("Downside Capture", f"{capture['downside_capture']:.1f}%")


st.divider()


# ================================
# PORTFOLIO VS BENCHMARK
# ================================
portfolio = get_portfolio_history()

try:
    benchmark_data = yf.download("^NSEI", period="1y", auto_adjust=True, progress=False)
    benchmark = benchmark_data["Close"]
except Exception as e:
    st.error(f"❌ Benchmark fetch failed: {e}")
    st.stop()

aligned = pd.concat([portfolio, benchmark], axis=1).dropna()
aligned.columns = ["Portfolio", "NIFTY"]

portfolio_norm = aligned["Portfolio"] / aligned["Portfolio"].iloc[0]
nifty_norm = aligned["NIFTY"] / aligned["NIFTY"].iloc[0]

fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=portfolio_norm.index, y=portfolio_norm.values, name="Portfolio"))
fig1.add_trace(go.Scatter(x=nifty_norm.index, y=nifty_norm.values, name="NIFTY"))
st.plotly_chart(fig1, use_container_width=True)


# ================================
# DRAWDOWN
# ================================
drawdown = get_drawdown_series(portfolio)

fig_dd = go.Figure()
fig_dd.add_trace(go.Scatter(
    x=drawdown.index,
    y=drawdown.values * 100,
    fill="tozeroy",
    name="Drawdown"
))
st.plotly_chart(fig_dd, use_container_width=True)


# ================================
# ALLOCATION
# ================================
fig2 = px.pie(df, names="Ticker", values="Value")
st.plotly_chart(fig2, use_container_width=True)


# ================================
# SECTOR
# ================================
sector_alloc = df.groupby("Sector")["Value"].sum().reset_index()

fig3 = px.bar(sector_alloc, x="Sector", y="Value")
st.plotly_chart(fig3, use_container_width=True)


# ================================
# HOLDINGS TABLE
# ================================
st.subheader("Holdings")
st.dataframe(df)


# ================================
# AI INSIGHTS
# ================================
st.subheader("🤖 AI Insights")

if st.button("Generate AI Insights"):
    try:
        prompt = f"""
Portfolio Value: ₹{total_value:,.0f}
Return: {metrics['annual_return']:.2%}
Sharpe: {metrics['sharpe']:.2f}
Beta: {metrics['beta']:.2f}
Max Drawdown: {metrics['max_drawdown']:.2%}

Holdings:
{df[['Ticker','Shares','Value','Weight','Sector']].to_string(index=False)}

Give 3 key insights.
"""
        response = ask_llm(prompt)
        st.success(response)
    except Exception as e:
        st.error(f"AI failed: {e}")


st.caption("Powered by Yahoo Finance + Streamlit")