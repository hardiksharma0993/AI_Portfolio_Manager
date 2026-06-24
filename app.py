import streamlit as st
import pandas as pd
import plotly.express as px

from portfolio import (
    load_portfolio,
    get_current_prices,
    get_portfolio_history,
    get_metrics,
    get_sector_map,
    get_drawdown_series,
    get_capture_ratios
)


# -------------------------
# APP CONFIG
# -------------------------
st.set_page_config(page_title="Portfolio Dashboard", layout="wide")

st.title("📊 AI Portfolio Dashboard")


# -------------------------
# LOAD DATA
# -------------------------
portfolio = load_portfolio()

tickers = portfolio["Ticker"].tolist()
prices = get_current_prices(tickers)

metrics = get_metrics(portfolio, prices)


# -------------------------
# SAFETY CHECKS
# -------------------------
if not prices:
    st.warning("⚠️ No prices fetched")

if "total_value" not in metrics:
    st.error("Metrics calculation failed")
    st.stop()


# -------------------------
# KPIs
# -------------------------
col1, col2, col3 = st.columns(3)

col1.metric("Portfolio Value", f"${metrics['total_value']:.2f}")
col2.metric("Total PnL", f"${metrics['total_pnl']:.2f}")
col3.metric("Holdings", len(portfolio))


# -------------------------
# HOLDINGS TABLE
# -------------------------
st.subheader("Holdings")

st.dataframe(metrics["holdings"], use_container_width=True)


# -------------------------
# PORTFOLIO HISTORY
# -------------------------
st.subheader("Portfolio History")

history = get_portfolio_history()

fig = px.line(history, x="Date", y="Value", title="Portfolio Value Over Time")
st.plotly_chart(fig, use_container_width=True)


# -------------------------
# DRAWDOWN
# -------------------------
st.subheader("Drawdown")

drawdown = get_drawdown_series(history)

fig2 = px.area(drawdown, title="Drawdown Curve")
st.plotly_chart(fig2, use_container_width=True)


# -------------------------
# SECTOR ALLOCATION
# -------------------------
st.subheader("Sector Map")

sector_map = get_sector_map()

st.json(sector_map)


# -------------------------
# CAPTURE RATIOS
# -------------------------
st.subheader("Capture Ratios")

st.json(get_capture_ratios())