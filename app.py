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
# SIDEBAR CONTROLS
# ================================
st.sidebar.header("Controls")
force_refresh = st.sidebar.button("🔄 Refresh Live Prices")

st.sidebar.markdown("---")
st.sidebar.caption("Prices: Last available close (NSE). Updates reflect latest trading session.")


# ================================
# LOAD DATA
# ================================
df = load_portfolio()

with st.spinner("Fetching live prices..."):
    prices = get_current_prices(df)

sector_map = get_sector_map()

df["CurrentPrice"] = df["Ticker"].map(prices)
df["Value"] = df["CurrentPrice"] * df["Shares"]
df["Sector"] = df["Ticker"].map(sector_map).fillna("Other")
df["Weight"] = df["Value"] / df["Value"].sum()

total_value = df["Value"].sum()

# Check for zero/nan prices and warn
missing = df[df["CurrentPrice"].isna() | (df["CurrentPrice"] == 0)]["Ticker"].tolist()
if missing:
    st.warning(f"⚠️ Could not fetch prices for: {', '.join(missing)}. These are excluded from total value.")


# ================================
# METRICS
# ================================
with st.spinner("Computing portfolio metrics..."):
    metrics = get_metrics()
    capture = get_capture_ratios(
        portfolio_returns=metrics["returns"],
        benchmark_returns=metrics["benchmark_returns"]
    )

c1, c2, c3, c4 = st.columns(4)
c1.metric("Portfolio Value", f"₹{total_value:,.0f}")
c2.metric("Annual Return", f"{metrics['annual_return']:.2%}")
c3.metric("Sharpe Ratio", f"{metrics['sharpe']:.2f}")
c4.metric("Beta", f"{metrics['beta']:.2f}")

c5, c6, c7, c8 = st.columns(4)
c5.metric("Max Drawdown", f"{metrics['max_drawdown']:.2%}", delta_color="inverse")
c6.metric("Volatility (Ann.)", f"{metrics['volatility']:.2%}")
c7.metric(
    "Upside Capture",
    f"{capture['upside_capture']:.1f}%",
    help="% of benchmark upside the portfolio captured. >100% is ideal."
)
c8.metric(
    "Downside Capture",
    f"{capture['downside_capture']:.1f}%",
    delta_color="inverse",
    help="% of benchmark downside the portfolio suffered. <100% is ideal."
)

st.divider()


# ================================
# PORTFOLIO VS BENCHMARK
# ================================
with st.spinner("Loading historical data..."):
    portfolio = get_portfolio_history()

benchmark = yf.download("^NSEI", period="1y", auto_adjust=True, progress=False)["Close"]

# Flatten if needed
if isinstance(benchmark, pd.DataFrame):
    benchmark = benchmark.iloc[:, 0]

aligned = pd.concat([portfolio, benchmark], axis=1).dropna()
aligned.columns = ["Portfolio", "NIFTY"]

portfolio_norm = aligned["Portfolio"] / aligned["Portfolio"].iloc[0]
nifty_norm = aligned["NIFTY"] / aligned["NIFTY"].iloc[0]

fig1 = go.Figure()
fig1.add_trace(go.Scatter(
    x=portfolio_norm.index, y=portfolio_norm.values,
    name="Portfolio", line=dict(color="#00C897", width=2)
))
fig1.add_trace(go.Scatter(
    x=nifty_norm.index, y=nifty_norm.values,
    name="NIFTY 50", line=dict(color="#FF6B6B", width=2, dash="dash")
))
fig1.update_layout(
    title="Portfolio vs Benchmark (Normalized to ₹1)",
    xaxis_title="Date",
    yaxis_title="Growth of ₹1",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
st.plotly_chart(fig1, use_container_width=True)


# ================================
# DRAWDOWN CHART
# ================================
st.subheader("📉 Rolling Drawdown from Peak")

drawdown = get_drawdown_series(portfolio)

fig_dd = go.Figure()
fig_dd.add_trace(go.Scatter(
    x=drawdown.index,
    y=drawdown.values * 100,  # Convert to percentage
    fill="tozeroy",
    fillcolor="rgba(255, 80, 80, 0.25)",
    line=dict(color="red", width=1.5),
    name="Drawdown (%)"
))
fig_dd.update_layout(
    title="Portfolio Drawdown from Rolling Peak (%)",
    xaxis_title="Date",
    yaxis_title="Drawdown (%)",
    yaxis=dict(ticksuffix="%"),
    hovermode="x unified"
)
fig_dd.add_hline(
    y=float(metrics["max_drawdown"] * 100),
    line_dash="dot",
    line_color="darkred",
    annotation_text=f"Max DD: {metrics['max_drawdown']:.2%}",
    annotation_position="bottom right"
)
st.plotly_chart(fig_dd, use_container_width=True)


# ================================
# CAPTURE RATIO CHART
# ================================
st.subheader("📈 Upside / Downside Capture Ratios")

fig_cap = go.Figure()

categories = ["Upside Capture", "Downside Capture"]
values = [capture["upside_capture"], capture["downside_capture"]]
colors = [
    "#00C897" if capture["upside_capture"] >= 100 else "#FFA500",
    "#00C897" if capture["downside_capture"] <= 100 else "#FF6B6B"
]

fig_cap.add_trace(go.Bar(
    x=categories,
    y=values,
    marker_color=colors,
    text=[f"{v:.1f}%" for v in values],
    textposition="outside"
))

fig_cap.add_hline(
    y=100,
    line_dash="dash",
    line_color="gray",
    annotation_text="Benchmark (100%)",
    annotation_position="top right"
)

fig_cap.update_layout(
    title="Capture Ratios vs NIFTY 50 (1Y)",
    yaxis_title="Capture Ratio (%)",
    yaxis=dict(ticksuffix="%", range=[0, max(values) * 1.25 + 10]),
    xaxis_title="",
    showlegend=False
)
st.plotly_chart(fig_cap, use_container_width=True)

with st.expander("ℹ️ How to read Capture Ratios"):
    st.markdown("""
    - **Upside Capture > 100%** → Portfolio gained *more* than the benchmark on up days. ✅
    - **Downside Capture < 100%** → Portfolio lost *less* than the benchmark on down days. ✅
    - Ideal active portfolio: High upside capture + Low downside capture.
    - A ratio of exactly 100% means the portfolio perfectly tracked the benchmark in that regime.
    """)

st.divider()


# ================================
# ALLOCATION PIE
# ================================
fig2 = px.pie(df, names="Ticker", values="Value", title="Portfolio Allocation by Holding")
st.plotly_chart(fig2, use_container_width=True)


# ================================
# SECTOR EXPOSURE
# ================================
sector_alloc = df.groupby("Sector")["Value"].sum().reset_index()
sector_alloc.columns = ["Sector", "Value"]

fig3 = px.bar(
    sector_alloc,
    x="Sector",
    y="Value",
    color="Sector",
    title="Sector Exposure (₹)",
    labels={"Value": "Exposure (₹)"}
)
st.plotly_chart(fig3, use_container_width=True)


# ================================
# HOLDINGS RANKING
# ================================
st.subheader("Holdings Ranking by Weight")

ranked = df.sort_values("Weight", ascending=False)

fig4 = px.bar(
    ranked,
    x="Ticker",
    y="Weight",
    text="Weight",
    title="Portfolio Weight Ranking",
    color="Sector"
)
fig4.update_traces(texttemplate='%{text:.2%}', textposition='outside')
fig4.update_layout(
    xaxis_title="Stock",
    yaxis_title="Weight in Portfolio",
    yaxis=dict(tickformat=".0%")
)
st.plotly_chart(fig4, use_container_width=True)


# ================================
# HOLDINGS TABLE
# ================================
st.subheader("Holdings Detail")
display_df = df.copy()
display_df["CurrentPrice"] = display_df["CurrentPrice"].map(lambda x: f"₹{x:,.2f}" if pd.notna(x) else "N/A")
display_df["Value"] = display_df["Value"].map(lambda x: f"₹{x:,.0f}" if pd.notna(x) else "N/A")
display_df["Weight"] = display_df["Weight"].map(lambda x: f"{x:.2%}" if pd.notna(x) else "N/A")
st.dataframe(display_df, use_container_width=True)


# ================================
# SMART INSIGHTS (AI-POWERED)
# ================================
st.subheader("🤖 Smart Insights")

if st.button("Generate AI Insights"):
    with st.spinner("Analyzing portfolio using local LLM..."):
        try:
            prompt = f"""
You are an expert Indian equity portfolio analyst. Analyze the following portfolio and provide a structured assessment.

=== PORTFOLIO SNAPSHOT ===
Total Value: ₹{total_value:,.0f}
Annual Return: {metrics['annual_return']:.2%}
Sharpe Ratio: {metrics['sharpe']:.2f}
Beta vs NIFTY 50: {metrics['beta']:.2f}
Volatility (Ann.): {metrics['volatility']:.2%}
Max Drawdown: {metrics['max_drawdown']:.2%}
Upside Capture Ratio: {capture['upside_capture']:.1f}%
Downside Capture Ratio: {capture['downside_capture']:.1f}%

=== HOLDINGS ===
{df[['Ticker', 'Shares', 'CurrentPrice', 'Value', 'Weight', 'Sector']].to_string(index=False)}

=== PROVIDE ===
1. Risk Analysis (Beta, Drawdown, Volatility interpretation)
2. Capture Ratio Assessment (Is this portfolio protecting downside and capturing upside?)
3. Concentration Risk (Any stock or sector too dominant?)
4. Sector Exposure Issues
5. Three specific, actionable improvements
Keep it concise and analytical. Avoid generic advice.
"""
            start = time.time()
            response = ask_llm(prompt)
            elapsed = time.time() - start

            if elapsed > 30:
                st.warning("⚠️ Response took longer than expected.")

            st.success(response)

        except Exception as e:
            st.error(f"Insights failed: {str(e)}")


# ================================
# FOOTNOTE
# ================================
st.caption("Data via Yahoo Finance. Prices reflect the last available trading session close. Capture ratios computed vs NIFTY 50 on a 1-year daily return basis.")
