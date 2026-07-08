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
    get_metrics,
    get_sector_map,
    get_drawdown_series,
    get_capture_ratios,
    run_backtest,
    compute_series_stats,
    run_optimizer,
    get_correlation_matrix,
    BENCHMARK_OPTIONS
)
import datetime

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
# CORRELATION MATRIX
# ================================
st.subheader("🧬 Holdings Correlation Matrix")
st.caption("Daily-return correlation over the trailing 1Y. Lower correlations between holdings mean more genuine diversification benefit.")

corr_matrix = get_correlation_matrix(period="1y")
if corr_matrix is not None:
    fig_corr = px.imshow(
        corr_matrix,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        aspect="auto",
        title="Pairwise Correlation of Daily Returns"
    )
    fig_corr.update_layout(coloraxis_colorbar=dict(title="Corr"))
    st.plotly_chart(fig_corr, use_container_width=True)
else:
    st.info("Need at least 2 holdings with overlapping price history to compute a correlation matrix.")


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

st.divider()


# ================================
# BACKTEST ENGINE
# ================================
st.subheader("🔁 Backtest Engine")
st.caption("Simulates your holdings' trailing history: rebalanced vs buy-and-hold, with configurable filters.")

all_tickers = df["Ticker"].tolist()

bt_col1, bt_col2, bt_col3 = st.columns(3)
with bt_col1:
    rebalance_freq = st.selectbox("Rebalancing Frequency", ["None", "Weekly", "Monthly", "Quarterly"], index=2)
with bt_col2:
    txn_cost = st.slider("Transaction Cost (bps per rebalance)", 0, 50, 10)
with bt_col3:
    rebalance_mode_label = st.selectbox(
        "Rebalancing Mode",
        ["Fixed Target Weights", "Walk-Forward Optimized (Max Sharpe)"],
        index=0,
        help="Fixed: always rebalances back to your current live weights. "
             "Walk-Forward: re-solves for max-Sharpe weights using only trailing data available at each rebalance date."
    )
rebalance_mode = "optimized" if "Walk-Forward" in rebalance_mode_label else "fixed"

bt_col4, bt_col5, bt_col6 = st.columns(3)
with bt_col4:
    include_tickers = st.multiselect("Include Holdings", all_tickers, default=all_tickers)
with bt_col5:
    benchmark_label = st.selectbox("Benchmark Overlay", ["None"] + list(BENCHMARK_OPTIONS.keys()), index=1)
with bt_col6:
    bt_rf_rate = st.slider("Risk-Free Rate (%)", 0.0, 12.0, 6.0, step=0.25) / 100.0

date_mode = st.radio("Lookback Window", ["Preset", "Custom Range"], horizontal=True)

bt_period, bt_start, bt_end = "1y", None, None
if date_mode == "Preset":
    bt_period = st.select_slider("Preset Window", options=["3mo", "6mo", "1y", "2y", "5y"], value="1y")
else:
    date_col1, date_col2 = st.columns(2)
    default_start = datetime.date.today() - datetime.timedelta(days=365)
    with date_col1:
        bt_start = st.date_input("Start Date", value=default_start)
    with date_col2:
        bt_end = st.date_input("End Date", value=datetime.date.today())

if st.button("Run Backtest"):
    if not include_tickers:
        st.error("Select at least one holding to include in the backtest.")
    else:
        with st.spinner("Running backtest simulation..."):
            benchmark_ticker = BENCHMARK_OPTIONS.get(benchmark_label) if benchmark_label != "None" else None
            bt_result = run_backtest(
                rebalance_freq=rebalance_freq,
                transaction_cost_bps=txn_cost,
                period=bt_period,
                start=bt_start,
                end=bt_end,
                include_tickers=include_tickers,
                benchmark_ticker=benchmark_ticker,
                risk_free_rate=bt_rf_rate,
                rebalance_mode=rebalance_mode
            )

        if bt_result is None:
            st.error("Not enough overlapping historical data for the selected holdings/date range to run a backtest.")
        else:
            rebal_series = bt_result["rebalanced"]
            bh_series = bt_result["buy_and_hold"]

            fig_bt = go.Figure()
            fig_bt.add_trace(go.Scatter(
                x=rebal_series.index, y=rebal_series.values,
                name=f"Rebalanced ({rebalance_freq})", line=dict(color="#00C897", width=2)
            ))
            fig_bt.add_trace(go.Scatter(
                x=bh_series.index, y=bh_series.values,
                name="Buy & Hold", line=dict(color="#FFA500", width=2, dash="dash")
            ))
            if "benchmark" in bt_result:
                fig_bt.add_trace(go.Scatter(
                    x=bt_result["benchmark"].index, y=bt_result["benchmark"].values,
                    name=benchmark_label, line=dict(color="#888888", width=1.5, dash="dot")
                ))
            fig_bt.update_layout(
                title="Backtest: Rebalanced vs Buy & Hold — Growth of ₹1",
                xaxis_title="Date",
                yaxis_title="Growth of ₹1",
                hovermode="x unified"
            )
            st.plotly_chart(fig_bt, use_container_width=True)

            stats_data = {
                "Rebalanced": compute_series_stats(rebal_series, risk_free_rate=bt_rf_rate),
                "Buy & Hold": compute_series_stats(bh_series, risk_free_rate=bt_rf_rate)
            }
            if "benchmark" in bt_result:
                stats_data[benchmark_label] = compute_series_stats(bt_result["benchmark"], risk_free_rate=bt_rf_rate)

            stats_df = pd.DataFrame(stats_data).T
            stats_df = stats_df.rename(columns={
                "cagr": "CAGR", "volatility": "Volatility", "sharpe": "Sharpe", "max_drawdown": "Max Drawdown"
            })
            stats_df["CAGR"] = stats_df["CAGR"].map(lambda x: f"{x:.2%}")
            stats_df["Volatility"] = stats_df["Volatility"].map(lambda x: f"{x:.2%}")
            stats_df["Sharpe"] = stats_df["Sharpe"].map(lambda x: f"{x:.2f}")
            stats_df["Max Drawdown"] = stats_df["Max Drawdown"].map(lambda x: f"{x:.2%}")

            st.dataframe(stats_df, use_container_width=True)

            if rebalance_freq != "None":
                st.caption(
                    f"Cumulative transaction-cost drag from rebalancing: "
                    f"{bt_result['total_transaction_cost']:.4%} of portfolio value over the window."
                )

            with st.expander("ℹ️ How this backtest works"):
                st.markdown("""
                - **Fixed mode:** target weights come from your current holdings (shares × latest price);
                  rebalancing pulls drifted weights back to those targets.
                - **Walk-Forward Optimized mode:** at each rebalance date, weights are re-solved for
                  max Sharpe ratio using only the trailing lookback window available up to that date
                  (no future data is used) — this approximates how a systematic strategy would actually
                  be run in real time.
                - **Buy & Hold** always starts at your live target weights and never trades.
                - Transaction costs are a simple proportional charge (in bps) on turnover at each rebalance.
                - This is a simplified simulation: it ignores dividends, taxes, and intraday slippage.
                """)

st.divider()


# ================================
# MEAN-VARIANCE OPTIMIZER / EFFICIENT FRONTIER
# ================================
st.subheader("🎯 Portfolio Optimizer — Efficient Frontier")
st.caption("Classic Markowitz mean-variance optimization: compares your current weights against the max-Sharpe and min-volatility portfolios.")

opt_col1, opt_col2, opt_col3 = st.columns(3)
with opt_col1:
    opt_tickers = st.multiselect("Assets to Optimize Over", all_tickers, default=all_tickers, key="opt_tickers")
with opt_col2:
    opt_period = st.selectbox("Historical Window", ["6mo", "1y", "2y", "5y"], index=1, key="opt_period")
with opt_col3:
    opt_rf_rate = st.slider("Risk-Free Rate (%) ", 0.0, 12.0, 6.0, step=0.25, key="opt_rf") / 100.0

allow_short = st.checkbox("Allow short-selling (negative weights)", value=False)

if st.button("Run Optimizer"):
    if len(opt_tickers) < 2:
        st.error("Select at least 2 assets to build a meaningful efficient frontier.")
    else:
        with st.spinner("Solving for optimal portfolios..."):
            opt_result = run_optimizer(
                include_tickers=opt_tickers,
                period=opt_period,
                risk_free_rate=opt_rf_rate,
                allow_short=allow_short
            )

        if opt_result is None:
            st.error("Not enough overlapping historical data for the selected assets to run the optimizer.")
        else:
            fig_frontier = go.Figure()

            # Efficient frontier curve
            fig_frontier.add_trace(go.Scatter(
                x=[v * 100 for v in opt_result["frontier_vols"]],
                y=[r * 100 for r in opt_result["frontier_returns"]],
                mode="lines", name="Efficient Frontier",
                line=dict(color="#6C63FF", width=2)
            ))

            # Individual assets
            asset_x = [opt_result["asset_vols"][t] * 100 for t in opt_result["tickers"]]
            asset_y = [opt_result["asset_returns"][t] * 100 for t in opt_result["tickers"]]
            fig_frontier.add_trace(go.Scatter(
                x=asset_x, y=asset_y, mode="markers+text", name="Individual Holdings",
                text=opt_result["tickers"], textposition="top center",
                marker=dict(size=9, color="#AAAAAA")
            ))

            # Current portfolio
            fig_frontier.add_trace(go.Scatter(
                x=[opt_result["current_vol"] * 100], y=[opt_result["current_return"] * 100],
                mode="markers", name="Your Current Portfolio",
                marker=dict(size=14, color="#FF6B6B", symbol="diamond")
            ))

            # Max Sharpe portfolio
            fig_frontier.add_trace(go.Scatter(
                x=[opt_result["max_sharpe_vol"] * 100], y=[opt_result["max_sharpe_return"] * 100],
                mode="markers", name="Max Sharpe Portfolio",
                marker=dict(size=14, color="#00C897", symbol="star")
            ))

            # Min volatility portfolio
            fig_frontier.add_trace(go.Scatter(
                x=[opt_result["min_vol_vol"] * 100], y=[opt_result["min_vol_return"] * 100],
                mode="markers", name="Min Volatility Portfolio",
                marker=dict(size=14, color="#FFA500", symbol="square")
            ))

            fig_frontier.update_layout(
                title="Efficient Frontier — Risk vs Return",
                xaxis_title="Annualised Volatility (%)",
                yaxis_title="Annualised Return (%)",
                hovermode="closest"
            )
            st.plotly_chart(fig_frontier, use_container_width=True)

            # Weight comparison table
            weight_rows = []
            for t in opt_result["tickers"]:
                weight_rows.append({
                    "Ticker": t,
                    "Current Weight": opt_result["current_weights"][t],
                    "Max Sharpe Weight": opt_result["max_sharpe_weights"][t],
                    "Min Volatility Weight": opt_result["min_vol_weights"][t]
                })
            weight_df = pd.DataFrame(weight_rows).set_index("Ticker")
            # Series.map (not DataFrame.applymap/.map, which vary across pandas versions) for safety
            weight_df_display = weight_df.apply(lambda col: col.map(lambda x: f"{x:.2%}"))
            st.dataframe(weight_df_display, use_container_width=True)

            # --- Suggested rebalancing trades: turn the abstract weights into concrete actions ---
            st.markdown("#### 💡 Suggested Rebalancing Trades")
            target_choice = st.radio(
                "Target Allocation", ["Max Sharpe Portfolio", "Min Volatility Portfolio"],
                horizontal=True, key="opt_target_choice"
            )
            target_weights_map = (
                opt_result["max_sharpe_weights"] if target_choice == "Max Sharpe Portfolio"
                else opt_result["min_vol_weights"]
            )

            subset_df = df[df["Ticker"].isin(opt_result["tickers"])].set_index("Ticker")
            subset_value = subset_df["Value"].sum()

            trade_rows = []
            for t in opt_result["tickers"]:
                if t not in subset_df.index or pd.isna(subset_df.loc[t, "CurrentPrice"]):
                    continue
                price = subset_df.loc[t, "CurrentPrice"]
                current_shares = subset_df.loc[t, "Shares"]
                target_value = target_weights_map[t] * subset_value
                target_shares = target_value / price
                delta_shares = target_shares - current_shares

                trade_rows.append({
                    "Ticker": t,
                    "Current Shares": round(current_shares, 1),
                    "Target Shares": round(target_shares, 1),
                    "Action": "Buy" if delta_shares > 0.05 else ("Sell" if delta_shares < -0.05 else "Hold"),
                    "Shares to Trade": round(abs(delta_shares), 1),
                    "Approx ₹ Value": f"₹{abs(delta_shares * price):,.0f}"
                })

            if trade_rows:
                trade_df = pd.DataFrame(trade_rows).set_index("Ticker")
                st.dataframe(trade_df, use_container_width=True)
                st.caption(
                    "Share counts are implied by target weights and may include fractions — round to your "
                    "broker's tradable lot size. This ignores brokerage, taxes, and market impact."
                )

            summary_col1, summary_col2, summary_col3 = st.columns(3)
            summary_col1.metric("Current Portfolio", f"{opt_result['current_return']:.2%} return",
                                 f"{opt_result['current_vol']:.2%} volatility")
            summary_col2.metric("Max Sharpe Portfolio", f"{opt_result['max_sharpe_return']:.2%} return",
                                 f"{opt_result['max_sharpe_vol']:.2%} volatility")
            summary_col3.metric("Min Volatility Portfolio", f"{opt_result['min_vol_return']:.2%} return",
                                 f"{opt_result['min_vol_vol']:.2%} volatility")

            with st.expander("ℹ️ How to read this"):
                st.markdown("""
                - Each point on the **purple curve** is the lowest possible volatility achievable for that level of return,
                  given your selected assets' historical means, variances, and correlations.
                - The **red diamond** is where your current holdings actually sit — often *inside* the frontier,
                  meaning the same or better return is achievable at lower risk with different weights.
                - The **green star** is the max-Sharpe (best risk-adjusted return) portfolio; the **orange square**
                  is the lowest-volatility portfolio achievable from these assets.
                - This is based purely on **trailing historical data** — it assumes the past return/risk/correlation
                  pattern continues, which is a real limitation of mean-variance optimization in practice.
                """)

st.divider()


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
st.caption("Data via Yahoo Finance. Prices reflect the last available trading session close. Capture ratios and backtests computed vs NIFTY 50 / current holdings on a daily-return basis.")
