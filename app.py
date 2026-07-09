import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf
import time
import datetime

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
    get_portfolio_var_cvar,
    get_rolling_metrics,
    run_stress_test,
    get_advanced_ratios,
    get_risk_contributions,
    get_concentration_index,
    get_tail_risk_stats,
    run_monte_carlo_var,
    get_liquidity_metrics,
    get_factor_decomposition,
    BENCHMARK_OPTIONS
)

from llm import ask_llm


# ================================
# CONFIG
# ================================
st.set_page_config(page_title="Portfolio Intelligence", layout="wide", page_icon="📊")

# Light styling polish — subtle card backgrounds behind metrics, tighter spacing,
# consistent accent color matching the chart palette used throughout (#00C897).
st.markdown("""
    <style>
    div[data-testid="stMetric"] {
        background-color: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 14px 16px 10px 16px;
    }
    div[data-testid="stMetricLabel"] { font-size: 0.85rem; opacity: 0.85; }
    h1 { padding-bottom: 0rem; }
    .block-container { padding-top: 2rem; }
    </style>
""", unsafe_allow_html=True)

st.title("📊 Pro Portfolio Intelligence Dashboard")
st.caption("Live NSE portfolio analytics — risk, backtesting, optimization, and AI-generated insights.")


# ================================
# SIDEBAR CONTROLS
# ================================
st.sidebar.header("Controls")
force_refresh = st.sidebar.button("🔄 Refresh Live Prices")
st.sidebar.markdown("---")
st.sidebar.caption("Prices: Last available close (NSE). Updates reflect latest trading session.")
st.sidebar.caption(f"Last loaded: {datetime.datetime.now().strftime('%d %b %Y, %I:%M %p')}")


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
all_tickers = df["Ticker"].tolist()

missing = df[df["CurrentPrice"].isna() | (df["CurrentPrice"] == 0)]["Ticker"].tolist()
if missing:
    st.warning(f"⚠️ Could not fetch prices for: {', '.join(missing)}. These are excluded from total value.")


# ================================
# METRICS (computed once, used across tabs)
# ================================
with st.spinner("Computing portfolio metrics..."):
    metrics = get_metrics()
    capture = get_capture_ratios(
        portfolio_returns=metrics["returns"],
        benchmark_returns=metrics["benchmark_returns"]
    )

with st.spinner("Loading historical data..."):
    portfolio = get_portfolio_history()

benchmark = yf.download("^NSEI", period="1y", auto_adjust=True, progress=False)["Close"]
if isinstance(benchmark, pd.DataFrame):
    benchmark = benchmark.iloc[:, 0]

aligned = pd.concat([portfolio, benchmark], axis=1).dropna()
aligned.columns = ["Portfolio", "NIFTY"]
portfolio_norm = aligned["Portfolio"] / aligned["Portfolio"].iloc[0]
nifty_norm = aligned["NIFTY"] / aligned["NIFTY"].iloc[0]


# ================================
# AT-A-GLANCE HEADER (always visible, above the tabs)
# ================================
with st.container(border=True):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Portfolio Value", f"₹{total_value:,.0f}")
    c2.metric("Annual Return", f"{metrics['annual_return']:.2%}")
    c3.metric("Sharpe Ratio", f"{metrics['sharpe']:.2f}")
    c4.metric("Beta", f"{metrics['beta']:.2f}")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Max Drawdown", f"{metrics['max_drawdown']:.2%}", delta_color="inverse")
    c6.metric("Volatility (Ann.)", f"{metrics['volatility']:.2%}")
    c7.metric(
        "Upside Capture", f"{capture['upside_capture']:.1f}%",
        help="% of benchmark upside the portfolio captured. >100% is ideal."
    )
    c8.metric(
        "Downside Capture", f"{capture['downside_capture']:.1f}%", delta_color="inverse",
        help="% of benchmark downside the portfolio suffered. <100% is ideal."
    )

st.write("")


# ================================
# TABS
# ================================
tab_overview, tab_risk, tab_advrisk, tab_backtest, tab_optimizer, tab_ai = st.tabs([
    "📊 Overview", "⚠️ Risk Analytics", "🧮 Advanced Risk", "🔁 Backtesting", "🎯 Optimizer", "🤖 AI Insights"
])


# ============================================================
# TAB 1 — OVERVIEW
# ============================================================
with tab_overview:

    # --- Portfolio vs Benchmark ---
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
        xaxis_title="Date", yaxis_title="Growth of ₹1",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig1, use_container_width=True)

    # --- Drawdown Chart ---
    st.subheader("📉 Rolling Drawdown from Peak")
    drawdown = get_drawdown_series(portfolio)

    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(
        x=drawdown.index, y=drawdown.values * 100,
        fill="tozeroy", fillcolor="rgba(255, 80, 80, 0.25)",
        line=dict(color="red", width=1.5), name="Drawdown (%)"
    ))
    fig_dd.update_layout(
        title="Portfolio Drawdown from Rolling Peak (%)",
        xaxis_title="Date", yaxis_title="Drawdown (%)",
        yaxis=dict(ticksuffix="%"), hovermode="x unified"
    )
    fig_dd.add_hline(
        y=float(metrics["max_drawdown"] * 100), line_dash="dot", line_color="darkred",
        annotation_text=f"Max DD: {metrics['max_drawdown']:.2%}", annotation_position="bottom right"
    )
    st.plotly_chart(fig_dd, use_container_width=True)

    # --- Capture Ratio Chart ---
    st.subheader("📈 Upside / Downside Capture Ratios")
    fig_cap = go.Figure()
    categories = ["Upside Capture", "Downside Capture"]
    values = [capture["upside_capture"], capture["downside_capture"]]
    colors = [
        "#00C897" if capture["upside_capture"] >= 100 else "#FFA500",
        "#00C897" if capture["downside_capture"] <= 100 else "#FF6B6B"
    ]
    fig_cap.add_trace(go.Bar(
        x=categories, y=values, marker_color=colors,
        text=[f"{v:.1f}%" for v in values], textposition="outside"
    ))
    fig_cap.add_hline(
        y=100, line_dash="dash", line_color="gray",
        annotation_text="Benchmark (100%)", annotation_position="top right"
    )
    fig_cap.update_layout(
        title="Capture Ratios vs NIFTY 50 (1Y)", yaxis_title="Capture Ratio (%)",
        yaxis=dict(ticksuffix="%", range=[0, max(values) * 1.25 + 10]),
        xaxis_title="", showlegend=False
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

    # --- Allocation Pie ---
    fig2 = px.pie(df, names="Ticker", values="Value", title="Portfolio Allocation by Holding")
    st.plotly_chart(fig2, use_container_width=True)

    # --- Sector Exposure ---
    sector_alloc = df.groupby("Sector")["Value"].sum().reset_index()
    sector_alloc.columns = ["Sector", "Value"]
    fig3 = px.bar(
        sector_alloc, x="Sector", y="Value", color="Sector",
        title="Sector Exposure (₹)", labels={"Value": "Exposure (₹)"}
    )
    st.plotly_chart(fig3, use_container_width=True)

    # --- Correlation Matrix ---
    st.subheader("🧬 Holdings Correlation Matrix")
    st.caption("Daily-return correlation over the trailing 1Y. Lower correlations between holdings mean more genuine diversification benefit.")

    corr_matrix = get_correlation_matrix(period="1y")
    if corr_matrix is not None:
        fig_corr = px.imshow(
            corr_matrix, text_auto=".2f", color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1, aspect="auto", title="Pairwise Correlation of Daily Returns"
        )
        fig_corr.update_layout(coloraxis_colorbar=dict(title="Corr"))
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.info("Need at least 2 holdings with overlapping price history to compute a correlation matrix.")

    # --- Holdings Ranking ---
    st.subheader("Holdings Ranking by Weight")
    ranked = df.sort_values("Weight", ascending=False)
    fig4 = px.bar(
        ranked, x="Ticker", y="Weight", text="Weight",
        title="Portfolio Weight Ranking", color="Sector"
    )
    fig4.update_traces(texttemplate='%{text:.2%}', textposition='outside')
    fig4.update_layout(
        xaxis_title="Stock", yaxis_title="Weight in Portfolio", yaxis=dict(tickformat=".0%")
    )
    st.plotly_chart(fig4, use_container_width=True)

    # --- Holdings Table ---
    st.subheader("Holdings Detail")
    display_df = df.copy()
    display_df["CurrentPrice"] = display_df["CurrentPrice"].map(lambda x: f"₹{x:,.2f}" if pd.notna(x) else "N/A")
    display_df["Value"] = display_df["Value"].map(lambda x: f"₹{x:,.0f}" if pd.notna(x) else "N/A")
    display_df["Weight"] = display_df["Weight"].map(lambda x: f"{x:.2%}" if pd.notna(x) else "N/A")
    st.dataframe(display_df, use_container_width=True)


# ============================================================
# TAB 2 — RISK ANALYTICS (VaR/CVaR, Rolling Metrics, Stress Testing)
# ============================================================
with tab_risk:

    # --- VaR / CVaR ---
    st.subheader("📉 Value at Risk (VaR) & Conditional VaR (CVaR)")
    st.caption("Estimates the magnitude of loss at a given confidence level and time horizon.")

    var_col1, var_col2, var_col3 = st.columns(3)
    with var_col1:
        var_confidence = st.selectbox("Confidence Level", [0.90, 0.95, 0.99], index=1,
                                       format_func=lambda x: f"{x:.0%}")
    with var_col2:
        var_method = st.selectbox("Method", ["historical", "parametric"],
                                   format_func=lambda x: x.capitalize())
    with var_col3:
        var_horizon = st.selectbox("Horizon (Days)", [1, 5, 10, 21], index=0)

    var_result = get_portfolio_var_cvar(confidence=var_confidence, method=var_method, horizon_days=var_horizon)

    vcol1, vcol2 = st.columns(2)
    vcol1.metric(
        f"VaR ({var_confidence:.0%}, {var_horizon}d)",
        f"{var_result['var_pct']:.2%}",
        f"≈ ₹{var_result['var_amount']:,.0f}"
    )
    vcol2.metric(
        f"CVaR / Expected Shortfall ({var_confidence:.0%}, {var_horizon}d)",
        f"{var_result['cvar_pct']:.2%}",
        f"≈ ₹{var_result['cvar_amount']:,.0f}"
    )

    with st.expander("ℹ️ How to read VaR/CVaR"):
        st.markdown(f"""
        - **VaR** answers: *"What's the worst loss I'd expect, {var_confidence:.0%} of the time, over {var_horizon} day(s)?"*
          A VaR of {var_result['var_pct']:.2%} means there's roughly a {1-var_confidence:.0%} chance of losing
          *more* than that over the period.
        - **CVaR (Expected Shortfall)** answers: *"If it IS a bad day beyond that VaR threshold, how bad on average?"*
          It's always ≥ VaR, and is considered a more complete tail-risk measure.
        - **Historical method**: uses your actual past return distribution — no assumptions, but limited by available history.
        - **Parametric method**: assumes returns are normally distributed — smoother, but can understate real
          "fat tail" risk that equities are prone to.
        - The {var_horizon}-day scaling uses the standard √t approximation, which assumes returns are independent
          day-to-day — a simplification worth keeping in mind for longer horizons.
        """)

    st.divider()

    # --- Rolling Risk Metrics ---
    st.subheader("📈 Rolling Risk Metrics")
    st.caption("How Sharpe, Beta, and Volatility have evolved over time, rather than a single point-in-time snapshot.")

    roll_window = st.select_slider(
        "Rolling Window (trading days)", options=[21, 63, 126, 252], value=63,
        help="21 ≈ 1 month, 63 ≈ 1 quarter, 126 ≈ 6 months, 252 ≈ 1 year"
    )
    rolling_df = get_rolling_metrics(window_days=roll_window)

    if rolling_df.empty:
        st.info("Not enough historical data yet for this rolling window — try a shorter window.")
    else:
        fig_roll_sharpe = go.Figure()
        fig_roll_sharpe.add_trace(go.Scatter(
            x=rolling_df.index, y=rolling_df["Rolling Sharpe"],
            line=dict(color="#00C897", width=2), name="Rolling Sharpe"
        ))
        fig_roll_sharpe.add_hline(y=0, line_dash="dot", line_color="gray")
        fig_roll_sharpe.update_layout(title=f"Rolling Sharpe Ratio ({roll_window}-day window)",
                                       xaxis_title="Date", yaxis_title="Sharpe Ratio")
        st.plotly_chart(fig_roll_sharpe, use_container_width=True)

        fig_roll_beta = go.Figure()
        fig_roll_beta.add_trace(go.Scatter(
            x=rolling_df.index, y=rolling_df["Rolling Beta"],
            line=dict(color="#6C63FF", width=2), name="Rolling Beta"
        ))
        fig_roll_beta.add_hline(y=1, line_dash="dot", line_color="gray", annotation_text="Beta = 1 (Market)")
        fig_roll_beta.update_layout(title=f"Rolling Beta vs NIFTY 50 ({roll_window}-day window)",
                                     xaxis_title="Date", yaxis_title="Beta")
        st.plotly_chart(fig_roll_beta, use_container_width=True)

        fig_roll_vol = go.Figure()
        fig_roll_vol.add_trace(go.Scatter(
            x=rolling_df.index, y=rolling_df["Rolling Volatility"] * 100,
            line=dict(color="#FFA500", width=2), name="Rolling Volatility",
            fill="tozeroy", fillcolor="rgba(255, 165, 0, 0.15)"
        ))
        fig_roll_vol.update_layout(title=f"Rolling Annualised Volatility ({roll_window}-day window)",
                                    xaxis_title="Date", yaxis_title="Volatility (%)", yaxis=dict(ticksuffix="%"))
        st.plotly_chart(fig_roll_vol, use_container_width=True)

    st.divider()

    # --- Stress Testing ---
    st.subheader("🧨 Stress Testing")
    st.caption("Estimated portfolio impact under hypothetical and historical index shocks, using the portfolio's current beta.")

    stress_df = run_stress_test()

    fig_stress = go.Figure()
    stress_colors = ["#00C897" if v >= 0 else "#FF6B6B" for v in stress_df["Est. Portfolio Impact"]]
    fig_stress.add_trace(go.Bar(
        x=stress_df.index, y=stress_df["Est. Portfolio Impact"] * 100,
        marker_color=stress_colors,
        text=[f"{v:.1%}" for v in stress_df["Est. Portfolio Impact"]],
        textposition="outside"
    ))
    fig_stress.update_layout(
        title="Estimated Portfolio Impact by Scenario",
        yaxis_title="Estimated Portfolio Return (%)", yaxis=dict(ticksuffix="%"),
        xaxis_title="", xaxis=dict(tickangle=-20)
    )
    st.plotly_chart(fig_stress, use_container_width=True)

    stress_display = stress_df.copy()
    stress_display["Index Shock"] = stress_display["Index Shock"].map(lambda x: f"{x:.1%}")
    stress_display["Est. Portfolio Impact"] = stress_display["Est. Portfolio Impact"].map(lambda x: f"{x:.2%}")
    stress_display["Est. Rupee Impact"] = stress_display["Est. Rupee Impact"].map(lambda x: f"₹{x:,.0f}")
    st.dataframe(stress_display, use_container_width=True)

    with st.expander("ℹ️ How this stress test works"):
        st.markdown("""
        - Each scenario applies a hypothetical or historical index shock and scales it by your
          portfolio's **current beta** to estimate the impact: `estimated return = beta × index shock`.
        - This is a **simplified, single-factor approximation** — it does not re-price individual
          holdings under the actual historical scenario, and assumes beta stays constant during the
          shock. In real crises, correlations across holdings often rise, which can make real losses
          worse than a simple beta-scaling suggests.
        - Treat this as a directional risk indicator, not a precise forecast.
        """)


# ============================================================
# TAB 3 — ADVANCED RISK (Ratios, Decomposition, Tail Risk, Liquidity, Factor Model)
# ============================================================
with tab_advrisk:

    # --- Advanced Risk-Adjusted Ratios ---
    st.subheader("📐 Advanced Risk-Adjusted Ratios")
    st.caption("Sortino, Tracking Error, and Information Ratio — complements the Sharpe Ratio shown at the top of the page.")

    adv_rf_rate = st.slider("Risk-Free Rate (%)", 0.0, 12.0, 6.0, step=0.25, key="adv_rf") / 100.0
    adv_ratios = get_advanced_ratios(risk_free_rate=adv_rf_rate)

    ar1, ar2, ar3 = st.columns(3)
    ar1.metric("Sortino Ratio", f"{adv_ratios['sortino']:.2f}",
               help="Like Sharpe, but only penalises downside volatility.")
    ar2.metric("Tracking Error (Ann.)", f"{adv_ratios['tracking_error']:.2%}",
               help="Standard deviation of (portfolio return - benchmark return). Measures active risk.")
    ar3.metric("Information Ratio", f"{adv_ratios['information_ratio']:.2f}",
               help="Active return / Tracking error. Is the active risk being rewarded?")

    with st.expander("ℹ️ How to read these"):
        st.markdown(f"""
        - **Sortino Ratio** ({adv_ratios['sortino']:.2f}): uses downside deviation
          ({adv_ratios['downside_deviation']:.2%} annualised) instead of total volatility — a stock that
          only ever surprises to the upside would score well here even if "Volatility" looks high.
        - **Tracking Error** ({adv_ratios['tracking_error']:.2%}): how far your daily returns stray from
          NIFTY 50's, annualised. A closet-index fund has near-zero tracking error; a concentrated
          active book has high tracking error by design — neither is automatically good or bad.
        - **Information Ratio** ({adv_ratios['information_ratio']:.2f}): active return
          ({adv_ratios['active_return']:.2%} annualised) divided by tracking error. This is the key
          number active managers are judged on — it answers whether the active bets are actually paying off
          relative to the active risk taken, not just whether returns are positive.
        """)

    st.divider()

    # --- Risk Decomposition: Component Risk Contribution + Concentration ---
    st.subheader("🧩 Risk Decomposition")
    st.caption("How much each holding contributes to TOTAL portfolio risk — which can differ meaningfully from its weight.")

    risk_contrib_df = get_risk_contributions(period="1y", confidence=0.95, var_method="historical", horizon_days=1)

    if risk_contrib_df is None:
        st.info("Need at least 2 holdings with overlapping price history to compute risk contributions.")
    else:
        fig_riskcontrib = go.Figure()
        fig_riskcontrib.add_trace(go.Bar(
            x=risk_contrib_df.index, y=risk_contrib_df["Weight"] * 100,
            name="Portfolio Weight", marker_color="#AAAAAA"
        ))
        fig_riskcontrib.add_trace(go.Bar(
            x=risk_contrib_df.index, y=risk_contrib_df["% of Portfolio Risk"] * 100,
            name="% of Portfolio Risk", marker_color="#FF6B6B"
        ))
        fig_riskcontrib.update_layout(
            title="Portfolio Weight vs. Risk Contribution by Holding",
            barmode="group", yaxis_title="%", xaxis_title=""
        )
        st.plotly_chart(fig_riskcontrib, use_container_width=True)

        display_contrib = risk_contrib_df.copy()
        display_contrib["Weight"] = display_contrib["Weight"].map(lambda x: f"{x:.2%}")
        display_contrib["% of Portfolio Risk"] = display_contrib["% of Portfolio Risk"].map(lambda x: f"{x:.2%}")
        display_contrib["Risk-to-Weight Ratio"] = display_contrib["Risk-to-Weight Ratio"].map(lambda x: f"{x:.2f}x")
        display_contrib["Component VaR (%)"] = display_contrib["Component VaR (%)"].map(lambda x: f"{x:.2%}")
        display_contrib["Component VaR (₹)"] = display_contrib["Component VaR (₹)"].map(lambda x: f"₹{x:,.0f}")
        st.dataframe(display_contrib, use_container_width=True)

        st.caption(
            "**Risk-to-Weight Ratio > 1x** means a holding contributes proportionally more risk than its "
            "portfolio weight — usually because it's more volatile and/or more correlated with the rest of the book."
        )

        # Concentration Index — holding level and sector level
        conc_col1, conc_col2 = st.columns(2)
        holding_weights = df.set_index("Ticker")["Weight"]
        holding_conc = get_concentration_index(holding_weights, basis="holding")
        with conc_col1:
            st.metric("Concentration Index (by Holding)", f"{holding_conc['hhi']:.0f} HHI", holding_conc["label"])

        sector_weights = df.groupby("Sector")["Value"].sum()
        sector_weights = sector_weights / sector_weights.sum()
        sector_conc = get_concentration_index(sector_weights, basis="sector")
        with conc_col2:
            st.metric("Concentration Index (by Sector)", f"{sector_conc['hhi']:.0f} HHI", sector_conc["label"])

        st.caption(
            "HHI (Herfindahl-Hirschman Index) ranges 0-10,000. Bands (<1,500 / 1,500-2,500 / >2,500) are "
            "adapted from the standard antitrust-analysis convention as a rule-of-thumb scale for portfolio concentration."
        )

    st.divider()

    # --- Tail Risk & Monte Carlo VaR ---
    st.subheader("🎲 Tail Risk & Monte Carlo Simulation")
    st.caption("How far your actual return distribution deviates from 'normal', and a simulation-based view of potential losses.")

    tail_stats = get_tail_risk_stats()
    tr1, tr2 = st.columns(2)
    tr1.metric("Skewness", f"{tail_stats['skewness']:.2f}",
               help="Negative = more/larger down days than up days. 0 = symmetric like a normal distribution.")
    tr2.metric("Excess Kurtosis", f"{tail_stats['kurtosis']:.2f}",
               help="Above 0 means 'fatter tails' than normal — more extreme days than a normal distribution predicts.")

    mc_col1, mc_col2, mc_col3 = st.columns(3)
    with mc_col1:
        mc_method = st.selectbox("Simulation Method", ["bootstrap", "parametric"],
                                  format_func=lambda x: x.capitalize())
    with mc_col2:
        mc_horizon = st.selectbox("Horizon (Days)", [10, 21, 63], index=1)
    with mc_col3:
        mc_confidence = st.selectbox("Confidence", [0.90, 0.95, 0.99], index=1, format_func=lambda x: f"{x:.0%}")

    if st.button("Run Monte Carlo Simulation"):
        with st.spinner(f"Running {5000:,} simulations..."):
            mc_result = run_monte_carlo_var(
                n_simulations=5000, horizon_days=mc_horizon, confidence=mc_confidence, method=mc_method
            )

        if mc_result is None:
            st.error("Not enough historical return data to run a simulation.")
        else:
            mcol1, mcol2 = st.columns(2)
            mcol1.metric(f"Simulated VaR ({mc_confidence:.0%}, {mc_horizon}d)",
                         f"{mc_result['var_pct']:.2%}", f"≈ ₹{mc_result['var_amount']:,.0f}")
            mcol2.metric(f"Simulated CVaR ({mc_confidence:.0%}, {mc_horizon}d)",
                         f"{mc_result['cvar_pct']:.2%}", f"≈ ₹{mc_result['cvar_amount']:,.0f}")

            fig_mc = go.Figure()
            fig_mc.add_trace(go.Histogram(
                x=mc_result["simulated_returns"] * 100, nbinsx=60,
                marker_color="#6C63FF", opacity=0.75, name="Simulated Outcomes"
            ))
            fig_mc.add_vline(
                x=-mc_result["var_pct"] * 100, line_dash="dash", line_color="#FF6B6B",
                annotation_text=f"VaR ({mc_confidence:.0%})", annotation_position="top"
            )
            fig_mc.update_layout(
                title=f"Distribution of Simulated {mc_horizon}-Day Portfolio Returns ({mc_result['n_simulations']:,} runs)",
                xaxis_title="Cumulative Return (%)", yaxis_title="Frequency"
            )
            st.plotly_chart(fig_mc, use_container_width=True)

            with st.expander("ℹ️ How to read this"):
                st.markdown(f"""
                - **Bootstrap method** resamples your *actual* historical daily returns thousands of times —
                  it makes no assumption about the shape of the distribution, so real fat tails and skew
                  carry through into the simulation.
                - **Parametric method** draws from a normal distribution fitted to your historical mean/volatility —
                  smoother, but inherits the same normality limitation flagged in the VaR/CVaR section.
                - The dashed red line marks the VaR threshold: {(1-mc_confidence):.0%} of simulated outcomes
                  fell below it.
                """)

    st.divider()

    # --- Liquidity Risk ---
    st.subheader("💧 Liquidity Risk")
    st.caption("Estimated days to fully exit each position without dominating its own trading volume.")

    participation_rate = st.slider(
        "Max Participation Rate (% of Avg Daily Volume)", 5, 30, 10,
        help="The maximum share of a stock's average daily trading volume you'd trade in a single day to limit market impact."
    ) / 100.0

    with st.spinner("Fetching trading volume data..."):
        liquidity_df = get_liquidity_metrics(participation_rate=participation_rate, period="1mo")

    if liquidity_df.empty:
        st.info("Could not fetch volume data for these holdings.")
    else:
        liq_display = liquidity_df.copy()
        liq_display["Avg Daily Volume"] = liq_display["Avg Daily Volume"].map(
            lambda x: f"{x:,.0f}" if pd.notna(x) else "N/A"
        )
        liq_display["Days to Liquidate"] = liq_display["Days to Liquidate"].map(
            lambda x: f"{x:.1f}" if pd.notna(x) else "N/A"
        )
        st.dataframe(liq_display, use_container_width=True)

        low_liquidity = liquidity_df[liquidity_df["Liquidity Flag"] == "Low Liquidity"]
        if not low_liquidity.empty:
            st.warning(f"⚠️ Low liquidity flagged for: {', '.join(low_liquidity.index.tolist())} "
                       f"(more than 5 days to exit at a {participation_rate:.0%} participation rate).")

        st.caption(
            "Based on trailing 1-month average daily volume (20 trading days). This is a simplified estimate — "
            "it ignores intraday liquidity patterns, block/negotiated deals, and market-impact costs beyond the "
            "participation-rate assumption."
        )

    st.divider()

    # --- Single-Factor (Market) Risk Decomposition ---
    st.subheader("📡 Factor Risk Decomposition (Single-Factor / Market Model)")
    st.caption("Splits each holding's and the portfolio's total variance into market-driven (systematic) vs stock-specific (idiosyncratic) risk.")

    st.info(
        "ℹ️ This uses a **single-factor (market) model** — beta and R² against NIFTY 50. "
        "A genuine multi-factor model (size, value, momentum, quality) would need external factor-return "
        "datasets that aren't available through this pipeline, so it isn't included here.",
        icon="ℹ️"
    )

    factor_result = get_factor_decomposition(period="1y")

    if factor_result is None:
        st.info("Not enough overlapping historical data to run the factor decomposition.")
    else:
        port_factor = factor_result["portfolio"]
        asset_factor = factor_result["assets"]

        fcol1, fcol2, fcol3 = st.columns(3)
        fcol1.metric("Portfolio Beta (Market Factor)", f"{port_factor['beta']:.2f}")
        fcol2.metric("Systematic Risk %", f"{port_factor['systematic_pct']:.1f}%",
                     help="Share of total portfolio variance explained by market movements.")
        fcol3.metric("Idiosyncratic Risk %", f"{port_factor['idiosyncratic_pct']:.1f}%",
                     help="Share of total portfolio variance that is stock-specific, not market-driven.")

        fig_factor_port = go.Figure(data=[go.Pie(
            labels=["Systematic (Market) Risk", "Idiosyncratic (Stock-Specific) Risk"],
            values=[port_factor["systematic_pct"], port_factor["idiosyncratic_pct"]],
            marker_colors=["#6C63FF", "#FFA500"], hole=0.45
        )])
        fig_factor_port.update_layout(title="Portfolio Variance: Systematic vs Idiosyncratic")
        st.plotly_chart(fig_factor_port, use_container_width=True)

        st.markdown("#### Asset-Level Factor Contributions")
        asset_display = asset_factor.copy()
        asset_display["Weight"] = asset_display["Weight"].map(lambda x: f"{x:.2%}")
        asset_display["Beta"] = asset_display["Beta"].map(lambda x: f"{x:.2f}")
        asset_display["R-Squared"] = asset_display["R-Squared"].map(lambda x: f"{x:.2f}")
        asset_display["Systematic Var %"] = asset_display["Systematic Var %"].map(lambda x: f"{x:.1f}%")
        asset_display["Idiosyncratic Var %"] = asset_display["Idiosyncratic Var %"].map(lambda x: f"{x:.1f}%")
        st.dataframe(asset_display, use_container_width=True)

        with st.expander("ℹ️ How to read this"):
            st.markdown("""
            - **Beta**: each holding's sensitivity to NIFTY 50 — the same concept as portfolio Beta, applied per stock.
            - **R-Squared**: how much of that stock's own variance is explained by the market. A high R² (close to 1)
              means the stock moves largely in lockstep with the index; a low R² means most of its risk is
              genuinely stock-specific, not market-driven.
            - **Systematic vs Idiosyncratic %**: the variance split for each holding individually, using the
              same single-index model as the portfolio-level chart above.
            - A portfolio full of high-R² holdings is effectively a leveraged or de-leveraged bet on the index,
              regardless of how many "different" stocks it holds — true diversification shows up as low R²
              and low cross-holding correlation (see the Correlation Matrix in the Overview tab).
            """)


# ============================================================
# TAB 4 — BACKTESTING
# ============================================================
with tab_backtest:
    st.subheader("🔁 Backtest Engine")
    st.caption("Simulates your holdings' trailing history: rebalanced vs buy-and-hold, with configurable filters.")

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
                    xaxis_title="Date", yaxis_title="Growth of ₹1", hovermode="x unified"
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


# ============================================================
# TAB 4 — OPTIMIZER / EFFICIENT FRONTIER
# ============================================================
with tab_optimizer:
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

                fig_frontier.add_trace(go.Scatter(
                    x=[v * 100 for v in opt_result["frontier_vols"]],
                    y=[r * 100 for r in opt_result["frontier_returns"]],
                    mode="lines", name="Efficient Frontier",
                    line=dict(color="#6C63FF", width=2)
                ))

                asset_x = [opt_result["asset_vols"][t] * 100 for t in opt_result["tickers"]]
                asset_y = [opt_result["asset_returns"][t] * 100 for t in opt_result["tickers"]]
                fig_frontier.add_trace(go.Scatter(
                    x=asset_x, y=asset_y, mode="markers+text", name="Individual Holdings",
                    text=opt_result["tickers"], textposition="top center",
                    marker=dict(size=9, color="#AAAAAA")
                ))

                fig_frontier.add_trace(go.Scatter(
                    x=[opt_result["current_vol"] * 100], y=[opt_result["current_return"] * 100],
                    mode="markers", name="Your Current Portfolio",
                    marker=dict(size=14, color="#FF6B6B", symbol="diamond")
                ))

                fig_frontier.add_trace(go.Scatter(
                    x=[opt_result["max_sharpe_vol"] * 100], y=[opt_result["max_sharpe_return"] * 100],
                    mode="markers", name="Max Sharpe Portfolio",
                    marker=dict(size=14, color="#00C897", symbol="star")
                ))

                fig_frontier.add_trace(go.Scatter(
                    x=[opt_result["min_vol_vol"] * 100], y=[opt_result["min_vol_return"] * 100],
                    mode="markers", name="Min Volatility Portfolio",
                    marker=dict(size=14, color="#FFA500", symbol="square")
                ))

                fig_frontier.update_layout(
                    title="Efficient Frontier — Risk vs Return",
                    xaxis_title="Annualised Volatility (%)", yaxis_title="Annualised Return (%)",
                    hovermode="closest"
                )
                st.plotly_chart(fig_frontier, use_container_width=True)

                weight_rows = []
                for t in opt_result["tickers"]:
                    weight_rows.append({
                        "Ticker": t,
                        "Current Weight": opt_result["current_weights"][t],
                        "Max Sharpe Weight": opt_result["max_sharpe_weights"][t],
                        "Min Volatility Weight": opt_result["min_vol_weights"][t]
                    })
                weight_df = pd.DataFrame(weight_rows).set_index("Ticker")
                weight_df_display = weight_df.apply(lambda col: col.map(lambda x: f"{x:.2%}"))
                st.dataframe(weight_df_display, use_container_width=True)

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


# ============================================================
# TAB 5 — SMART INSIGHTS (AI-POWERED)
# ============================================================
with tab_ai:
    st.subheader("🤖 Smart Insights")
    st.caption("AI-generated portfolio commentary via a locally-hosted LLM (Ollama).")

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
st.divider()
st.caption("Data via Yahoo Finance. Prices reflect the last available trading session close. Capture ratios, backtests, VaR, and stress tests are computed vs NIFTY 50 / current holdings on a daily-return basis.")
