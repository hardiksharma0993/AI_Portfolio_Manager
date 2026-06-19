import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from sklearn.linear_model import LinearRegression

# =========================================================
# CONFIG
# =========================================================
st.set_page_config(
    page_title="Pro Trading Intelligence Dashboard",
    page_icon="📊",
    layout="wide"
)

# =========================================================
# UI (LIGHT INSTITUTIONAL)
# =========================================================
st.markdown("""
<style>
.stApp {
    background-color: #F5F7FA;
    color: #111827;
}

h1, h2, h3 {
    color: #111827;
}

div[data-testid="stMetric"] {
    background-color: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-radius: 10px;
    padding: 12px;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# TITLE
# =========================================================
st.title("📊 Pro Trading Intelligence Dashboard")

# =========================================================
# DATA
# =========================================================
df = pd.read_csv("data/portfolio.csv")

# =========================================================
# PRICE ENGINE
# =========================================================
@st.cache_data(ttl=300)
def get_price(ticker):
    try:
        return yf.Ticker(ticker).history(period="1d")["Close"].iloc[-1]
    except:
        return np.nan

df["Price"] = df["Ticker"].apply(get_price)
df["Value"] = df["Price"] * df["Shares"]
df["Weight"] = df["Value"] / df["Value"].sum()
df["PnL"] = (df["Price"] - df["Avg_Buy_Price"]) * df["Shares"]

total_value = df["Value"].sum()

# =========================================================
# SECTOR MAP
# =========================================================
sector_map = {
    "TCS.NS": "IT",
    "INFY.NS": "IT",
    "HDFCBANK.NS": "Banking",
    "RELIANCE.NS": "Energy",
    "ITC.NS": "FMCG"
}

df["Sector"] = df["Ticker"].map(sector_map).fillna("Others")

# =========================================================
# SIDEBAR FILTERS
# =========================================================
st.sidebar.header("Portfolio Filters")

selected_sector = st.sidebar.multiselect(
    "Sector Filter",
    df["Sector"].unique(),
    df["Sector"].unique()
)

selected_ticker = st.sidebar.multiselect(
    "Stock Filter",
    df["Ticker"].unique(),
    df["Ticker"].unique()
)

st.sidebar.markdown("### Exposure Filters")

weight_range = st.sidebar.slider(
    "Portfolio Weight Range",
    0.0, 1.0,
    (0.0, 1.0)
)

pnl_range = st.sidebar.slider(
    "Unrealized PnL Range (₹)",
    float(df["PnL"].min()),
    float(df["PnL"].max()),
    (float(df["PnL"].min()), float(df["PnL"].max()))
)

# =========================================================
# FILTER APPLY
# =========================================================
df = df[
    (df["Sector"].isin(selected_sector)) &
    (df["Ticker"].isin(selected_ticker)) &
    (df["Weight"] >= weight_range[0]) &
    (df["Weight"] <= weight_range[1]) &
    (df["PnL"] >= pnl_range[0]) &
    (df["PnL"] <= pnl_range[1])
]

# =========================================================
# MARKET DATA
# =========================================================
market = yf.Ticker("^NSEI").history(period="6mo")["Close"].pct_change().dropna()
portfolio_returns = df["PnL"].pct_change().fillna(0)

# =========================================================
# SHARPE RATIO (CORRECT)
# =========================================================
rf = 0.05
Rp = portfolio_returns.mean() * 252
sigma = portfolio_returns.std() * np.sqrt(252)
sharpe = (Rp - rf) / (sigma + 1e-9)

# =========================================================
# CAPM BETA
# =========================================================
min_len = min(len(portfolio_returns), len(market))
X = market.values[:min_len].reshape(-1, 1)
y = portfolio_returns.values[:min_len]

beta = 0.0
if len(X) > 2:
    model = LinearRegression().fit(X, y)
    beta = float(model.coef_[0])

# =========================================================
# ROLLING BETA
# =========================================================
rolling_beta = pd.Series(y).rolling(10).corr(pd.Series(X.flatten())).fillna(0)

# =========================================================
# METRICS
# =========================================================
c1, c2, c3, c4 = st.columns(4)

c1.metric("Portfolio Value", f"₹ {total_value:,.0f}")
c2.metric("Annual Return", f"{Rp*100:.2f}%")
c3.metric("Sharpe Ratio", f"{sharpe:.2f}")
c4.metric("Beta", f"{beta:.2f}")

# =========================================================
# HOLDINGS
# =========================================================
st.subheader("Holdings")
st.dataframe(df, use_container_width=True)

# =========================================================
# ATTRIBUTION
# =========================================================
df["Contribution"] = df["Weight"] * df["PnL"]

fig1 = px.bar(
    df,
    x="Ticker",
    y="Contribution",
    color="Contribution",
    color_continuous_scale="Blues"
)
st.plotly_chart(fig1, use_container_width=True)

# =========================================================
# PERFORMANCE CURVE
# =========================================================
st.subheader("Performance Curve")

cumulative = (1 + portfolio_returns).cumprod()

fig2 = go.Figure()
fig2.add_trace(go.Scatter(y=cumulative, line=dict(color="#1F4E79")))
st.plotly_chart(fig2, use_container_width=True)

# =========================================================
# ROLLING BETA
# =========================================================
st.subheader("Rolling Beta")

fig3 = go.Figure()
fig3.add_trace(go.Scatter(y=rolling_beta, line=dict(color="#F59E0B")))
st.plotly_chart(fig3, use_container_width=True)

# =========================================================
# RISK ANALYSIS
# =========================================================
st.subheader("Risk Analysis")

risk = []

if beta > 1.2:
    risk.append("High Market Sensitivity (Beta > 1.2)")
if sharpe < 0:
    risk.append("Negative Risk-Adjusted Returns")
if df["Weight"].max() > 0.5:
    risk.append("Concentration Risk (>50% in one asset)")

if len(risk) == 0:
    risk.append("Portfolio risk is within acceptable limits")

for r in risk:
    st.write("⚠️ " + r if "risk" in r.lower() else "✅ " + r)

# =========================================================
# INSIGHTS (FIXED)
# =========================================================
st.subheader("Insights")

if st.button("Generate Insights"):
    try:
        import ollama

        prompt = f"""
        Portfolio Value: {total_value}
        Sharpe Ratio: {sharpe}
        Beta: {beta}

        Provide institutional hedge fund level insights.
        """

        res = ollama.chat(
            model="qwen2.5:3b",
            messages=[{"role": "user", "content": prompt}]
        )

        st.success(res["message"]["content"])

    except Exception as e:
        st.error(f"Insights error: {str(e)}")

# =========================================================
# EXPORT
# =========================================================
st.download_button(
    "Export Portfolio Data",
    df.to_csv(index=False),
    "portfolio.csv",
    "text/csv"
)