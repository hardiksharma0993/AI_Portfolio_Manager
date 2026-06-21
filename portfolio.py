import pandas as pd
import numpy as np
import yfinance as yf


# ================================
# LOAD PORTFOLIO
# ================================
def load_portfolio():
    return pd.read_csv("data/portfolio.csv")


# ================================
# CURRENT PRICES
# ================================
def get_current_prices(df):
    prices = {}

    for t in df["Ticker"]:
        try:
            prices[t] = yf.Ticker(t).history(period="5d")["Close"].iloc[-1]
        except:
            prices[t] = np.nan

    return prices


# ================================
# HISTORICAL PRICES
# ================================
def get_historical_prices(df):

    tickers = df["Ticker"].tolist()

    data = yf.download(
        tickers=tickers,
        period="1y",
        auto_adjust=True,
        progress=False
    )

    if isinstance(data.columns, pd.MultiIndex):
        data = data["Close"]

    return data.dropna(how="all")


# ================================
# ✅ THIS IS THE MISSING FUNCTION
# ================================
def get_portfolio_history():

    df = load_portfolio()
    prices = get_historical_prices(df)

    portfolio = pd.Series(0.0, index=prices.index)

    for _, row in df.iterrows():
        if row["Ticker"] in prices.columns:
            portfolio += prices[row["Ticker"]] * row["Shares"]

    return portfolio


# ================================
# TOTAL VALUE
# ================================
def get_total_value():

    df = load_portfolio()
    prices = get_current_prices(df)

    return float(sum(prices[t] * s for t, s in zip(df["Ticker"], df["Shares"])))


# ================================
# METRICS
# ================================
def get_metrics():

    portfolio = get_portfolio_history()
    returns = portfolio.pct_change().dropna()

    benchmark = yf.download("^NSEI", period="1y")["Close"].pct_change().dropna()

    aligned = pd.concat([returns, benchmark], axis=1).dropna()
    aligned.columns = ["p", "b"]

    annual_return = returns.mean() * 252
    volatility = returns.std() * np.sqrt(252)

    sharpe = (annual_return - 0.06) / (volatility + 1e-9)

    beta = aligned["p"].cov(aligned["b"]) / (aligned["b"].var() + 1e-9)

    cum = (1 + returns).cumprod()
    dd = cum / cum.cummax() - 1

    return {
        "annual_return": float(annual_return),
        "volatility": float(volatility),
        "sharpe": float(sharpe),
        "beta": float(beta),
        "max_drawdown": float(dd.min()),
        "returns": returns,
        "benchmark_returns": benchmark
    }


# ================================
# SECTOR MAP
# ================================
def get_sector_map():
    return {
        "RELIANCE.NS": "Energy",
        "TCS.NS": "IT",
        "INFY.NS": "IT",
        "HDFCBANK.NS": "Banking",
        "ITC.NS": "FMCG",
        "NIFTYBEES.NS": "ETF",
        "GOLDBEES.NS": "Gold"
    }