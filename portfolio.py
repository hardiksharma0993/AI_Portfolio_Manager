import pandas as pd
import numpy as np
import yfinance as yf


# ================================
# LOAD PORTFOLIO
# ================================
def load_portfolio():
    return pd.read_csv("data/portfolio.csv")


# ================================
# CURRENT PRICES (BATCH - FAST & RELIABLE)
# Uses last available close — works even when market is closed
# ================================
def get_current_prices(df):
    tickers = df["Ticker"].tolist()
    prices = {}

    try:
        # Batch download — much faster and avoids per-ticker rate limits
        raw = yf.download(
            tickers=tickers,
            period="5d",
            auto_adjust=True,
            progress=False,
            group_by="ticker"
        )

        for t in tickers:
            try:
                if len(tickers) == 1:
                    # Single ticker: columns are flat
                    series = raw["Close"].dropna()
                else:
                    series = raw[t]["Close"].dropna()

                if not series.empty:
                    prices[t] = float(series.iloc[-1])
                else:
                    prices[t] = 0.0
            except Exception:
                prices[t] = np.nan

    except Exception:
        # Fallback: individual fetch if batch fails
        for t in tickers:
            try:
                hist = yf.Ticker(t).history(period="5d")["Close"].dropna()
                prices[t] = float(hist.iloc[-1]) if not hist.empty else np.nan
            except Exception:
                prices[t] = np.nan

    return prices


# ================================
# HISTORICAL PRICES (1Y, BATCH)
# ================================
def get_historical_prices(df):
    tickers = df["Ticker"].tolist()

    data = yf.download(
        tickers=tickers,
        period="1y",
        auto_adjust=True,
        progress=False
    )

    # Flatten MultiIndex — always keep "Close" level
    if isinstance(data.columns, pd.MultiIndex):
        data = data["Close"]

    # If only one ticker was downloaded, yfinance returns a Series — convert
    if isinstance(data, pd.Series):
        data = data.to_frame(name=tickers[0])

    return data.dropna(how="all")


# ================================
# PORTFOLIO HISTORY (WEIGHTED SUM OF HOLDINGS)
# ================================
def get_portfolio_history():
    df = load_portfolio()
    prices = get_historical_prices(df)

    if prices.empty:
    return pd.Series(dtype=float)

portfolio = pd.Series(0.0, index=prices.index)

    for _, row in df.iterrows():
        ticker = row["Ticker"]
        shares = row["Shares"]
        if ticker in prices.columns:
            portfolio += prices[ticker].ffill() * shares

    return portfolio


# ================================
# TOTAL VALUE
# ================================
def get_total_value():
    df = load_portfolio()
    prices = get_current_prices(df)

    total = 0.0
    for _, row in df.iterrows():
        price = prices.get(row["Ticker"], np.nan)
        if not np.isnan(price):
            total += price * row["Shares"]

    return total


# ================================
# DRAWDOWN SERIES
# Returns a pd.Series of rolling drawdown from peak (negative values)
# ================================
def get_drawdown_series(portfolio=None):
    if portfolio is None:
        portfolio = get_portfolio_history()

    cumulative = (1 + portfolio.pct_change().dropna()).cumprod()
    rolling_max = cumulative.cummax()
    drawdown = (cumulative - rolling_max) / rolling_max

    return drawdown


# ================================
# CAPTURE RATIOS
# Upside Capture: how much of benchmark UP months portfolio captures
# Downside Capture: how much of benchmark DOWN months portfolio captures
# Ratio > 100% upside and < 100% downside = ideal active manager
# ================================
def get_capture_ratios(portfolio_returns=None, benchmark_returns=None):
    # Always safe fallback
    if portfolio_returns is None or benchmark_returns is None:
        metrics = get_metrics()
        portfolio_returns = metrics.get("returns", pd.Series(dtype=float))
        benchmark_returns = metrics.get("benchmark_returns", pd.Series(dtype=float))

    aligned = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()

    if aligned.empty:
        return {"upside_capture": 0.0, "downside_capture": 0.0}

    aligned.columns = ["p", "b"]

    up = aligned[aligned["b"] > 0]
    down = aligned[aligned["b"] < 0]

    def capture(p, b):
        if len(p) == 0 or len(b) == 0:
            return 0.0
        try:
            return (p.mean() / (b.mean() + 1e-9)) * 100
        except Exception:
            return 0.0

    return {
        "upside_capture": float(capture(up["p"], up["b"])),
        "downside_capture": float(capture(down["p"], down["b"]))
    }


# ================================
# FULL METRICS
# ================================
def get_metrics():
    portfolio = get_portfolio_history()
    returns = portfolio.pct_change().dropna()

    benchmark_raw = yf.download("^NSEI", period="1y", auto_adjust=True, progress=False)["Close"]
    benchmark = benchmark_raw.pct_change().dropna()

    # Flatten if MultiIndex (rare but possible)
    if isinstance(benchmark, pd.DataFrame):
        benchmark = benchmark.iloc[:, 0]

    aligned = pd.concat([returns, benchmark], axis=1).dropna()

if aligned.empty:
    return {
        "annual_return": 0.0,
        "volatility": 0.0,
        "sharpe": 0.0,
        "beta": 0.0,
        "max_drawdown": 0.0,
        "returns": pd.Series(dtype=float),
        "benchmark_returns": pd.Series(dtype=float)
    }

aligned.columns = ["p", "b"]

    annual_return = float(aligned["p"].mean() * 252)
    volatility = float(aligned["p"].std() * np.sqrt(252))
    risk_free_rate = 0.06  # 6% — approximate Indian risk-free rate

    sharpe = (annual_return - risk_free_rate) / (volatility + 1e-9)

    beta = float(aligned["p"].cov(aligned["b"]) / (aligned["b"].var() + 1e-9))

    cum = (1 + aligned["p"]).cumprod()
    dd = cum / cum.cummax() - 1
    max_drawdown = float(dd.min())

    return {
        "annual_return": annual_return,
        "volatility": volatility,
        "sharpe": float(sharpe),
        "beta": beta,
        "max_drawdown": max_drawdown,
        "returns": aligned["p"],
        "benchmark_returns": aligned["b"]
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
