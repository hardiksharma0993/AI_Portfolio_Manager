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

        # Don't guess shape from len(tickers) == 1 — check the actual columns.
        # yfinance's column structure can vary by version even for single tickers.
        is_multi = isinstance(raw.columns, pd.MultiIndex)

        for t in tickers:
            try:
                if is_multi:
                    series = raw[t]["Close"].dropna()
                else:
                    series = raw["Close"].dropna()

                prices[t] = float(series.iloc[-1]) if not series.empty else np.nan
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
def get_historical_prices(df, period="1y"):
    tickers = df["Ticker"].tolist()

    data = yf.download(
        tickers=tickers,
        period=period,
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
# Upside Capture: how much of benchmark UP-day returns the portfolio captures
# Downside Capture: how much of benchmark DOWN-day returns the portfolio captures
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


# ================================
# GENERIC SERIES STATS
# Used to score any equity curve (backtest legs, benchmark, etc.)
# ================================
def compute_series_stats(series, risk_free_rate=0.06):
    series = series.dropna()
    returns = series.pct_change().dropna()

    if returns.empty or len(series) < 2:
        return {"cagr": 0.0, "volatility": 0.0, "sharpe": 0.0, "max_drawdown": 0.0}

    n_years = len(returns) / 252
    total_return = series.iloc[-1] / series.iloc[0]

    cagr = float(total_return ** (1 / n_years) - 1) if n_years > 0 else 0.0
    volatility = float(returns.std() * np.sqrt(252))
    sharpe = (cagr - risk_free_rate) / (volatility + 1e-9)

    cum = series / series.iloc[0]
    dd = cum / cum.cummax() - 1
    max_dd = float(dd.min())

    return {
        "cagr": cagr,
        "volatility": volatility,
        "sharpe": float(sharpe),
        "max_drawdown": max_dd
    }


# ================================
# BACKTEST ENGINE
# Simulates two strategies over the same historical window using the
# CURRENT portfolio's tickers and target weights (derived from live shares):
#   1. Buy & Hold      — weights drift naturally with price moves, never rebalanced
#   2. Rebalanced      — weights pulled back to target at a chosen frequency,
#                        with a simple proportional transaction-cost drag
# ================================
def run_backtest(rebalance_freq="Monthly", transaction_cost_bps=10.0, period="1y"):
    df = load_portfolio()
    prices = get_historical_prices(df, period=period)

    if prices.empty:
        return None

    tickers = [t for t in df["Ticker"].tolist() if t in prices.columns]
    if not tickers:
        return None

    df = df.set_index("Ticker").loc[tickers]
    prices = prices[tickers].ffill().dropna(how="all")
    prices = prices.dropna()  # require full history across all holdings for a clean sim

    if prices.empty or len(prices) < 2:
        return None

    # Target weights derived from most recent prices * current shares held
    latest_prices = prices.iloc[-1]
    dollar_values = df["Shares"] * latest_prices
    target_weights = dollar_values / dollar_values.sum()

    returns = prices.pct_change().fillna(0.0)
    dates = prices.index

    # Determine rebalance dates
    rebal_dates = set()
    if rebalance_freq != "None":
        freq_map = {"Weekly": "W", "Monthly": "ME", "Quarterly": "QE"}
        freq_code = freq_map.get(rebalance_freq, "ME")
        resampled = prices.resample(freq_code).first()
        rebal_dates = set(resampled.index)

    weights = target_weights.copy()
    bh_weights = target_weights.copy()

    equity = [1.0]
    equity_bh = [1.0]
    total_cost = 0.0

    for i in range(1, len(dates)):
        day_ret_active = float((weights * returns.iloc[i]).sum())
        day_ret_bh = float((bh_weights * returns.iloc[i]).sum())

        equity.append(equity[-1] * (1 + day_ret_active))
        equity_bh.append(equity_bh[-1] * (1 + day_ret_bh))

        # Weights drift with price moves each day
        weights = weights * (1 + returns.iloc[i])
        weights = weights / weights.sum()

        bh_weights = bh_weights * (1 + returns.iloc[i])
        bh_weights = bh_weights / bh_weights.sum()

        # On a rebalance date, pull the active leg back to target and charge turnover cost
        if dates[i] in rebal_dates:
            turnover = float((weights - target_weights).abs().sum())
            cost = turnover * (transaction_cost_bps / 10000.0)
            equity[-1] *= (1 - cost)
            total_cost += cost
            weights = target_weights.copy()

    rebal_series = pd.Series(equity, index=dates, name="Rebalanced")
    bh_series = pd.Series(equity_bh, index=dates, name="BuyAndHold")

    return {
        "rebalanced": rebal_series,
        "buy_and_hold": bh_series,
        "total_transaction_cost": total_cost,
        "target_weights": target_weights
    }
