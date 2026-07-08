import pandas as pd
import numpy as np
import yfinance as yf
from scipy.optimize import minimize


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
# HISTORICAL PRICES (BATCH)
# Pass either `period` (e.g. "1y") OR an explicit `start`/`end` date range.
# start/end take priority over period when both are given.
# ================================
def get_historical_prices(df, period="1y", start=None, end=None):
    tickers = df["Ticker"].tolist()

    if start is not None:
        data = yf.download(
            tickers=tickers,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False
        )
    else:
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
# BENCHMARK OPTIONS
# ================================
BENCHMARK_OPTIONS = {
    "NIFTY 50": "^NSEI",
    "Sensex": "^BSESN",
    "Nifty Bank": "^NSEBANK",
    "Nifty Next 50": "^NSMIDCP"
}


def get_benchmark_history(ticker="^NSEI", period="1y", start=None, end=None):
    if start is not None:
        data = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)["Close"]
    else:
        data = yf.download(ticker, period=period, auto_adjust=True, progress=False)["Close"]

    if isinstance(data, pd.DataFrame):
        data = data.iloc[:, 0]

    return data.dropna()


# ================================
# CORRELATION MATRIX
# Daily-return correlation across holdings — complements the efficient
# frontier by showing *why* diversification helps (or doesn't).
# ================================
def get_correlation_matrix(include_tickers=None, period="1y"):
    df = load_portfolio()

    if include_tickers:
        df = df[df["Ticker"].isin(include_tickers)]

    prices = get_historical_prices(df, period=period)
    if prices.empty:
        return None

    tickers = [t for t in df["Ticker"].tolist() if t in prices.columns]
    if len(tickers) < 2:
        return None

    prices = prices[tickers].dropna()
    if prices.empty or len(prices) < 2:
        return None

    returns = prices.pct_change().dropna()
    return returns.corr()


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
#   2. Rebalanced      — weights pulled back to a target at a chosen frequency,
#                        with a simple proportional transaction-cost drag
#
# Filters supported:
#   - include_tickers:   restrict the simulation to a subset of your holdings
#   - period / (start,end): preset lookback OR an explicit custom date range
#   - benchmark_ticker:  overlay an index series over the same window for comparison
#   - risk_free_rate:    used when scoring the two equity curves
#   - rebalance_mode:
#       "fixed"     — rebalance back to the current live target weights (default)
#       "optimized" — walk-forward: at each rebalance date, re-solve for the
#                     max-Sharpe weights using only trailing data up to that
#                     date (no lookahead), then hold until the next rebalance
# ================================
def run_backtest(
    rebalance_freq="Monthly",
    transaction_cost_bps=10.0,
    period="1y",
    start=None,
    end=None,
    include_tickers=None,
    benchmark_ticker="^NSEI",
    risk_free_rate=0.06,
    rebalance_mode="fixed",
    optimizer_lookback_days=126
):
    df = load_portfolio()

    if include_tickers:
        df = df[df["Ticker"].isin(include_tickers)]
        if df.empty:
            return None

    prices = get_historical_prices(df, period=period, start=start, end=end)

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

        # On a rebalance date, reset the active leg and charge turnover cost
        if dates[i] in rebal_dates:
            if rebalance_mode == "optimized":
                # Walk-forward: only use data up to (and including) today — no lookahead
                trailing = returns.iloc[max(0, i - optimizer_lookback_days):i + 1]
                if len(trailing) >= 30:
                    mean_ret = trailing.mean() * 252
                    cov = trailing.cov() * 252
                    try:
                        new_weights_arr = optimize_max_sharpe(mean_ret, cov, risk_free_rate, bounds=(0, 1))
                        rebalance_target = pd.Series(new_weights_arr, index=mean_ret.index)
                    except Exception:
                        rebalance_target = target_weights
                else:
                    rebalance_target = target_weights
            else:
                rebalance_target = target_weights

            turnover = float((weights - rebalance_target).abs().sum())
            cost = turnover * (transaction_cost_bps / 10000.0)
            equity[-1] *= (1 - cost)
            total_cost += cost
            weights = rebalance_target.copy()

    rebal_series = pd.Series(equity, index=dates, name="Rebalanced")
    bh_series = pd.Series(equity_bh, index=dates, name="BuyAndHold")

    result = {
        "rebalanced": rebal_series,
        "buy_and_hold": bh_series,
        "total_transaction_cost": total_cost,
        "target_weights": target_weights,
        "tickers_used": tickers
    }

    # Optional benchmark overlay over the identical window
    if benchmark_ticker:
        try:
            bench = get_benchmark_history(benchmark_ticker, period=period, start=dates[0], end=dates[-1])
            bench = bench.reindex(dates).ffill().dropna()
            if not bench.empty:
                result["benchmark"] = bench / bench.iloc[0]
        except Exception:
            pass

    return result


# ================================
# MEAN-VARIANCE OPTIMIZER / EFFICIENT FRONTIER
# Classic Markowitz optimization using scipy.optimize (SLSQP).
# All weights sum to 1. Long-only by default (bounds 0-1); pass
# bounds=(-1, 1) to allow shorting.
# ================================
def portfolio_performance(weights, mean_returns, cov_matrix):
    """Returns (annualised return, annualised volatility) for a weight vector."""
    weights = np.array(weights)
    ret = float(np.dot(weights, mean_returns))
    vol = float(np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))))
    return ret, vol


def _neg_sharpe(weights, mean_returns, cov_matrix, risk_free_rate):
    ret, vol = portfolio_performance(weights, mean_returns, cov_matrix)
    return -(ret - risk_free_rate) / (vol + 1e-9)


def _portfolio_vol(weights, mean_returns, cov_matrix):
    return portfolio_performance(weights, mean_returns, cov_matrix)[1]


def optimize_max_sharpe(mean_returns, cov_matrix, risk_free_rate=0.06, bounds=(0, 1)):
    """Solves for the weight vector that maximizes the Sharpe ratio (the tangency portfolio)."""
    n = len(mean_returns)
    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1},)
    bounds_list = tuple(bounds for _ in range(n))
    init_guess = np.array(n * [1.0 / n])

    result = minimize(
        _neg_sharpe, init_guess, args=(mean_returns, cov_matrix, risk_free_rate),
        method="SLSQP", bounds=bounds_list, constraints=constraints
    )
    return result.x if result.success else init_guess


def optimize_min_volatility(mean_returns, cov_matrix, target_return=None, bounds=(0, 1)):
    """Solves for the minimum-volatility weight vector, optionally constrained to a target return."""
    n = len(mean_returns)
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]

    if target_return is not None:
        constraints.append({
            "type": "eq",
            "fun": lambda w: portfolio_performance(w, mean_returns, cov_matrix)[0] - target_return
        })

    bounds_list = tuple(bounds for _ in range(n))
    init_guess = np.array(n * [1.0 / n])

    result = minimize(
        _portfolio_vol, init_guess, args=(mean_returns, cov_matrix),
        method="SLSQP", bounds=bounds_list, constraints=constraints
    )
    return result.x if result.success else init_guess


def compute_efficient_frontier(mean_returns, cov_matrix, n_points=40, bounds=(0, 1)):
    """Sweeps target returns from min to max asset return and solves min-vol at each point."""
    min_ret = float(mean_returns.min())
    max_ret = float(mean_returns.max())
    target_returns = np.linspace(min_ret, max_ret, n_points)

    frontier_ret, frontier_vol = [], []
    for t in target_returns:
        try:
            w = optimize_min_volatility(mean_returns, cov_matrix, target_return=t, bounds=bounds)
            ret, vol = portfolio_performance(w, mean_returns, cov_matrix)
            frontier_ret.append(ret)
            frontier_vol.append(vol)
        except Exception:
            continue

    return frontier_ret, frontier_vol


def run_optimizer(include_tickers=None, period="1y", risk_free_rate=0.06, allow_short=False, n_frontier_points=40):
    """
    End-to-end optimizer: pulls historical data for the current holdings (or a
    subset via include_tickers), computes the max-Sharpe and min-volatility
    portfolios, the efficient frontier, and compares both against your
    current live weights.
    """
    df = load_portfolio()

    if include_tickers:
        df = df[df["Ticker"].isin(include_tickers)]
        if len(df) < 2:
            return None  # need at least 2 assets for a meaningful frontier

    prices = get_historical_prices(df, period=period)
    if prices.empty:
        return None

    tickers = [t for t in df["Ticker"].tolist() if t in prices.columns]
    if len(tickers) < 2:
        return None

    df = df.set_index("Ticker").loc[tickers]
    prices = prices[tickers].ffill().dropna()

    if prices.empty or len(prices) < 30:
        return None

    returns = prices.pct_change().dropna()
    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252

    bounds = (-1.0, 1.0) if allow_short else (0.0, 1.0)

    max_sharpe_w = optimize_max_sharpe(mean_returns, cov_matrix, risk_free_rate, bounds)
    min_vol_w = optimize_min_volatility(mean_returns, cov_matrix, bounds=bounds)

    max_sharpe_ret, max_sharpe_vol = portfolio_performance(max_sharpe_w, mean_returns, cov_matrix)
    min_vol_ret, min_vol_vol = portfolio_performance(min_vol_w, mean_returns, cov_matrix)

    # Current live weights, for comparison against the optimized portfolios
    latest_prices = prices.iloc[-1]
    dollar_values = df["Shares"] * latest_prices
    current_weights = (dollar_values / dollar_values.sum())
    current_weights = current_weights.reindex(tickers).values
    current_ret, current_vol = portfolio_performance(current_weights, mean_returns, cov_matrix)

    frontier_ret, frontier_vol = compute_efficient_frontier(
        mean_returns, cov_matrix, n_points=n_frontier_points, bounds=bounds
    )

    return {
        "tickers": tickers,
        "mean_returns": mean_returns,
        "cov_matrix": cov_matrix,
        "max_sharpe_weights": dict(zip(tickers, max_sharpe_w)),
        "max_sharpe_return": max_sharpe_ret,
        "max_sharpe_vol": max_sharpe_vol,
        "min_vol_weights": dict(zip(tickers, min_vol_w)),
        "min_vol_return": min_vol_ret,
        "min_vol_vol": min_vol_vol,
        "current_weights": dict(zip(tickers, current_weights)),
        "current_return": current_ret,
        "current_vol": current_vol,
        "frontier_returns": frontier_ret,
        "frontier_vols": frontier_vol,
        "asset_returns": mean_returns.to_dict(),
        "asset_vols": {t: float(np.sqrt(cov_matrix.loc[t, t])) for t in tickers}
    }
