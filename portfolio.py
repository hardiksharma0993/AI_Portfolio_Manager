import pandas as pd
import numpy as np
import yfinance as yf
import datetime
import io
from scipy.optimize import minimize
from scipy.stats import norm, skew, kurtosis


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


# ================================
# VALUE AT RISK (VaR) & CONDITIONAL VaR (CVaR)
# Both are expressed as POSITIVE numbers representing the magnitude of loss
# (e.g. 0.023 means "a 2.3% loss"), at a given confidence level and horizon.
#
# - "historical": empirical percentile of actual past daily returns — makes
#   no distributional assumption, but is limited by how much history you have.
# - "parametric": assumes daily returns are normally distributed — smoother,
#   but can understate real-world tail risk (fat tails) common in equities.
#
# horizon_days scales the 1-day estimate using the sqrt(time) rule, a common
# approximation that assumes i.i.d. returns — a simplification worth noting
# when the horizon is longer.
# ================================
def compute_var_cvar(returns, confidence=0.95, method="historical"):
    returns = returns.dropna()
    if returns.empty:
        return {"var": 0.0, "cvar": 0.0}

    alpha = 1 - confidence

    if method == "parametric":
        mu = float(returns.mean())
        sigma = float(returns.std())
        z = norm.ppf(alpha)  # negative value
        var = -(mu + z * sigma)
        # Expected shortfall under a normal-distribution assumption
        cvar = -(mu - sigma * norm.pdf(z) / alpha)
    else:
        var = -float(np.percentile(returns, alpha * 100))
        tail = returns[returns <= -var]
        cvar = -float(tail.mean()) if not tail.empty else var

    return {"var": float(var), "cvar": float(cvar)}


def get_portfolio_var_cvar(confidence=0.95, method="historical", horizon_days=1):
    """
    Convenience wrapper: pulls the portfolio's own daily return series via
    get_metrics() and scores VaR/CVaR, scaled to the requested horizon and
    also converted to an approximate ₹ amount using the current total value.
    """
    metrics = get_metrics()
    returns = metrics.get("returns", pd.Series(dtype=float))

    result = compute_var_cvar(returns, confidence=confidence, method=method)
    scale = np.sqrt(horizon_days)

    var_pct = result["var"] * scale
    cvar_pct = result["cvar"] * scale

    total_value = get_total_value()

    return {
        "var_pct": var_pct,
        "cvar_pct": cvar_pct,
        "var_amount": var_pct * total_value,
        "cvar_amount": cvar_pct * total_value,
        "confidence": confidence,
        "horizon_days": horizon_days,
        "method": method
    }


# ================================
# ROLLING RISK METRICS
# Rolling Sharpe, Beta, and annualised Volatility over a trailing window —
# shows how the portfolio's risk profile has evolved, rather than a single
# point-in-time snapshot.
# ================================
def get_rolling_metrics(window_days=63, risk_free_rate=0.06):
    metrics = get_metrics()
    returns = metrics.get("returns", pd.Series(dtype=float))
    benchmark_returns = metrics.get("benchmark_returns", pd.Series(dtype=float))

    aligned = pd.concat([returns, benchmark_returns], axis=1).dropna()
    if aligned.empty or len(aligned) <= window_days:
        return pd.DataFrame(columns=["Rolling Sharpe", "Rolling Beta", "Rolling Volatility"])

    aligned.columns = ["p", "b"]

    rolling_vol = aligned["p"].rolling(window_days).std() * np.sqrt(252)
    rolling_mean_annualised = aligned["p"].rolling(window_days).mean() * 252
    rolling_sharpe = (rolling_mean_annualised - risk_free_rate) / (rolling_vol + 1e-9)

    rolling_cov = aligned["p"].rolling(window_days).cov(aligned["b"])
    rolling_var_b = aligned["b"].rolling(window_days).var()
    rolling_beta = rolling_cov / (rolling_var_b + 1e-9)

    out = pd.DataFrame({
        "Rolling Sharpe": rolling_sharpe,
        "Rolling Beta": rolling_beta,
        "Rolling Volatility": rolling_vol
    }).dropna()

    return out


# ================================
# STRESS TESTING
# Approximates the portfolio's impact under hypothetical/historical index
# shocks using the portfolio's CURRENT beta as a linear scaling factor:
#     estimated portfolio return  =  beta  x  index shock
#
# This is a simplified, single-factor approximation — it does NOT re-price
# individual holdings under each historical scenario, and assumes beta stays
# constant during a shock (in reality, correlations often rise in a crisis,
# which can make real losses worse than a simple beta scaling suggests).
# ================================
DEFAULT_STRESS_SCENARIOS = {
    "NIFTY -5% (Mild Correction)": -0.05,
    "NIFTY -10% (Correction)": -0.10,
    "NIFTY -20% (Bear Market)": -0.20,
    "COVID Crash (Feb-Mar 2020, NIFTY approx. -38%)": -0.38,
    "Global Financial Crisis (2008, NIFTY approx. -55%)": -0.55,
    "NIFTY +10% (Rally)": 0.10,
}


def run_stress_test(scenarios=None):
    if scenarios is None:
        scenarios = DEFAULT_STRESS_SCENARIOS

    metrics = get_metrics()
    beta = metrics.get("beta", 0.0)
    total_value = get_total_value()

    rows = []
    for name, shock in scenarios.items():
        estimated_return = beta * shock
        estimated_value_change = total_value * estimated_return
        rows.append({
            "Scenario": name,
            "Index Shock": shock,
            "Est. Portfolio Impact": estimated_return,
            "Est. Rupee Impact": estimated_value_change
        })

    return pd.DataFrame(rows).set_index("Scenario")


# ================================
# ADVANCED RISK-ADJUSTED RETURN RATIOS
# Sortino Ratio: like Sharpe, but only penalises downside volatility —
#   a more realistic risk measure since investors don't mind upside swings.
# Tracking Error: how far the portfolio's daily returns stray from the
#   benchmark's — a standard measure of active risk.
# Information Ratio: active return / tracking error — answers whether the
#   active risk being taken is actually being rewarded.
# ================================
def get_advanced_ratios(risk_free_rate=0.06):
    metrics = get_metrics()
    returns = metrics.get("returns", pd.Series(dtype=float))
    benchmark_returns = metrics.get("benchmark_returns", pd.Series(dtype=float))

    if returns.empty:
        return {"sortino": 0.0, "tracking_error": 0.0, "information_ratio": 0.0,
                "active_return": 0.0, "downside_deviation": 0.0}

    annual_return = metrics["annual_return"]

    downside_returns = returns[returns < 0]
    downside_deviation = float(downside_returns.std() * np.sqrt(252)) if not downside_returns.empty else 0.0
    sortino = (annual_return - risk_free_rate) / (downside_deviation + 1e-9)

    aligned = pd.concat([returns, benchmark_returns], axis=1).dropna()
    if aligned.empty:
        return {"sortino": float(sortino), "tracking_error": 0.0, "information_ratio": 0.0,
                "active_return": 0.0, "downside_deviation": downside_deviation}

    aligned.columns = ["p", "b"]
    active_returns = aligned["p"] - aligned["b"]
    tracking_error = float(active_returns.std() * np.sqrt(252))
    active_return_annual = float(active_returns.mean() * 252)
    information_ratio = active_return_annual / (tracking_error + 1e-9)

    return {
        "sortino": float(sortino),
        "tracking_error": tracking_error,
        "information_ratio": float(information_ratio),
        "active_return": active_return_annual,
        "downside_deviation": downside_deviation
    }


# ================================
# RISK DECOMPOSITION — Component Risk Contribution & Component VaR
# Shows how much each holding contributes to TOTAL portfolio risk, which can
# differ meaningfully from its weight (a stock can be 10% of value but 25% of
# risk if it's volatile and/or highly correlated with the rest of the book).
# Uses the standard Euler/marginal-contribution decomposition:
#   portfolio_vol = sqrt(w' Sigma w)
#   marginal_i    = (Sigma w)_i / portfolio_vol
#   component_i   = w_i * marginal_i        (sums exactly to portfolio_vol)
# Component VaR allocates the portfolio's total VaR proportionally to each
# holding's share of total risk (an exact allocation under the
# variance-covariance/parametric VaR framework).
# ================================
def get_risk_contributions(period="1y", confidence=0.95, var_method="historical", horizon_days=1):
    df = load_portfolio()
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
    cov = returns.cov().values * 252

    latest_prices = prices.iloc[-1]
    dollar_values = df["Shares"] * latest_prices
    weights = (dollar_values / dollar_values.sum()).values

    port_var = float(weights @ cov @ weights)
    port_vol = np.sqrt(port_var) if port_var > 0 else 1e-9

    marginal = (cov @ weights) / port_vol
    component = weights * marginal
    pct_contribution = component / port_vol

    total_var_result = get_portfolio_var_cvar(confidence=confidence, method=var_method, horizon_days=horizon_days)

    rows = []
    for i, t in enumerate(tickers):
        rows.append({
            "Ticker": t,
            "Weight": weights[i],
            "% of Portfolio Risk": float(pct_contribution[i]),
            "Risk-to-Weight Ratio": float(pct_contribution[i] / weights[i]) if weights[i] > 0 else np.nan,
            "Component VaR (%)": float(pct_contribution[i]) * total_var_result["var_pct"],
            "Component VaR (₹)": float(pct_contribution[i]) * total_var_result["var_amount"]
        })

    return pd.DataFrame(rows).set_index("Ticker")


# ================================
# CONCENTRATION RISK — Herfindahl-Hirschman Index (HHI)
# Standard concentration measure: sum of squared weights, scaled to 0-10,000.
# The <1500 / 1500-2500 / >2500 bands are borrowed from the convention used
# in antitrust/merger analysis as a widely recognised rule-of-thumb scale —
# adapted here to portfolio concentration rather than market concentration.
# ================================
def get_concentration_index(weights, basis="holding"):
    weights = pd.Series(weights).dropna()
    if weights.empty:
        return {"hhi": 0.0, "label": "N/A", "basis": basis}

    weights = weights / weights.sum()
    hhi = float((weights ** 2).sum() * 10000)

    if hhi < 1500:
        label = "Low Concentration (Well Diversified)"
    elif hhi < 2500:
        label = "Moderate Concentration"
    else:
        label = "High Concentration"

    return {"hhi": hhi, "label": label, "basis": basis}


# ================================
# TAIL RISK — Skewness & Excess Kurtosis
# Quantifies exactly how far the portfolio's actual return distribution
# deviates from the normal-distribution assumption used by parametric VaR.
# Skewness: negative = more/larger down days than up days (typical for equities).
# Excess Kurtosis: >0 means "fatter tails" than normal — more extreme days
# than a normal distribution would predict.
# ================================
def get_tail_risk_stats():
    metrics = get_metrics()
    returns = metrics.get("returns", pd.Series(dtype=float)).dropna()

    if returns.empty or len(returns) < 10:
        return {"skewness": 0.0, "kurtosis": 0.0}

    return {
        "skewness": float(skew(returns)),
        "kurtosis": float(kurtosis(returns))  # excess kurtosis (normal = 0)
    }


# ================================
# MONTE CARLO VaR / CVaR
# Simulates many possible future return paths and reads VaR/CVaR off the
# resulting distribution of outcomes, rather than a single closed-form estimate.
#   "bootstrap": resamples actual historical daily returns with replacement —
#                makes no distributional assumption, preserves real fat tails.
#   "parametric": draws from a normal distribution using historical mean/std —
#                 smoother, but inherits the same normality limitation as
#                 parametric VaR.
# ================================
def run_monte_carlo_var(n_simulations=5000, horizon_days=21, confidence=0.95, method="bootstrap"):
    metrics = get_metrics()
    returns = metrics.get("returns", pd.Series(dtype=float)).dropna()

    if returns.empty or len(returns) < 30:
        return None

    total_value = get_total_value()
    returns_arr = returns.values
    rng = np.random.default_rng()

    if method == "parametric":
        mu, sigma = returns_arr.mean(), returns_arr.std()
        daily_draws = rng.normal(mu, sigma, size=(n_simulations, horizon_days))
    else:
        daily_draws = rng.choice(returns_arr, size=(n_simulations, horizon_days), replace=True)

    cum_returns = (1 + daily_draws).prod(axis=1) - 1

    alpha = 1 - confidence
    var_pct = -float(np.percentile(cum_returns, alpha * 100))
    tail = cum_returns[cum_returns <= -var_pct]
    cvar_pct = -float(tail.mean()) if len(tail) > 0 else var_pct

    return {
        "var_pct": var_pct,
        "cvar_pct": cvar_pct,
        "var_amount": var_pct * total_value,
        "cvar_amount": cvar_pct * total_value,
        "simulated_returns": cum_returns,
        "mean_simulated_return": float(cum_returns.mean()),
        "n_simulations": n_simulations,
        "horizon_days": horizon_days,
        "confidence": confidence,
        "method": method
    }


# ================================
# LIQUIDITY RISK
# Estimates how many trading days it would take to fully exit each position
# without dominating its own trading volume, using a standard "participation
# rate" assumption (the max share of a stock's average daily volume you
# trade in a single day to limit market impact — 10% is a common convention).
# ================================
def get_volume_history(df, period="1mo"):
    tickers = df["Ticker"].tolist()

    try:
        data = yf.download(tickers=tickers, period=period, auto_adjust=True, progress=False, group_by="ticker")
        is_multi = isinstance(data.columns, pd.MultiIndex)
    except Exception:
        return {t: pd.Series(dtype=float) for t in tickers}

    volumes = {}
    for t in tickers:
        try:
            vol_series = data[t]["Volume"].dropna() if is_multi else data["Volume"].dropna()
            volumes[t] = vol_series
        except Exception:
            volumes[t] = pd.Series(dtype=float)

    return volumes


def get_liquidity_metrics(participation_rate=0.10, period="1mo"):
    df = load_portfolio()
    volumes = get_volume_history(df, period=period)
    prices_dict = get_current_prices(df)

    rows = []
    for _, row in df.iterrows():
        t = row["Ticker"]
        shares = row["Shares"]
        vol_series = volumes.get(t, pd.Series(dtype=float))
        adv = float(vol_series.mean()) if not vol_series.empty else np.nan

        if adv and adv > 0 and not np.isnan(adv):
            tradable_per_day = adv * participation_rate
            days_to_liquidate = shares / tradable_per_day if tradable_per_day > 0 else np.nan
        else:
            days_to_liquidate = np.nan

        if pd.isna(days_to_liquidate):
            flag = "Unknown (no volume data)"
        elif days_to_liquidate <= 1:
            flag = "High Liquidity"
        elif days_to_liquidate <= 5:
            flag = "Moderate Liquidity"
        else:
            flag = "Low Liquidity"

        rows.append({
            "Ticker": t,
            "Shares Held": shares,
            "Avg Daily Volume": adv,
            "Days to Liquidate": days_to_liquidate,
            "Liquidity Flag": flag
        })

    return pd.DataFrame(rows).set_index("Ticker")


# ================================
# SINGLE-FACTOR (MARKET) RISK DECOMPOSITION
# Splits each holding's — and the portfolio's — total variance into:
#   Systematic variance   = beta^2 x benchmark variance   (market-driven risk)
#   Idiosyncratic variance = total variance - systematic  (stock-specific risk)
# This is the classical single-index/CAPM decomposition. It does NOT
# decompose risk into style factors (size, value, momentum, quality, etc.) —
# a genuine multi-factor model needs external factor-return datasets that
# aren't available through this pipeline. Labelled explicitly as a
# single-factor (market) model for that reason.
# ================================
def get_factor_decomposition(period="1y"):
    df = load_portfolio()
    prices = get_historical_prices(df, period=period)
    if prices.empty:
        return None

    tickers = [t for t in df["Ticker"].tolist() if t in prices.columns]
    if not tickers:
        return None

    df_idx = df.set_index("Ticker").loc[tickers]
    prices = prices[tickers].ffill().dropna()
    if prices.empty or len(prices) < 30:
        return None

    bench = get_benchmark_history("^NSEI", period=period)
    asset_returns = prices.pct_change().dropna()
    bench_returns = bench.pct_change().dropna()

    aligned = asset_returns.join(bench_returns.rename("BENCH"), how="inner").dropna()
    if aligned.empty:
        return None

    bench_var = float(aligned["BENCH"].var())
    if bench_var <= 0:
        return None

    latest_prices = prices.iloc[-1]
    dollar_values = df_idx["Shares"] * latest_prices
    weights = dollar_values / dollar_values.sum()

    asset_rows = []
    for t in tickers:
        stock_ret = aligned[t]
        cov_tb = float(stock_ret.cov(aligned["BENCH"]))
        beta_i = cov_tb / bench_var
        corr_i = stock_ret.corr(aligned["BENCH"])
        r_squared = float(corr_i ** 2) if pd.notna(corr_i) else 0.0

        total_var_i = float(stock_ret.var() * 252)
        systematic_var_i = min((beta_i ** 2) * bench_var * 252, total_var_i) if total_var_i > 0 else 0.0
        idiosyncratic_var_i = max(total_var_i - systematic_var_i, 0.0)

        asset_rows.append({
            "Ticker": t,
            "Weight": float(weights[t]),
            "Beta": beta_i,
            "R-Squared": r_squared,
            "Systematic Var %": (systematic_var_i / total_var_i * 100) if total_var_i > 0 else 0.0,
            "Idiosyncratic Var %": (idiosyncratic_var_i / total_var_i * 100) if total_var_i > 0 else 0.0
        })

    asset_df = pd.DataFrame(asset_rows).set_index("Ticker")

    portfolio_returns = (aligned[tickers] * weights[tickers]).sum(axis=1)
    port_cov_b = float(portfolio_returns.cov(aligned["BENCH"]))
    port_beta = port_cov_b / bench_var
    port_total_var = float(portfolio_returns.var() * 252)
    port_systematic_var = min((port_beta ** 2) * bench_var * 252, port_total_var) if port_total_var > 0 else 0.0
    port_idio_var = max(port_total_var - port_systematic_var, 0.0)

    portfolio_level = {
        "beta": port_beta,
        "systematic_pct": (port_systematic_var / port_total_var * 100) if port_total_var > 0 else 0.0,
        "idiosyncratic_pct": (port_idio_var / port_total_var * 100) if port_total_var > 0 else 0.0,
        "total_annual_variance": port_total_var
    }

    return {"portfolio": portfolio_level, "assets": asset_df}


# ================================
# SECTOR CONTRIBUTION & ALLOCATION TILT
# NOTE: This is NOT full Brinson-Fachler attribution. True Brinson requires
# benchmark SECTOR-LEVEL returns (what NIFTY 50 did within Banking, IT, etc.
# specifically), which isn't available through this pipeline. Instead:
#   - "Contribution to Portfolio Return" is EXACT — wp_i x actual sector
#     return, computed from your real holdings. These sum precisely to your
#     total portfolio return over the window.
#   - "Approx. Benchmark Weight" and "Weight Tilt" use ILLUSTRATIVE NIFTY 50
#     sector weights (a reasonable approximation of typical index
#     composition) — refresh DEFAULT_BENCHMARK_SECTOR_WEIGHTS from the
#     latest NSE factsheet for precision.
# ================================
DEFAULT_BENCHMARK_SECTOR_WEIGHTS = {
    "Banking": 0.35,
    "IT": 0.13,
    "Energy": 0.10,
    "FMCG": 0.08,
    "Other": 0.34,   # residual bucket: autos, pharma, metals, telecom, construction, power, etc.
    "ETF": 0.0,
    "Gold": 0.0
}


def get_sector_attribution(period="1y", benchmark_sector_weights=None):
    if benchmark_sector_weights is None:
        benchmark_sector_weights = DEFAULT_BENCHMARK_SECTOR_WEIGHTS

    df = load_portfolio()
    prices = get_historical_prices(df, period=period)
    if prices.empty:
        return None

    tickers = [t for t in df["Ticker"].tolist() if t in prices.columns]
    if not tickers:
        return None

    df = df.set_index("Ticker").loc[tickers]
    prices = prices[tickers].ffill().dropna()
    if prices.empty or len(prices) < 2:
        return None

    sector_map = get_sector_map()
    df["Sector"] = [sector_map.get(t, "Other") for t in df.index]

    first_prices = prices.iloc[0]
    last_prices = prices.iloc[-1]
    stock_total_return = (last_prices - first_prices) / first_prices

    dollar_values = df["Shares"] * last_prices
    port_weights = dollar_values / dollar_values.sum()

    bench = get_benchmark_history("^NSEI", period=period)
    bench_aligned = bench.reindex(prices.index).dropna()
    if len(bench_aligned) < 2:
        return None
    bench_total_return = float((bench_aligned.iloc[-1] - bench_aligned.iloc[0]) / bench_aligned.iloc[0])

    sector_agg = {}
    for t in tickers:
        sec = df.loc[t, "Sector"]
        w = float(port_weights[t])
        r = float(stock_total_return[t])
        if sec not in sector_agg:
            sector_agg[sec] = {"weight": 0.0, "contribution": 0.0}
        sector_agg[sec]["weight"] += w
        sector_agg[sec]["contribution"] += w * r

    rows = []
    for sec, vals in sector_agg.items():
        wp = vals["weight"]
        sector_return = vals["contribution"] / wp if wp > 0 else 0.0
        wb = benchmark_sector_weights.get(sec, 0.0)
        rows.append({
            "Sector": sec,
            "Portfolio Weight": wp,
            "Approx. Benchmark Weight": wb,
            "Weight Tilt": wp - wb,
            "Sector Return (Your Holdings)": sector_return,
            "Contribution to Portfolio Return": vals["contribution"]
        })

    attribution_df = pd.DataFrame(rows).set_index("Sector")
    total_portfolio_return = float(attribution_df["Contribution to Portfolio Return"].sum())

    return {
        "attribution": attribution_df,
        "total_portfolio_return": total_portfolio_return,
        "total_benchmark_return": bench_total_return,
        "excess_return": total_portfolio_return - bench_total_return
    }


# ================================
# WHAT-IF NEW HOLDING SIMULATOR
# Simulates adding a hypothetical position to your CURRENT holdings and
# recomputes portfolio-level return, volatility, beta, and Sharpe — without
# actually modifying your saved portfolio. Also reports the new ticker's
# correlation with your existing (pre-addition) portfolio, as a quick
# "diversification fit" signal.
# ================================
def simulate_add_holding(new_ticker, new_shares, period="1y", risk_free_rate=0.06):
    df = load_portfolio()
    new_row = pd.DataFrame({"Ticker": [new_ticker], "Shares": [new_shares]})
    combined = pd.concat([df, new_row], ignore_index=True)

    prices = get_historical_prices(combined, period=period)
    if prices.empty or new_ticker not in prices.columns:
        return None

    tickers = [t for t in combined["Ticker"].tolist() if t in prices.columns]
    combined = combined.set_index("Ticker").loc[tickers]
    prices = prices[tickers].ffill().dropna()
    if prices.empty or len(prices) < 30:
        return None

    returns = prices.pct_change().dropna()
    latest_prices = prices.iloc[-1]
    dollar_values = combined["Shares"] * latest_prices
    new_weights = dollar_values / dollar_values.sum()

    new_portfolio_returns = (returns[tickers] * new_weights[tickers]).sum(axis=1)

    bench = get_benchmark_history("^NSEI", period=period)
    bench_returns = bench.pct_change().dropna()
    aligned = pd.concat([new_portfolio_returns, bench_returns], axis=1).dropna()
    if aligned.empty:
        return None
    aligned.columns = ["p", "b"]

    new_annual_return = float(aligned["p"].mean() * 252)
    new_vol = float(aligned["p"].std() * np.sqrt(252))
    new_beta = float(aligned["p"].cov(aligned["b"]) / (aligned["b"].var() + 1e-9))
    new_sharpe = (new_annual_return - risk_free_rate) / (new_vol + 1e-9)

    current_metrics = get_metrics()

    corr_with_current = np.nan
    if new_ticker in returns.columns:
        new_ticker_returns = returns[new_ticker]
        current_port_returns = current_metrics.get("returns", pd.Series(dtype=float))
        corr_aligned = pd.concat([new_ticker_returns, current_port_returns], axis=1).dropna()
        if len(corr_aligned) > 2:
            corr_with_current = float(corr_aligned.iloc[:, 0].corr(corr_aligned.iloc[:, 1]))

    return {
        "new_ticker": new_ticker,
        "new_weights": new_weights.to_dict(),
        "new_annual_return": new_annual_return,
        "new_volatility": new_vol,
        "new_beta": new_beta,
        "new_sharpe": new_sharpe,
        "current_annual_return": current_metrics["annual_return"],
        "current_volatility": current_metrics["volatility"],
        "current_beta": current_metrics["beta"],
        "current_sharpe": current_metrics["sharpe"],
        "correlation_with_current_portfolio": corr_with_current
    }


# ================================
# REPORT EXPORT — Excel & PDF
# Generic builders: pass in whatever DataFrames/metrics you want included.
# Both return raw bytes, ready for st.download_button.
# ================================
def build_excel_report(sheets):
    """sheets: dict of {sheet_name: DataFrame}. Sheet names are truncated to
    Excel's 31-character limit."""
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for name, sheet_df in sheets.items():
            safe_name = str(name)[:31]
            sheet_df.to_excel(writer, sheet_name=safe_name)
    return buffer.getvalue()


def build_pdf_report(holdings_df, metrics, capture, total_value):
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("Portfolio Intelligence Report", styles["Title"]))
    elements.append(Paragraph(
        f"Generated: {datetime.datetime.now().strftime('%d %b %Y, %I:%M %p')}", styles["Normal"]
    ))
    elements.append(Spacer(1, 14))

    elements.append(Paragraph("Key Metrics", styles["Heading2"]))
    metrics_data = [
        ["Metric", "Value"],
        ["Portfolio Value", f"Rs. {total_value:,.0f}"],
        ["Annual Return", f"{metrics['annual_return']:.2%}"],
        ["Sharpe Ratio", f"{metrics['sharpe']:.2f}"],
        ["Beta", f"{metrics['beta']:.2f}"],
        ["Max Drawdown", f"{metrics['max_drawdown']:.2%}"],
        ["Volatility (Ann.)", f"{metrics['volatility']:.2%}"],
        ["Upside Capture", f"{capture['upside_capture']:.1f}%"],
        ["Downside Capture", f"{capture['downside_capture']:.1f}%"]
    ]
    metrics_table = Table(metrics_data, hAlign="LEFT")
    metrics_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#00C897")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
    ]))
    elements.append(metrics_table)
    elements.append(Spacer(1, 16))

    elements.append(Paragraph("Holdings", styles["Heading2"]))
    holdings_str = holdings_df.astype(str)
    holdings_data = [list(holdings_str.columns)] + holdings_str.values.tolist()
    holdings_table = Table(holdings_data, hAlign="LEFT")
    holdings_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#333333")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
    ]))
    elements.append(holdings_table)

    doc.build(elements)
    return buffer.getvalue()


# ================================
# BLACK-LITTERMAN MODEL
# Bayesian portfolio optimization (Black & Litterman, 1990). Standard
# mean-variance optimization is notoriously unstable — small changes in
# expected-return inputs produce wildly different "optimal" weights. Black-
# Litterman fixes this by starting from MARKET-IMPLIED equilibrium returns
# (derived mathematically from market weights + covariance, not guessed),
# then blends in your own views with an explicit confidence level.
#
#   Pi (equilibrium returns) = delta * Sigma * w_market
#   Combined returns = [(tau*Sigma)^-1 + P'Omega^-1 P]^-1
#                       * [(tau*Sigma)^-1 * Pi + P'Omega^-1 * Q]
#
# where P/Q encode your views ("I think TCS.NS will return 15%") and Omega
# encodes how confident you are in each view (via the He-Litterman scaling:
# Omega_kk = (P_k Sigma P_k') * tau / confidence_k).
# ================================
def black_litterman_returns(cov_matrix, market_weights, delta, views, tau=0.025):
    """
    cov_matrix: pd.DataFrame (N x N annualised covariance)
    market_weights: pd.Series of market/equilibrium weights, same tickers
    delta: risk-aversion coefficient (scalar)
    views: list of dicts: [{"ticker": "TCS.NS", "expected_return": 0.15, "confidence": 0.5}, ...]
            confidence in (0, 1]; higher = more confident = view weighted more heavily
    Returns: (combined_returns: pd.Series, prior_equilibrium_returns: pd.Series)
    """
    tickers = cov_matrix.index.tolist()
    n = len(tickers)
    Sigma = cov_matrix.values

    w_mkt = market_weights.reindex(tickers).fillna(0.0).values
    if w_mkt.sum() <= 0:
        w_mkt = np.repeat(1.0 / n, n)
    else:
        w_mkt = w_mkt / w_mkt.sum()

    Pi = delta * (Sigma @ w_mkt)
    prior_series = pd.Series(Pi, index=tickers)

    valid_views = [v for v in views if v.get("ticker") in tickers]
    if not valid_views:
        return prior_series.copy(), prior_series.copy()

    K = len(valid_views)
    P = np.zeros((K, n))
    Q = np.zeros(K)
    confidences = np.zeros(K)

    for i, v in enumerate(valid_views):
        j = tickers.index(v["ticker"])
        P[i, j] = 1.0
        Q[i] = v["expected_return"]
        confidences[i] = min(max(v.get("confidence", 0.5), 1e-4), 1 - 1e-6)

    tau_Sigma = tau * Sigma
    # Omega scales by (1-confidence)/confidence so that confidence -> 1 drives
    # uncertainty -> 0 (full trust in the view) and confidence -> 0 drives
    # uncertainty -> infinity (view essentially ignored) — both required
    # asymptotic properties of a sensible confidence mapping.
    omega_diag = np.array([
        (P[i:i + 1] @ tau_Sigma @ P[i:i + 1].T).item() * (1 - confidences[i]) / confidences[i]
        for i in range(K)
    ])
    omega_diag = np.maximum(omega_diag, 1e-12)
    Omega = np.diag(omega_diag)

    tau_Sigma_inv = np.linalg.inv(tau_Sigma)
    Omega_inv = np.linalg.inv(Omega)

    M_inv = tau_Sigma_inv + P.T @ Omega_inv @ P
    M = np.linalg.inv(M_inv)
    combined = M @ (tau_Sigma_inv @ Pi + P.T @ Omega_inv @ Q)

    return pd.Series(combined, index=tickers), prior_series


def run_black_litterman(include_tickers=None, period="1y", views=None, tau=0.025,
                         risk_free_rate=0.06, use_market_cap=True, allow_short=False):
    df = load_portfolio()
    if include_tickers:
        df = df[df["Ticker"].isin(include_tickers)]
        if len(df) < 2:
            return None

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
    cov_matrix = returns.cov() * 252

    market_cap_source = "current portfolio weights (fallback proxy)"
    market_weights = None

    if use_market_cap:
        caps = {}
        for t in tickers:
            try:
                info = yf.Ticker(t).info
                mc = info.get("marketCap")
                if mc and mc > 0:
                    caps[t] = mc
            except Exception:
                pass
        if len(caps) == len(tickers):
            market_weights = pd.Series(caps)
            market_cap_source = "live market capitalization"

    if market_weights is None:
        latest_prices = prices.iloc[-1]
        dollar_values = df["Shares"] * latest_prices
        market_weights = dollar_values / dollar_values.sum()

    bench = get_benchmark_history("^NSEI", period=period)
    bench_returns = bench.pct_change().dropna()
    bench_annual_return = float(bench_returns.mean() * 252)
    bench_annual_var = float(bench_returns.var() * 252)
    delta = (bench_annual_return - risk_free_rate) / (bench_annual_var + 1e-9)

    combined_returns, prior_returns = black_litterman_returns(
        cov_matrix, market_weights, delta, views or [], tau=tau
    )

    bounds = (-1.0, 1.0) if allow_short else (0.0, 1.0)
    bl_weights_arr = optimize_max_sharpe(combined_returns, cov_matrix, risk_free_rate, bounds)
    bl_ret, bl_vol = portfolio_performance(bl_weights_arr, combined_returns, cov_matrix)

    latest_prices = prices.iloc[-1]
    dollar_values = df["Shares"] * latest_prices
    current_weights_arr = (dollar_values / dollar_values.sum()).reindex(tickers).values
    current_ret, current_vol = portfolio_performance(current_weights_arr, combined_returns, cov_matrix)

    return {
        "tickers": tickers,
        "prior_returns": prior_returns.to_dict(),
        "combined_returns": combined_returns.to_dict(),
        "bl_weights": dict(zip(tickers, bl_weights_arr)),
        "bl_return": bl_ret,
        "bl_vol": bl_vol,
        "current_weights": dict(zip(tickers, current_weights_arr)),
        "current_return_under_bl_view": current_ret,
        "current_vol_under_bl_view": current_vol,
        "delta": delta,
        "market_weights": market_weights.reindex(tickers).to_dict(),
        "market_cap_source": market_cap_source
    }


# ================================
# MARKET REGIME DETECTION (Gaussian Mixture Model)
# Classifies NIFTY 50's history into distinct volatility/return regimes
# using unsupervised learning (no labels are pre-assigned — the model finds
# the clusters itself from daily return + rolling volatility features).
# Regimes are then labelled post-hoc by sorting on mean return so results
# are consistently interpretable ("Stressed/Bear" = lowest-return cluster).
# ================================
def detect_market_regimes(period="2y", n_regimes=3, rolling_vol_window=21):
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler

    bench = get_benchmark_history("^NSEI", period=period)
    bench_returns = bench.pct_change().dropna()
    rolling_vol = bench_returns.rolling(rolling_vol_window).std()

    features_df = pd.DataFrame({"return": bench_returns, "vol": rolling_vol}).dropna()
    if len(features_df) < n_regimes * 15:
        return None

    X = features_df.values
    X_scaled = StandardScaler().fit_transform(X)

    gmm = GaussianMixture(n_components=n_regimes, random_state=42, n_init=5)
    raw_labels = gmm.fit_predict(X_scaled)
    features_df["regime_raw"] = raw_labels

    cluster_stats = features_df.groupby("regime_raw").agg({"return": "mean", "vol": "mean"})
    cluster_stats_sorted = cluster_stats.sort_values("return")

    if n_regimes == 2:
        names = ["Stressed / Bear", "Calm / Bull"]
    elif n_regimes == 3:
        names = ["Stressed / Bear", "Neutral / Transitional", "Calm / Bull"]
    else:
        names = [f"Regime {i + 1} (by ascending return)" for i in range(n_regimes)]

    label_map = {idx: names[rank] for rank, idx in enumerate(cluster_stats_sorted.index)}
    features_df["Regime"] = features_df["regime_raw"].map(label_map)

    regime_summary = features_df.groupby("Regime").agg(
        Days=("return", "count"),
        Avg_Daily_Return=("return", "mean"),
        Avg_Volatility=("vol", "mean")
    )

    return {
        "regime_series": features_df["Regime"],
        "regime_summary": regime_summary,
        "n_regimes": n_regimes
    }


def get_portfolio_returns_for_period(period="2y"):
    df = load_portfolio()
    prices = get_historical_prices(df, period=period)
    if prices.empty:
        return None

    tickers = [t for t in df["Ticker"].tolist() if t in prices.columns]
    if not tickers:
        return None

    df = df.set_index("Ticker").loc[tickers]
    prices = prices[tickers].ffill().dropna()
    if prices.empty or len(prices) < 2:
        return None

    returns = prices.pct_change().dropna()
    latest_prices = prices.iloc[-1]
    dollar_values = df["Shares"] * latest_prices
    weights = dollar_values / dollar_values.sum()

    return (returns[tickers] * weights[tickers]).sum(axis=1)


def get_regime_conditional_metrics(period="2y", n_regimes=3, rolling_vol_window=21):
    regime_result = detect_market_regimes(period=period, n_regimes=n_regimes, rolling_vol_window=rolling_vol_window)
    if regime_result is None:
        return None

    portfolio_returns = get_portfolio_returns_for_period(period=period)
    if portfolio_returns is None:
        return None

    bench = get_benchmark_history("^NSEI", period=period)
    bench_returns = bench.pct_change().dropna()

    regime_series = regime_result["regime_series"]
    aligned = pd.concat([
        portfolio_returns.rename("p"), bench_returns.rename("b"), regime_series.rename("Regime")
    ], axis=1).dropna()

    if aligned.empty:
        return None

    rows = []
    for regime_name, group in aligned.groupby("Regime"):
        if len(group) < 5:
            continue
        p_ret, b_ret = group["p"], group["b"]
        rows.append({
            "Regime": regime_name,
            "Days Observed": int(len(group)),
            "Portfolio Ann. Return": float(p_ret.mean() * 252),
            "Portfolio Ann. Volatility": float(p_ret.std() * np.sqrt(252)),
            "Beta vs NIFTY": float(p_ret.cov(b_ret) / (b_ret.var() + 1e-9)),
            "Correlation vs NIFTY": float(p_ret.corr(b_ret))
        })

    if not rows:
        return None

    return {
        "regime_conditional_metrics": pd.DataFrame(rows).set_index("Regime"),
        "regime_series": regime_series,
        "regime_summary": regime_result["regime_summary"],
        "aligned_benchmark_returns": aligned["b"]
    }


# ================================
# RISK PARITY (EQUAL RISK CONTRIBUTION) PORTFOLIO
# Instead of maximizing Sharpe or minimizing volatility, solves for the
# weight vector where EVERY holding contributes equally to total portfolio
# risk (the methodology Bridgewater built its reputation on). Complements
# Max Sharpe and Min Volatility as a third portfolio-construction philosophy.
# ================================
def _risk_parity_objective(weights, Sigma):
    port_vol = np.sqrt(weights @ Sigma @ weights)
    marginal = (Sigma @ weights) / (port_vol + 1e-12)
    contributions = weights * marginal
    target = port_vol / len(weights)
    return float(np.sum((contributions - target) ** 2))


def optimize_risk_parity(cov_matrix):
    Sigma = cov_matrix.values if hasattr(cov_matrix, "values") else np.asarray(cov_matrix)
    n = Sigma.shape[0]

    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1},)
    bounds_list = tuple((1e-4, 1.0) for _ in range(n))
    init_guess = np.array(n * [1.0 / n])

    result = minimize(
        _risk_parity_objective, init_guess, args=(Sigma,),
        method="SLSQP", bounds=bounds_list, constraints=constraints,
        options={"ftol": 1e-14, "maxiter": 2000}
    )
    return result.x if result.success else init_guess


def get_risk_parity_portfolio(include_tickers=None, period="1y"):
    df = load_portfolio()
    if include_tickers:
        df = df[df["Ticker"].isin(include_tickers)]
        if len(df) < 2:
            return None

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
    cov_matrix = returns.cov() * 252
    mean_returns = returns.mean() * 252

    rp_weights_arr = optimize_risk_parity(cov_matrix)
    Sigma = cov_matrix.values
    port_vol = float(np.sqrt(rp_weights_arr @ Sigma @ rp_weights_arr))
    marginal = (Sigma @ rp_weights_arr) / (port_vol + 1e-12)
    component = rp_weights_arr * marginal
    pct_risk_contribution = component / (port_vol + 1e-12)

    rp_ret, rp_vol = portfolio_performance(rp_weights_arr, mean_returns, cov_matrix)

    latest_prices = prices.iloc[-1]
    dollar_values = df["Shares"] * latest_prices
    current_weights_arr = (dollar_values / dollar_values.sum()).values

    return {
        "tickers": tickers,
        "risk_parity_weights": dict(zip(tickers, rp_weights_arr)),
        "risk_parity_return": rp_ret,
        "risk_parity_vol": rp_vol,
        "pct_risk_contribution": dict(zip(tickers, pct_risk_contribution)),
        "current_weights": dict(zip(tickers, current_weights_arr))
    }


# ================================
# TARGET RETURN SOLVER
# "I want X% annual return — what weights get me there with the LEAST risk?"
# Reuses the same optimize_min_volatility(target_return=...) machinery
# already used internally to sweep the efficient frontier — no new
# optimization logic, just exposed as a standalone, user-facing query.
# ================================
def solve_target_return_portfolio(target_return, include_tickers=None, period="1y", allow_short=False):
    df = load_portfolio()
    if include_tickers:
        df = df[df["Ticker"].isin(include_tickers)]
        if len(df) < 2:
            return None

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

    min_ret, max_ret = float(mean_returns.min()), float(mean_returns.max())
    feasible = min_ret <= target_return <= max_ret
    clamped_target = min(max(target_return, min_ret), max_ret)

    bounds = (-1.0, 1.0) if allow_short else (0.0, 1.0)
    weights_arr = optimize_min_volatility(mean_returns, cov_matrix, target_return=clamped_target, bounds=bounds)
    achieved_ret, achieved_vol = portfolio_performance(weights_arr, mean_returns, cov_matrix)

    latest_prices = prices.iloc[-1]
    dollar_values = df["Shares"] * latest_prices
    current_weights_arr = (dollar_values / dollar_values.sum()).reindex(tickers).values
    current_ret, current_vol = portfolio_performance(current_weights_arr, mean_returns, cov_matrix)

    return {
        "tickers": tickers,
        "target_return": target_return,
        "feasible": feasible,
        "clamped_target": clamped_target,
        "achieved_return": achieved_ret,
        "achieved_vol": achieved_vol,
        "weights": dict(zip(tickers, weights_arr)),
        "current_weights": dict(zip(tickers, current_weights_arr)),
        "current_return": current_ret,
        "current_vol": current_vol,
        "min_achievable_return": min_ret,
        "max_achievable_return": max_ret
    }
