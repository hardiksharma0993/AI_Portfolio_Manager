import pandas as pd
import yfinance as yf


# ---------------------------
# PORTFOLIO (keep simple or replace with CSV later)
# ---------------------------
def load_portfolio():
    return {
        "AAPL": 5,
        "MSFT": 3,
        "GOOGL": 2,
        "TSLA": 1
    }


# ---------------------------
# SAFE PRICE FETCH (FIXED)
# ---------------------------
def get_current_prices(tickers):
    prices = {}

    for t in tickers:
        try:
            ticker = yf.Ticker(t)
            data = ticker.history(period="1d")

            if not data.empty:
                prices[t] = float(data["Close"].iloc[-1])
            else:
                prices[t] = None

        except Exception:
            prices[t] = None

    return prices


# ---------------------------
# PRICE HISTORY
# ---------------------------
def get_portfolio_history(tickers):
    data = yf.download(tickers, period="6mo")["Close"]
    return data


# ---------------------------
# METRICS (returns + volatility)
# ---------------------------
def get_metrics(tickers):
    data = yf.download(tickers, period="6mo")["Close"]
    returns = data.pct_change().dropna()

    return {
        "mean_return": returns.mean().to_dict(),
        "volatility": returns.std().to_dict()
    }


# ---------------------------
# SECTOR MAP
# ---------------------------
def get_sector_map():
    return {
        "AAPL": "Technology",
        "MSFT": "Technology",
        "GOOGL": "Communication Services",
        "TSLA": "Automotive"
    }


# ---------------------------
# DRAWDOWN SERIES (FIXED)
# ---------------------------
def get_drawdown_series(prices: pd.Series):
    rolling_max = prices.cummax()
    drawdown = (prices / rolling_max) - 1
    return drawdown


# ---------------------------
# CAPTURE RATIOS (safe version)
# ---------------------------
def get_capture_ratios(tickers):
    data = yf.download(tickers, period="6mo")["Close"]
    returns = data.pct_change().dropna()

    up = returns[returns > 0].mean()
    down = returns[returns < 0].mean()

    return {
        "up_capture": up.to_dict() if hasattr(up, "to_dict") else {},
        "down_capture": down.to_dict() if hasattr(down, "to_dict") else {}
    }