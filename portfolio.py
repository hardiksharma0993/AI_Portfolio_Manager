import pandas as pd
import yfinance as yf


# -------------------------
# PORTFOLIO DATA
# -------------------------
def load_portfolio():
    return {
        "AAPL": 5,
        "MSFT": 3,
        "GOOGL": 2
    }


# -------------------------
# PRICES
# -------------------------
def get_current_prices(tickers):
    data = yf.download(tickers, period="1d")["Close"]
    return data.iloc[-1].to_dict()


def get_portfolio_history(tickers):
    data = yf.download(tickers, period="6mo")["Close"]
    return data


# -------------------------
# METRICS (basic version)
# -------------------------
def get_metrics(tickers):
    data = yf.download(tickers, period="6mo")["Close"]
    returns = data.pct_change().dropna()

    return {
        "mean_return": returns.mean().to_dict(),
        "volatility": returns.std().to_dict()
    }


# -------------------------
# SECTOR MAP (simple placeholder)
# -------------------------
def get_sector_map():
    return {
        "AAPL": "Technology",
        "MSFT": "Technology",
        "GOOGL": "Communication Services"
    }


# -------------------------
# DRAWDOWN (FIX FOR YOUR ERROR)
# -------------------------
def get_drawdown_series(prices: pd.Series):
    rolling_max = prices.cummax()
    drawdown = (prices - rolling_max) / rolling_max
    return drawdown


# -------------------------
# CAPTURE RATIOS (simple version)
# -------------------------
def get_capture_ratios(tickers):
    data = yf.download(tickers, period="6mo")["Close"]
    returns = data.pct_change().dropna()

    return {
        "up_capture": returns.mean().to_dict(),
        "down_capture": (returns[returns < 0].mean()).to_dict()
    }