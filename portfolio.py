import pandas as pd
import yfinance as yf


# -------------------------
# PORTFOLIO DATA
# -------------------------
def load_portfolio():
    return pd.DataFrame({
        "Ticker": ["AAPL", "TSLA", "MSFT"],
        "Shares": [10, 5, 8],
        "AvgCost": [150, 220, 300]
    })


# -------------------------
# SAFE PRICE FETCH
# -------------------------
def get_current_prices(tickers):
    prices = {}

    for t in tickers:
        try:
            data = yf.download(t, period="5d", progress=False)

            if data is not None and not data.empty:
                prices[t] = float(data["Close"].iloc[-1])
            else:
                prices[t] = 0

        except Exception:
            prices[t] = 0

    return prices


# -------------------------
# HISTORY (SIMPLE MOCK)
# -------------------------
def get_portfolio_history():
    return pd.DataFrame({
        "Date": pd.date_range("2024-01-01", periods=6),
        "Value": [10000, 10200, 10100, 10500, 10700, 11000]
    })


# -------------------------
# METRICS (FIXED)
# -------------------------
def get_metrics(df, prices):
    df = df.copy()

    df["CurrentPrice"] = df["Ticker"].apply(lambda x: prices.get(x, 0))
    df["MarketValue"] = df["Shares"] * df["CurrentPrice"]
    df["CostValue"] = df["Shares"] * df["AvgCost"]
    df["PnL"] = df["MarketValue"] - df["CostValue"]

    return {
        "total_value": float(df["MarketValue"].sum()),
        "total_pnl": float(df["PnL"].sum()),
        "holdings": df
    }


# -------------------------
# SECTOR MAP
# -------------------------
def get_sector_map():
    return {
        "Tech": ["AAPL", "MSFT", "TSLA"]
    }


# -------------------------
# DRAWDOWN
# -------------------------
def get_drawdown_series(history_df):
    peak = history_df["Value"].cummax()
    drawdown = (history_df["Value"] - peak) / peak
    return drawdown


# -------------------------
# CAPTURE RATIOS (MOCK)
# -------------------------
def get_capture_ratios():
    return {
        "upside_capture": 1.12,
        "downside_capture": 0.88
    }