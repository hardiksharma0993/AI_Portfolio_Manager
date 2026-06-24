import pandas as pd
import yfinance as yf


# ----------------------------
# LOAD PORTFOLIO
# ----------------------------
def load_portfolio():
    return pd.DataFrame({
        "Ticker": ["AAPL", "TSLA", "MSFT"],
        "Shares": [10, 5, 8],
        "AvgCost": [150, 220, 300]
    })


# ----------------------------
# GET CURRENT PRICES (FIXED)
# ----------------------------
def get_current_prices(tickers):
    prices = {}

    for t in tickers:
        try:
            data = yf.Ticker(t).history(period="1d")

            if data is not None and not data.empty:
                prices[t] = float(data["Close"].iloc[-1])
            else:
                prices[t] = 0

        except Exception:
            prices[t] = 0

    return prices


# ----------------------------
# PORTFOLIO HISTORY (MOCK SAFE)
# ----------------------------
def get_portfolio_history():
    return pd.DataFrame({
        "Date": pd.date_range("2024-01-01", periods=5),
        "Value": [10000, 10500, 10200, 11000, 11500]
    })


# ----------------------------
# METRICS
# ----------------------------
def get_metrics(df, prices):
    df = df.copy()

    df["CurrentPrice"] = df["Ticker"].apply(lambda x: prices.get(x, 0))
    df["MarketValue"] = df["Shares"] * df["CurrentPrice"]
    df["CostValue"] = df["Shares"] * df["AvgCost"]
    df["PnL"] = df["MarketValue"] - df["CostValue"]

    return {
        "total_value": df["MarketValue"].sum(),
        "total_pnl": df["PnL"].sum(),
        "holdings": df
    }


# ----------------------------
# SECTOR MAP (SAFE STATIC)
# ----------------------------
def get_sector_map():
    return {
        "Tech": ["AAPL", "MSFT", "TSLA"]
    }


# ----------------------------
# DRAWDOWN (FIXED)
# ----------------------------
def get_drawdown_series(history_df):
    values = history_df["Value"]
    peak = values.cummax()
    drawdown = (values - peak) / peak
    return drawdown


# ----------------------------
# CAPTURE RATIOS (SAFE MOCK)
# ----------------------------
def get_capture_ratios():
    return {
        "upside_capture": 1.1,
        "downside_capture": 0.9
    }