import pandas as pd
import yfinance as yf

def load_portfolio():
    return pd.read_csv("data/portfolio.csv")


def get_total_value():
    df = load_portfolio()

    total = 0

    for _, row in df.iterrows():
        ticker = row["Ticker"]
        shares = row["Shares"]

        price = yf.Ticker(ticker).history(period="1d")["Close"].iloc[-1]
        total += price * shares

    return total


def get_risk_score():
    df = load_portfolio()

    # simple risk logic (you can improve later)
    num_stocks = len(df)

    if num_stocks <= 2:
        return "High risk (low diversification)"
    elif num_stocks <= 5:
        return "Moderate risk"
    else:
        return "Low risk (well diversified)"


def get_portfolio_summary():
    df = load_portfolio()

    summary = []

    for _, row in df.iterrows():
        summary.append(f"{row['Ticker']}:{row['Shares']}")

    return ", ".join(summary)