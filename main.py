from portfolio import (
    get_total_value,
    get_metrics,
    get_portfolio_history,
    get_capture_ratios,
    get_drawdown_series,
    run_backtest,
    compute_series_stats
)

from llm import ask_llm


HELP_TEXT = """
Available commands:
  value       — Current portfolio value
  risk        — Risk metrics (Sharpe, Beta, Drawdown, Volatility)
  capture     — Upside / Downside capture ratios vs NIFTY 50
  drawdown    — Maximum drawdown and current drawdown from peak
  portfolio   — Last 10 days of portfolio value series
  backtest    — Run rebalanced vs buy-and-hold backtest (1Y, monthly rebalance)
  help        — Show this help
  exit        — Quit

Or type any free-form question and the AI will answer it.
"""


def handle_value():
    v = get_total_value()
    prompt = f"The portfolio's current total value is ₹{v:,.0f}. Briefly explain what this number represents and any key context about portfolio sizing."
    return ask_llm(prompt)


def handle_risk():
    m = get_metrics()
    prompt = f"""
Portfolio Risk Metrics (vs NIFTY 50, 1-year):
- Annual Return:  {m['annual_return']:.2%}
- Volatility:     {m['volatility']:.2%}
- Sharpe Ratio:   {m['sharpe']:.2f}
- Beta:           {m['beta']:.2f}
- Max Drawdown:   {m['max_drawdown']:.2%}

Interpret each metric clearly for an informed but non-technical investor. Flag any concerning values.
"""
    return ask_llm(prompt)


def handle_capture():
    m = get_metrics()
    c = get_capture_ratios(
        portfolio_returns=m["returns"],
        benchmark_returns=m["benchmark_returns"]
    )
    prompt = f"""
Capture Ratios vs NIFTY 50 (1-year daily returns):
- Upside Capture:   {c['upside_capture']:.1f}%
- Downside Capture: {c['downside_capture']:.1f}%

Explain what these ratios mean, whether they indicate a good or bad portfolio vs benchmark, and what the ideal range is.
"""
    return ask_llm(prompt)


def handle_drawdown():
    m = get_metrics()
    p = get_portfolio_history()
    dd_series = get_drawdown_series(p)
    current_dd = float(dd_series.iloc[-1])

    prompt = f"""
Drawdown Analysis:
- Maximum Drawdown (1Y): {m['max_drawdown']:.2%}
- Current Drawdown from Peak: {current_dd:.2%}

Explain what drawdown means, whether this level is typical for an Indian equity portfolio, and what recovery might look like.
"""
    return ask_llm(prompt)


def handle_portfolio():
    p = get_portfolio_history()
    prompt = f"""
Portfolio daily value series (last 10 trading days):
{p.tail(10).to_string()}

Comment on the trend, any notable moves, and overall direction.
"""
    return ask_llm(prompt)


def handle_backtest():
    result = run_backtest(rebalance_freq="Monthly", transaction_cost_bps=10.0, period="1y")

    if result is None:
        return "Not enough overlapping historical data across holdings to run a backtest."

    stats_rebal = compute_series_stats(result["rebalanced"])
    stats_bh = compute_series_stats(result["buy_and_hold"])

    prompt = f"""
Backtest Results (1Y, monthly rebalance, 10bps transaction cost):

Rebalanced strategy:
- CAGR: {stats_rebal['cagr']:.2%}
- Volatility: {stats_rebal['volatility']:.2%}
- Sharpe: {stats_rebal['sharpe']:.2f}
- Max Drawdown: {stats_rebal['max_drawdown']:.2%}

Buy & Hold strategy:
- CAGR: {stats_bh['cagr']:.2%}
- Volatility: {stats_bh['volatility']:.2%}
- Sharpe: {stats_bh['sharpe']:.2f}
- Max Drawdown: {stats_bh['max_drawdown']:.2%}

Total transaction cost drag from rebalancing: {result['total_transaction_cost']:.4%}

Compare the two strategies and explain which performed better on a risk-adjusted basis and why.
"""
    return ask_llm(prompt)


def main():
    print("\n📊 Portfolio AI Assistant")
    print(HELP_TEXT)

    while True:
        try:
            q = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye.")
            break

        if not q:
            continue

        if q.lower() == "exit":
            print("Goodbye.")
            break

        if q.lower() == "help":
            print(HELP_TEXT)
            continue

        print("\nAI:\n")

        try:
            lower_q = q.lower()

            if lower_q == "value":
                print(handle_value())

            elif lower_q == "risk":
                print(handle_risk())

            elif lower_q == "capture":
                print(handle_capture())

            elif lower_q == "drawdown":
                print(handle_drawdown())

            elif lower_q == "portfolio":
                print(handle_portfolio())

            elif lower_q == "backtest":
                print(handle_backtest())

            else:
                # Free-form: pass raw question to LLM with portfolio context
                m = get_metrics()
                c = get_capture_ratios(
                    portfolio_returns=m["returns"],
                    benchmark_returns=m["benchmark_returns"]
                )
                system_context = f"""
You are an expert Indian equity portfolio analyst.

Current portfolio context:
- Annual Return: {m['annual_return']:.2%}
- Sharpe: {m['sharpe']:.2f}
- Beta: {m['beta']:.2f}
- Max Drawdown: {m['max_drawdown']:.2%}
- Upside Capture: {c['upside_capture']:.1f}%
- Downside Capture: {c['downside_capture']:.1f}%

Answer the user's question with this context in mind. Be concise and analytical.
"""
                print(ask_llm(q, system=system_context))

        except Exception as e:
            print(f"ERROR: {e}")

        print()


if __name__ == "__main__":
    main()
