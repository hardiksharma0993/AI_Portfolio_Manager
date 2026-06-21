from portfolio import (
    get_total_value,
    get_metrics,
    get_portfolio_history
)

from llm import ask_llm


print("\n📊 Portfolio AI Assistant")
print("Type exit to quit\n")


while True:

    q = input("You: ")

    if q.lower() == "exit":
        break

    try:

        if "value" in q.lower():

            v = get_total_value()

            prompt = f"Portfolio Value: ₹{v:,.0f}. Explain simply."

        elif "risk" in q.lower():

            m = get_metrics()

            prompt = f"""
Annual Return: {m['annual_return']:.2%}
Sharpe: {m['sharpe']:.2f}
Beta: {m['beta']:.2f}
Max Drawdown: {m['max_drawdown']:.2%}

Explain risk.
"""

        elif "portfolio" in q.lower():

            p = get_portfolio_history()

            prompt = f"Portfolio series:\n{p.tail(10)}"

        else:
            prompt = q

        print("\nAI:\n")
        print(ask_llm(prompt))
        print("\n")

    except Exception as e:
        print("ERROR:", e)