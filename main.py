from portfolio import (
    get_total_value,
    get_risk_score,
    get_portfolio_summary
)

from llm import ask_llm

print("\n📊 AI Portfolio Agent (Tool Version)")
print("Type exit to quit\n")

while True:

    user_query = input("You: ")

    if user_query.lower() == "exit":
        break

    # 🧠 TOOL SELECTION LOGIC
    if "value" in user_query:
        result = get_total_value()

        prompt = f"""
User asked: {user_query}
Portfolio value is ₹{result}
Explain this simply.
"""

    elif "risk" in user_query:
        result = get_risk_score()

        prompt = f"""
User asked: {user_query}
Risk analysis: {result}
Explain this in simple terms.
"""

    elif "portfolio" in user_query:
        result = get_portfolio_summary()

        prompt = f"""
User asked: {user_query}
Holdings: {result}
Explain this simply.
"""

    else:
        prompt = f"""
You are a financial assistant.

Answer this:
{user_query}
"""

    response = ask_llm(prompt)

    print("\nAI:", response)