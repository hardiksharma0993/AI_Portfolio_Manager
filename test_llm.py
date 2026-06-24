"""
test_llm.py — Diagnostic test for Ollama + qwen3:8b connectivity.

Run directly:
    python test_llm.py

Tests:
  1. Raw Ollama connection
  2. ask_llm() wrapper (with think-tag stripping)
  3. Portfolio-style prompt end-to-end
"""

import ollama
import time
from llm import ask_llm, MODEL_NAME


def separator(title: str):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print('=' * 60)


# ================================
# TEST 1: RAW OLLAMA CALL
# ================================
separator("TEST 1: Raw Ollama Connection")

try:
    start = time.time()
    response = ollama.chat(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": "Say: hello world"}]
    )
    elapsed = time.time() - start

    print(f"✅ Ollama responded in {elapsed:.2f}s")
    print(f"\nFull response object:\n{response}")
    print(f"\nExtracted content:\n{response['message']['content']}")

except Exception as e:
    print(f"❌ Ollama connection FAILED: {e}")
    print("\nTroubleshooting:")
    print("  1. Is Ollama running? → `ollama serve`")
    print(f"  2. Is the model pulled? → `ollama pull {MODEL_NAME}`")
    print("  3. Check `ollama ps` for loaded models")


# ================================
# TEST 2: ask_llm() WRAPPER
# ================================
separator("TEST 2: ask_llm() Wrapper (think-tag stripped)")

try:
    start = time.time()
    result = ask_llm("What is a Sharpe Ratio? Answer in one sentence.")
    elapsed = time.time() - start

    print(f"✅ ask_llm() succeeded in {elapsed:.2f}s")
    print(f"\nResponse:\n{result}")

    if result.startswith("LLM ERROR"):
        print(f"\n⚠️  ask_llm returned an error string: {result}")

except Exception as e:
    print(f"❌ ask_llm() raised exception: {e}")


# ================================
# TEST 3: PORTFOLIO PROMPT
# ================================
separator("TEST 3: Portfolio-Style Prompt")

portfolio_prompt = """
You are a portfolio analyst. Briefly assess this portfolio in 3 bullet points:

Portfolio Value: ₹10,00,000
Annual Return: 14.5%
Sharpe Ratio: 0.92
Beta: 1.15
Max Drawdown: -18.3%
Upside Capture: 112%
Downside Capture: 95%

Provide 3 concise bullet points only.
"""

try:
    start = time.time()
    result = ask_llm(portfolio_prompt)
    elapsed = time.time() - start

    print(f"✅ Portfolio prompt responded in {elapsed:.2f}s")
    print(f"\nResponse:\n{result}")

except Exception as e:
    print(f"❌ Portfolio prompt failed: {e}")


separator("ALL TESTS COMPLETE")
