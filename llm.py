import ollama

MODEL_NAME = "qwen3:8b"


def ask_llm(prompt):

    try:
        res = ollama.chat(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}]
        )

        return res["message"]["content"]

    except Exception as e:
        return f"LLM ERROR: {str(e)}"