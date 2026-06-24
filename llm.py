import re

MODEL_NAME = "qwen3:8b"


def _strip_think_tags(text):
    if not text:
        return ""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def ask_llm(prompt, system=None):
    try:
        import ollama
    except Exception:
        return "AI unavailable (Ollama not supported in deployment)."

    messages = []
    if system:
        messages.append({"role": "system", "content": system})

    messages.append({"role": "user", "content": prompt})

    try:
        res = ollama.chat(
            model=MODEL_NAME,
            messages=messages,
            options={"temperature": 0.3, "num_predict": 1024}
        )

        return _strip_think_tags(res["message"]["content"])

    except Exception as e:
        return f"AI error: {e}"