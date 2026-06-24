import re

MODEL_NAME = "qwen3:8b"


def _strip_think_tags(text: str) -> str:
    """
    Qwen3 models sometimes emit <think>...</think> reasoning blocks.
    Remove them before displaying output.
    """
    if not text:
        return ""

    return re.sub(
        r"<think>.*?</think>",
        "",
        text,
        flags=re.DOTALL
    ).strip()


def ask_llm(prompt: str, system: str = None) -> str:
    """
    Send a prompt to Ollama and return a clean response.

    This version safely handles:
    - Ollama package not installed
    - Ollama server not running
    - Model not downloaded
    - Any runtime errors

    The dashboard will continue working even if AI Insights fail.
    """

    try:
        import ollama
    except Exception as e:
        return (
            "AI Insights unavailable.\n\n"
            "Reason: Ollama package could not be loaded.\n"
            f"Details: {str(e)}"
        )

    messages = []

    if system:
        messages.append({
            "role": "system",
            "content": system
        })

    messages.append({
        "role": "user",
        "content": prompt
    })

    try:
        res = ollama.chat(
            model=MODEL_NAME,
            messages=messages,
            options={
                "temperature": 0.3,
                "num_predict": 1024
            }
        )

        raw = res.get("message", {}).get("content", "")

        return _strip_think_tags(raw)

    except Exception as e:
        return (
            "AI Insights unavailable.\n\n"
            "Possible causes:\n"
            "- Ollama server is not running\n"
            "- Model is not installed\n"
            "- Network access to Ollama is blocked\n\n"
            f"Error: {str(e)}"
        )