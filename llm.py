import ollama
import re

MODEL_NAME = "qwen3:8b"


def _strip_think_tags(text: str) -> str:
    """
    Qwen3 models emit <think>...</think> reasoning blocks before the final answer.
    We strip those so only the clean response is shown in the dashboard.
    """
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def ask_llm(prompt: str, system: str = None) -> str:
    """
    Send a prompt to the local Ollama model and return the clean response.

    Args:
        prompt: User-facing prompt / question.
        system: Optional system message to set the model's persona/context.

    Returns:
        Clean response string (think tags stripped for Qwen3 models).
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    try:
        res = ollama.chat(
            model=MODEL_NAME,
            messages=messages,
            options={
                "temperature": 0.3,      # Lower = more factual, less hallucination
                "num_predict": 1024,     # Cap output length for dashboard use
            }
        )
        raw = res["message"]["content"]
        return _strip_think_tags(raw)
    except ollama.ResponseError as e:
        return f"LLM ERROR (model response): {str(e)}"
    except Exception as e:
        return f"LLM ERROR: {str(e)}"
