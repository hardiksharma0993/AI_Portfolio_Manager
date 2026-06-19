import ollama

MODEL_NAME = "qwen3:8b"

def ask_llm(prompt):

    try:
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        print("DEBUG OLLAMA RESPONSE:", response)

        return response["message"]["content"]

    except Exception as e:
        return f"ERROR FROM OLLAMA: {str(e)}"


if __name__ == "__main__":
    response = ask_llm("Say hello in one sentence.")
    print(response)