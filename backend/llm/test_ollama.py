from backend.llm.ollama import ollama_llm


def main():
    question = (
        "Explique en trois phrases ce qu'est "
        "un impact environnemental."
    )

    print("\n=== TEST QWEN3 AVEC OLLAMA ===")
    print("Question :", question)

    response = ollama_llm.generate(question)

    print("\n=== REPONSE ===")
    print(response)


if __name__ == "__main__":
    main()