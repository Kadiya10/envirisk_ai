from backend.llm.rag_llm import rag_llm


def main():
    print("\n" + "=" * 70)
    print(" ASSISTANT ENVIRONNEMENTAL - RAG + QWEN3")
    print("=" * 70)
    print("Écris une question sur ton document PDF.")
    print("Écris 'quitter' pour arrêter le programme.")

    while True:
        print("\n" + "-" * 70)

        question = input("Ta question : ").strip()

        if question.lower() in ["quitter", "q", "exit"]:
            print("\nProgramme terminé.")
            break

        if not question:
            print("Veuillez écrire une question.")
            continue

        print("\nRecherche dans les documents...")
        print("Génération de la réponse...\n")

        try:
            result = rag_llm.answer(
                question=question,
                top_k=5,
            )

            print("=== RÉPONSE ===")
            print(result["answer"])

            print("\n=== SOURCES UTILISÉES ===")

            if not result["sources"]:
                print("Aucune source trouvée.")
            else:
                for i, source in enumerate(
                    result["sources"],
                    start=1,
                ):
                    print(
                        f"{i}. Document : {source['document']} "
                        f"| Page : {source['page']} "
                        f"| Pertinence : "
                        f"{source['similarity_percent']} %"
                    )

        except Exception as error:
            print("\nUne erreur est survenue :")
            print(error)


if __name__ == "__main__":
    main()