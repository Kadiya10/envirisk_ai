from backend.rag.retriever import retriever


def main():

    question = (
        "Quels sont les principaux impacts "
        "environnementaux du projet ?"
    )

    print("\n=== QUESTION ===")
    print(question)

    print("\n=== RECHERCHE RAG ===")

    results = retriever.search(
        query=question,
        top_k=5,
        min_similarity=0.0,
    )

    print(
        f"\nNombre de résultats : {len(results)}"
    )

    for i, result in enumerate(results, start=1):

        print("\n" + "=" * 70)
        print(f"RESULTAT {i}")

        print(
            f"Document : {result['document']}"
        )

        print(
            f"Page : {result['page']}"
        )

        print(
            f"Distance : {result['distance']}"
        )

        print(
            f"Pertinence : "
            f"{result['similarity_percent']} %"
        )

        print("\nPassage :")
        print(result["text"])


if __name__ == "__main__":
    main()