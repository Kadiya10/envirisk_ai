from backend.rag.loader import extract_pdf_text
from backend.rag.chunker import create_chunks
from backend.rag.embeddings import embedding_model
from backend.rag.vector_store import vector_store


PDF_PATH = "data/documents/EIES_test.pdf"


def main():
    print("\n=== 1. EXTRACTION DU PDF ===")

    pages = extract_pdf_text(PDF_PATH)

    print(f"Pages extraites : {len(pages)}")

    if not pages:
        print("Aucun texte trouvé dans le PDF.")
        return

    print("\n=== 2. CREATION DES CHUNKS ===")

    chunks = create_chunks(pages)

    print(f"Chunks créés : {len(chunks)}")

    if not chunks:
        print("Aucun chunk créé.")
        return

    print("\n=== 3. CREATION DES EMBEDDINGS ===")

    texts = [chunk["text"] for chunk in chunks]

    embeddings = embedding_model.encode(texts)

    print(f"Embeddings créés : {len(embeddings)}")

    print("\n=== 4. ENREGISTREMENT DANS CHROMADB ===")

    vector_store.add_chunks(
        chunks=chunks,
        embeddings=embeddings,
        document_name="EIES_test",
    )

    print("\n=== RESULTAT ===")

    print(
        "Nombre total de chunks dans ChromaDB :",
        vector_store.count()
    )


if __name__ == "__main__":
    main()