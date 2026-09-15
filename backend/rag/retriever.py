from typing import List, Dict

from backend.rag.embeddings import embedding_model
from backend.rag.vector_store import vector_store


class Retriever:

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_similarity: float = 0.0
    ) -> List[Dict]:

        if not query or not query.strip():
            raise ValueError("La requête ne peut pas être vide.")

        if top_k <= 0:
            raise ValueError("top_k doit être supérieur à 0.")

        # Création de l'embedding de la question
        query_embedding = embedding_model.encode([query])[0]

        # Recherche dans ChromaDB
        results = vector_store.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=[
                "documents",
                "metadatas",
                "distances"
            ],
        )

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        retrieved_chunks = []

        for document, metadata, distance in zip(
            documents,
            metadatas,
            distances
        ):

            similarity = max(
                0.0,
                1.0 - float(distance)
            )

            similarity_percent = round(
                similarity * 100,
                2
            )

            if similarity < min_similarity:
                continue

            retrieved_chunks.append({
                "text": document,

                "document": metadata.get(
                    "document",
                    "Document inconnu"
                ),

                "document_type": metadata.get(
                    "document_type",
                    "AUTRE"
                ),

                "page": metadata.get(
                    "page",
                    "Inconnue"
                ),

                "distance": round(
                    float(distance),
                    4
                ),

                "similarity": round(
                    similarity,
                    4
                ),

                "similarity_percent": similarity_percent,
            })

        return retrieved_chunks


retriever = Retriever()