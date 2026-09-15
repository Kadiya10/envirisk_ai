from typing import List, Dict

import chromadb

from backend.config import settings


class VectorStore:

    def __init__(self):

        self.client = chromadb.PersistentClient(
            path=settings.chroma_persist_directory
        )

        self.collection = (
            self.client.get_or_create_collection(
                name="environmental_documents"
            )
        )

        print(
            "ChromaDB initialisée avec succès."
        )

    def add_chunks(
        self,
        chunks: List[Dict],
        embeddings: List[List[float]],
        document_name: str,
        document_type: str
    ):

        if not chunks:
            return

        if len(chunks) != len(embeddings):
            raise ValueError(
                "Le nombre de chunks doit être "
                "égal au nombre d'embeddings."
            )

        ids = []
        documents = []
        metadatas = []

        for chunk in chunks:

            ids.append(
                f"{document_name}_{chunk['id']}"
            )

            documents.append(
                chunk["text"]
            )

            metadatas.append({
                "document": document_name,
                "document_type": document_type,
                "page": chunk["page"],
            })

        self.collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        print(
            f"{len(chunks)} chunks "
            f"enregistrés dans ChromaDB."
        )

    def count(self):

        return self.collection.count()


vector_store = VectorStore()