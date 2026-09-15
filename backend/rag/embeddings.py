from typing import List

from sentence_transformers import SentenceTransformer

from backend.config import settings


class EmbeddingModel:
    """
    Gestion du modèle d'embeddings utilisé par le RAG.
    """

    def __init__(self):
        print(
            f"Chargement du modèle d'embeddings : "
            f"{settings.embedding_model}"
        )

        self.model = SentenceTransformer(
            settings.embedding_model
        )

        print("Modèle d'embeddings chargé avec succès.")

    def encode(self, texts: List[str]) -> List[List[float]]:
        """
        Transforme une liste de textes en vecteurs.
        """

        if not texts:
            return []

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

        return embeddings.tolist()


embedding_model = EmbeddingModel()