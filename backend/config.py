from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


# Dossier racine du projet
BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    # Application
    app_name: str = "EnviRisk AI"
    app_version: str = "1.0.0"

    # Google Gemini
    gemini_api_key: str
    gemini_model: str = "gemini-2.5-flash"

    # Modèle d'embeddings
    embedding_model: str = "BAAI/bge-m3"

    # Base vectorielle ChromaDB
    chroma_persist_directory: str = "./data/chroma"

    # Documents
    documents_directory: str = "./data/documents"

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()