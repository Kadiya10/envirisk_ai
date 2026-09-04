from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    app_name: str = "EnviRisk AI"
    app_version: str = "1.0.0"

    gemini_api_key: str
    gemini_model: str = "gemini-2.5-flash"

    embedding_model: str = "BAAI/bge-m3"

    chroma_persist_directory: str = "./data/chroma"
    documents_directory: str = "./data/documents"

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()