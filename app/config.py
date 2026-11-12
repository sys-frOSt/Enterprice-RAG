import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError, field_validator


class Settings(BaseModel):
    groq_api_key: str = Field(default="", repr=False)
    groq_model: str = "openai/gpt-oss-20b"
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = Field(default=None, repr=False)
    collection_name: str = "enterprise_rag"
    log_level: str = "INFO"
    rag_top_k: int = Field(default=4, ge=1, le=50)
    rag_score_threshold: float | None = Field(default=None, ge=0, le=1)
    embedding_model: str = "BAAI/bge-m3"
    embedding_dimension: int = Field(default=1024, gt=0)
    chunk_size: int = Field(default=1000, ge=100)
    chunk_overlap: int = Field(default=150, ge=0)
    request_timeout_seconds: float = Field(default=60, gt=0)
    max_upload_size_mb: int = Field(default=25, ge=1, le=500)

    @field_validator("groq_model", "collection_name", "embedding_model")
    @classmethod
    def require_non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value must not be empty")
        return value

    @field_validator("chunk_overlap")
    @classmethod
    def overlap_less_than_chunk_size(cls, value: int, info: object) -> int:
        chunk_size = info.data.get("chunk_size", 1000)
        if value >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return value

    def validate_startup(self) -> None:
        missing: list[str] = []
        if not self.groq_api_key:
            missing.append("GROQ_API_KEY")
        if missing:
            raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")
        if self.embedding_dimension != 1024:
            raise RuntimeError("EMBEDDING_DIMENSION must be 1024 for BAAI/bge-m3")


def _read_environment() -> dict[str, object]:
    load_dotenv(Path(".env"))
    return {
        "groq_api_key": os.getenv("GROQ_API_KEY", ""),
        "groq_model": os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
        "qdrant_url": os.getenv("QDRANT_URL", "http://localhost:6333"),
        "qdrant_api_key": os.getenv("QDRANT_API_KEY") or None,
        "collection_name": os.getenv("COLLECTION_NAME", "enterprise_rag"),
        "log_level": os.getenv("LOG_LEVEL", "INFO"),
        "rag_top_k": os.getenv("RAG_TOP_K", 4),
        "rag_score_threshold": os.getenv("RAG_SCORE_THRESHOLD") or None,
        "embedding_model": os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3"),
        "embedding_dimension": os.getenv("EMBEDDING_DIMENSION", 1024),
        "chunk_size": os.getenv("CHUNK_SIZE", 1000),
        "chunk_overlap": os.getenv("CHUNK_OVERLAP", 150),
        "request_timeout_seconds": os.getenv("REQUEST_TIMEOUT_SECONDS", 60),
        "max_upload_size_mb": os.getenv("MAX_UPLOAD_SIZE_MB", 25),
    }


@lru_cache
def get_settings() -> Settings:
    try:
        return Settings.model_validate(_read_environment())
    except ValidationError as exc:
        raise RuntimeError(f"Invalid application configuration: {exc}") from exc
