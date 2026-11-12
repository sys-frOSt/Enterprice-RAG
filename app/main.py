import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI

from app.api.routes import create_router
from app.config import get_settings
from app.embeddings.embedding_model import BGEEmbeddings
from app.ingestion.pipeline import IngestionPipeline
from app.llm.groq import create_groq
from app.rag.chain import RAGChain
from app.retrieval.retriever import Retriever
from app.utils.logger import configure_logging
from app.vectorstore.qdrant_store import QdrantStore

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level)
    services: dict[str, Any] = {}

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        settings.validate_startup()
        embeddings = BGEEmbeddings(settings.embedding_model)
        store = QdrantStore(settings, embeddings)
        retriever = Retriever(store, settings.rag_top_k, settings.rag_score_threshold)
        services["store"] = store
        services["pipeline"] = IngestionPipeline(settings, store)
        services["rag_chain"] = RAGChain(create_groq(settings), retriever)
        store.health()
        yield
        services.clear()

    app = FastAPI(title="Enterprise RAG API", version="0.1.0", lifespan=lifespan)
    app.include_router(
        create_router(
            pipeline=services,
            rag_chain=services,
            store=services,
            max_upload_size=settings.max_upload_size_mb * 1024 * 1024,
        )
    )
    return app


app = create_app()
