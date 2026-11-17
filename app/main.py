
import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI

from app.api.routes import create_router
from app.config import get_settings
from app.embeddings.embedding_model import BGEEmbeddings
from app.graph.agents import create_agent
from app.ingestion.pipeline import IngestionPipeline
from app.rag.tools import create_rag_tools
from app.retrieval.retriever import Retriever
from app.utils.logger import configure_logging
from app.vectorstore.qdrant_store import QdrantStore

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level)

    services: dict[str, Any] = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        settings.validate_startup()

        try:
            # Initialize embeddings
            embeddings = BGEEmbeddings(settings.embedding_model)

            # Initialize Qdrant
            store = QdrantStore(settings, embeddings)
            store.health()

            # Initialize retriever
            retriever = Retriever(
                store,
                settings.rag_top_k,
                settings.rag_score_threshold,
            )

            # Initialize ingestion pipeline
            pipeline = IngestionPipeline(settings, store)

            services["store"] = store
            services["pipeline"] = pipeline
            services["retriever"] = retriever

            # Create RAG tools directly from Retriever
            rag_tools = create_rag_tools(retriever)

            # Initialize LangGraph agent
            agent = await create_agent(
                settings,
                rag_tools,
                settings.atlassian_email,
                settings.atlassian_api_token,
            )

            services["agent"] = agent

            app.state.settings = settings
            app.state.services = services

            logger.info("Enterprise RAG Agent initialized successfully")

            yield

        except Exception:
            logger.exception("Application initialization failed")
            raise

        finally:
            services.clear()
            logger.info("Enterprise RAG services shut down")

    app = FastAPI(
        title="Enterprise RAG API",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.include_router(
        create_router(
            pipeline=services,
            services=services,
            store=services,
            max_upload_size=settings.max_upload_size_mb * 1024 * 1024,
        )
    )

    return app


app = create_app()
