
import logging
import tempfile
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# --------------------------------------------------
# SERVICE HELPER
# --------------------------------------------------

def _service(container: Any, name: str) -> Any:
    """
    Extract a service from the application services dictionary.
    """
    if isinstance(container, dict):
        if name not in container:
            raise RuntimeError(
                f"Service '{name}' is not initialized"
            )
        return container[name]

    return container


# --------------------------------------------------
# REQUEST / RESPONSE MODELS
# --------------------------------------------------

class ChatRequest(BaseModel):
    question: str = Field(
        min_length=1,
        max_length=10_000,
        description="Question for the Enterprise AI Agent",
    )

    thread_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=128,
        description="Conversation ID for maintaining chat history",
    )


class SourceReference(BaseModel):
    source_name: str
    page: int
    score: float


class ChatResponse(BaseModel):
    answer: str
    thread_id: str
    sources: list[SourceReference] = Field(
        default_factory=list
    )


class UploadResponse(BaseModel):
    document_id: str
    status: str
    chunks: int = 0


class HealthResponse(BaseModel):
    status: str
    qdrant: str


# --------------------------------------------------
# ROUTER FACTORY
# --------------------------------------------------

def create_router(
    *,
    pipeline: object,
    services: object,
    store: object,
    max_upload_size: int,
) -> APIRouter:

    router = APIRouter()

    # --------------------------------------------------
    # UPLOAD DOCUMENT
    # --------------------------------------------------

    @router.post(
        "/documents/upload",
        response_model=UploadResponse,
        status_code=status.HTTP_201_CREATED,
    )
    async def upload_document(
        file: UploadFile = File(...),  # noqa: B008
    ) -> UploadResponse:

        if file.content_type != "application/pdf":
            raise HTTPException(
                status_code=415,
                detail="Only PDF files are supported",
            )

        suffix = Path(file.filename or "").suffix.lower()

        if suffix != ".pdf":
            raise HTTPException(
                status_code=415,
                detail="File name must have a .pdf extension",
            )

        data = await file.read(max_upload_size + 1)

        if len(data) > max_upload_size:
            raise HTTPException(
                status_code=413,
                detail="File exceeds configured size limit",
            )

        if not data:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty",
            )

        path = None

        try:
            with tempfile.NamedTemporaryFile(
                suffix=".pdf",
                delete=False,
            ) as handle:

                handle.write(data)
                path = Path(handle.name)

            ingestion_pipeline = _service(
                pipeline,
                "pipeline",
            )

            result = ingestion_pipeline.ingest(path)

            return UploadResponse(**result)

        except HTTPException:
            raise

        except (ValueError, FileNotFoundError, RuntimeError) as exc:
            logger.exception("Document ingestion failed")

            raise HTTPException(
                status_code=422,
                detail=str(exc),
            ) from exc

        except Exception as exc:
            logger.exception("Unexpected document upload failure")

            raise HTTPException(
                status_code=500,
                detail="Failed to upload document",
            ) from exc

        finally:
            if path is not None:
                path.unlink(missing_ok=True)

            await file.close()

    # --------------------------------------------------
    # CHAT WITH ENTERPRISE AI AGENT
    # --------------------------------------------------

    @router.post(
        "/chat",
        response_model=ChatResponse,
    )
    async def chat(
        request: ChatRequest,
    ) -> ChatResponse:

        thread_id = request.thread_id or str(uuid4())

        try:
            agent = _service(
                services,
                "agent",
            )

            result = await agent.ainvoke(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": request.question,
                        }
                    ]
                },
                config={
                    "configurable": {
                        "thread_id": thread_id,
                    }
                },
            )

            messages = result.get("messages", [])

            if not messages:
                answer = "No response generated."
            else:
                answer = messages[-1].content

            if not isinstance(answer, str):
                answer = str(answer)

            return ChatResponse(
                answer=answer,
                thread_id=thread_id,
                sources=[],
            )

        except Exception as exc:
            logger.exception("Chat request failed")

            raise HTTPException(
                status_code=500,
                detail="Failed to process chat request",
            ) from exc

    # --------------------------------------------------
    # HEALTH CHECK
    # --------------------------------------------------

    @router.get(
        "/health",
        response_model=HealthResponse,
    )
    async def health() -> HealthResponse:

        try:
            qdrant_store = _service(
                store,
                "store",
            )

            qdrant_store.health()

            return HealthResponse(
                status="ok",
                qdrant="ok",
            )

        except Exception:
            logger.exception("Qdrant health check failed")

            return HealthResponse(
                status="degraded",
                qdrant="unavailable",
            )

    # --------------------------------------------------
    # LIST COLLECTIONS
    # --------------------------------------------------

    @router.get(
        "/collections",
        response_model=list[str],
    )
    async def collections() -> list[str]:

        try:
            qdrant_store = _service(
                store,
                "store",
            )

            return qdrant_store.list_collections()

        except Exception as exc:
            logger.exception("Failed to retrieve Qdrant collections")

            raise HTTPException(
                status_code=503,
                detail="Qdrant unavailable",
            ) from exc

    # --------------------------------------------------
    # DELETE DOCUMENT
    # --------------------------------------------------

    @router.delete(
        "/documents/{document_id}",
        status_code=status.HTTP_204_NO_CONTENT,
    )
    async def delete_document(
        document_id: str,
    ) -> None:

        try:
            qdrant_store = _service(
                store,
                "store",
            )

            qdrant_store.delete_document(document_id)

        except Exception as exc:
            logger.exception(
                "Failed to delete document: %s",
                document_id,
            )

            raise HTTPException(
                status_code=503,
                detail="Unable to delete document",
            ) from exc

    return router
