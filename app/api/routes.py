import tempfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from pydantic import BaseModel, Field

from app.rag.chain import RAGChain


def _service(container: object, name: str) -> object:
    if isinstance(container, dict):
        return container[name]
    return container


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=10_000)
    top_k: int | None = Field(default=None, ge=1, le=50)
    score_threshold: float | None = Field(default=None, ge=0, le=1)
    document_id: str | None = None


class SourceReference(BaseModel):
    source_name: str
    page: int
    score: float


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceReference]


class UploadResponse(BaseModel):
    document_id: str
    status: str
    chunks: int = 0


class HealthResponse(BaseModel):
    status: str
    qdrant: str


def create_router(
    *,
    pipeline: object,
    rag_chain: RAGChain,
    store: object,
    max_upload_size: int,
) -> APIRouter:
    router = APIRouter()

    @router.post(
        "/documents/upload",
        response_model=UploadResponse,
        status_code=status.HTTP_201_CREATED,
    )
    async def upload_document(
        file: UploadFile = File(...),  # noqa: B008
    ) -> UploadResponse:
        if file.content_type != "application/pdf":
            raise HTTPException(status_code=415, detail="Only PDF files are supported")
        data = await file.read(max_upload_size + 1)
        if len(data) > max_upload_size:
            raise HTTPException(status_code=413, detail="File exceeds configured size limit")
        if not data:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")
        suffix = Path(file.filename or "").suffix.lower()
        if suffix != ".pdf":
            raise HTTPException(status_code=415, detail="File name must have a .pdf extension")
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as handle:
            handle.write(data)
            path = Path(handle.name)
        try:
            result = _service(pipeline, "pipeline").ingest(path)
        except (ValueError, FileNotFoundError, RuntimeError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        finally:
            path.unlink(missing_ok=True)
        return UploadResponse(**result)

    @router.post("/chat", response_model=ChatResponse)
    async def chat(request: ChatRequest) -> ChatResponse:
        kwargs = {"top_k": request.top_k, "score_threshold": request.score_threshold}
        if request.document_id:
            kwargs["document_id"] = request.document_id
        answer, chunks = _service(rag_chain, "rag_chain").invoke(request.question, **kwargs)
        return ChatResponse(
            answer=answer,
            sources=[
                SourceReference(source_name=chunk.source_name, page=chunk.page, score=chunk.score)
                for chunk in chunks
            ],
        )

    @router.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        try:
            _service(store, "store").health()
        except Exception:
            return HealthResponse(status="degraded", qdrant="unavailable")
        return HealthResponse(status="ok", qdrant="ok")

    @router.get("/collections", response_model=list[str])
    async def collections() -> list[str]:
        try:
            return _service(store, "store").list_collections()
        except Exception as exc:
            raise HTTPException(status_code=503, detail="Qdrant unavailable") from exc

    @router.delete("/documents/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
    async def delete_document(document_id: str) -> None:
        try:
            _service(store, "store").delete_document(document_id)
        except Exception as exc:
            raise HTTPException(status_code=503, detail="Unable to delete document") from exc

    return router
