from dataclasses import dataclass
from typing import Any

from langchain_core.documents import Document
from qdrant_client import models

from app.vectorstore.qdrant_store import QdrantStore


@dataclass(frozen=True)
class RetrievedChunk:
    content: str
    score: float
    source_name: str
    page: int
    metadata: dict[str, Any]


class Retriever:
    def __init__(self, store: QdrantStore, top_k: int = 4, score_threshold: float | None = None):
        self.store = store
        self.top_k = top_k
        self.score_threshold = score_threshold

    def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        score_threshold: float | None = None,
        document_id: str | None = None,
    ) -> list[RetrievedChunk]:
        metadata_filter = None
        if document_id:
            metadata_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="metadata.document_id",
                        match=models.MatchValue(value=document_id),
                    )
                ]
            )
        results = self.store.search(
            query,
            top_k or self.top_k,
            self.score_threshold if score_threshold is None else score_threshold,
            metadata_filter,
        )
        return [self._to_chunk(document, score) for document, score in results]

    @staticmethod
    def _to_chunk(document: Document, score: float) -> RetrievedChunk:
        metadata = dict(document.metadata)
        return RetrievedChunk(
            content=document.page_content,
            score=float(score),
            source_name=str(metadata.get("source_name", metadata.get("source", "unknown"))),
            page=int(metadata.get("page", 0)),
            metadata=metadata,
        )
