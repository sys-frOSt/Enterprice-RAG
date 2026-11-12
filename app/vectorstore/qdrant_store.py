import logging
from typing import Any
from uuid import uuid4

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models

from app.config import Settings

logger = logging.getLogger(__name__)


class QdrantStore:
    def __init__(
        self,
        settings: Settings,
        embeddings: Any,
        client: QdrantClient | None = None,
    ) -> None:
        self.settings = settings
        self.client = client or QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            timeout=settings.request_timeout_seconds,
        )
        self.embeddings = embeddings
        self._ensure_collection()
        self.store = QdrantVectorStore(
            client=self.client,
            collection_name=settings.collection_name,
            embedding=embeddings,
            distance=models.Distance.COSINE,
        )

    def _ensure_collection(self) -> None:
        collections = self.client.get_collections().collections
        if any(item.name == self.settings.collection_name for item in collections):
            return
        self.client.create_collection(
            collection_name=self.settings.collection_name,
            vectors_config=models.VectorParams(
                size=self.settings.embedding_dimension,
                distance=models.Distance.COSINE,
            ),
        )

    def health(self) -> bool:
        self.client.get_collections()
        return True

    def list_collections(self) -> list[str]:
        return [item.name for item in self.client.get_collections().collections]

    def add_documents(self, documents: list[Document], document_id: str) -> list[str]:
        ids = [str(uuid4()) for _ in documents]
        for document in documents:
            document.metadata["document_id"] = document_id
        self.store.add_documents(documents, ids=ids)
        return ids

    def search(
        self,
        query: str,
        top_k: int,
        score_threshold: float | None = None,
        metadata_filter: models.Filter | None = None,
    ) -> list[tuple[Document, float]]:
        return self.store.similarity_search_with_score(
            query,
            k=top_k,
            score_threshold=score_threshold,
            filter=metadata_filter,
        )

    def delete_document(self, document_id: str) -> None:
        self.client.delete(
            collection_name=self.settings.collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="metadata.document_id",
                            match=models.MatchValue(value=document_id),
                        )
                    ]
                )
            ),
            wait=True,
        )

    def count_document(self, document_id: str) -> int:
        result = self.client.count(
            collection_name=self.settings.collection_name,
            count_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="metadata.document_id",
                        match=models.MatchValue(value=document_id),
                    )
                ]
            ),
            exact=True,
        )
        return result.count
