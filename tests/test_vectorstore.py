from langchain_core.documents import Document
from qdrant_client import models

from app.config import Settings
from app.vectorstore import qdrant_store


class FakeClient:
    def __init__(self) -> None:
        self.created = False

    def get_collections(self) -> models.CollectionsResponse:
        return models.CollectionsResponse(collections=[])

    def create_collection(self, **kwargs: object) -> None:
        self.created = True


class FakeVectorStore:
    def __init__(self, **kwargs: object) -> None:
        self.documents: list[Document] = []

    def add_documents(self, documents: list[Document], ids: list[str]) -> list[str]:
        self.documents.extend(documents)
        return ids


def test_qdrant_store_creates_collection_and_inserts_documents(monkeypatch) -> None:
    client = FakeClient()
    monkeypatch.setattr(qdrant_store, "QdrantVectorStore", FakeVectorStore)
    settings = Settings(groq_api_key="test")
    store = qdrant_store.QdrantStore(settings, object(), client)

    ids = store.add_documents([Document(page_content="content", metadata={})], "doc-1")

    assert client.created
    assert len(ids) == 1
    assert store.store.documents[0].metadata["document_id"] == "doc-1"
