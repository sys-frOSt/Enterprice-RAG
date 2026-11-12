from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.routes import create_router
from app.retrieval.retriever import RetrievedChunk


class FakePipeline:
    def ingest(self, path: object) -> dict[str, object]:
        return {"document_id": "doc-1", "status": "ingested", "chunks": 2}


class FakeRAG:
    def invoke(self, question: str, **kwargs: object) -> tuple[str, list[RetrievedChunk]]:
        return "Answer", [RetrievedChunk("content", 0.9, "file.pdf", 1, {})]


class FakeStore:
    def health(self) -> bool:
        return True

    def list_collections(self) -> list[str]:
        return ["enterprise_rag"]

    def delete_document(self, document_id: str) -> None:
        assert document_id == "doc-1"


def client() -> TestClient:
    app = FastAPI()
    app.include_router(
        create_router(
            pipeline=FakePipeline(),
            rag_chain=FakeRAG(),
            store=FakeStore(),
            max_upload_size=1024,
        )
    )
    return TestClient(app)


def test_chat_returns_sources() -> None:
    response = client().post("/chat", json={"question": "What?"})

    assert response.status_code == 200
    assert response.json()["sources"][0]["source_name"] == "file.pdf"


def test_upload_rejects_non_pdf() -> None:
    response = client().post(
        "/documents/upload",
        files={"file": ("notes.txt", b"not a pdf", "text/plain")},
    )

    assert response.status_code == 415


def test_collections_and_delete() -> None:
    test_client = client()

    assert test_client.get("/collections").json() == ["enterprise_rag"]
    assert test_client.delete("/documents/doc-1").status_code == 204
