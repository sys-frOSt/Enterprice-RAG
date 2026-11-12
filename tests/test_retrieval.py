from langchain_core.documents import Document

from app.retrieval.retriever import Retriever


class FakeStore:
    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []

    def search(self, *args: object, **kwargs: object) -> list[tuple[Document, float]]:
        self.calls.append((*args, kwargs))
        return [
            (
                Document(
                    page_content="Relevant answer",
                    metadata={"source_name": "policy.pdf", "page": 4, "document_id": "doc-1"},
                ),
                0.91,
            )
        ]


def test_retrieval_preserves_source_and_page_and_filter() -> None:
    store = FakeStore()
    retriever = Retriever(store, top_k=4, score_threshold=0.7)

    chunks = retriever.retrieve("What is the policy?", document_id="doc-1")

    assert chunks[0].content == "Relevant answer"
    assert chunks[0].source_name == "policy.pdf"
    assert chunks[0].page == 4
    assert store.calls[0][0] == "What is the policy?"
    assert store.calls[0][1] == 4
