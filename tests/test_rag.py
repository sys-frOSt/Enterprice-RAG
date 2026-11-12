from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from app.rag.chain import RAGChain, format_context
from app.retrieval.retriever import RetrievedChunk


class FakeRetriever:
    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self.chunks = chunks

    def retrieve(self, query: str, **kwargs: object) -> list[RetrievedChunk]:
        assert query == "What?"
        return self.chunks


def fake_llm() -> RunnableLambda:
    def invoke(values: object) -> AIMessage:
        assert "[Source: source.pdf, page 2]" in str(values)
        return AIMessage(content="Grounded response")

    return RunnableLambda(invoke)


def test_rag_returns_answer_and_sources() -> None:
    chunk = RetrievedChunk("Facts", 0.8, "source.pdf", 2, {})
    rag = RAGChain(fake_llm(), FakeRetriever([chunk]))

    answer, sources = rag.invoke("What?")

    assert answer == "Grounded response"
    assert sources == [chunk]
    assert format_context(sources).startswith("[Source: source.pdf, page 2]")


def test_rag_returns_explicit_unavailable_answer() -> None:
    rag = RAGChain(fake_llm(), FakeRetriever([]))

    answer, sources = rag.invoke("What?")

    assert "unavailable" in answer
    assert sources == []
