from collections.abc import Sequence
from typing import Any

from langchain_core.output_parsers import StrOutputParser

from app.rag.prompt import RAG_PROMPT
from app.retrieval.retriever import RetrievedChunk


def format_context(chunks: Sequence[RetrievedChunk]) -> str:
    return "\n\n".join(
        f"[Source: {chunk.source_name}, page {chunk.page}]\n{chunk.content}" for chunk in chunks
    )


class RAGChain:
    def __init__(self, llm: Any, retriever: Any) -> None:
        self.retriever = retriever
        self.chain = RAG_PROMPT | llm | StrOutputParser()

    def invoke(self, question: str, **retrieval_kwargs: Any) -> tuple[str, list[RetrievedChunk]]:
        chunks = self.retriever.retrieve(question, **retrieval_kwargs)
        if not chunks:
            return "The information is unavailable in the provided documents.", []
        answer = self.chain.invoke({"question": question, "context": format_context(chunks)})
        return answer, chunks
