from langchain_core.tools import tool

from app.retrieval.retriever import Retriever


def create_rag_tools(retriever: Retriever):

    @tool("search_company_documents")
    def search_company_documents(question: str) -> str:
        """Search internal company documents using semantic retrieval."""

        chunks = retriever.retrieve(question)

        if not chunks:
            return "The information is unavailable in the provided documents."

        context = "\n\n".join(
            f"[Source: {chunk.source_name}, page {chunk.page}]\n"
            f"{chunk.content}"
            for chunk in chunks
        )

        return context

    return [search_company_documents]