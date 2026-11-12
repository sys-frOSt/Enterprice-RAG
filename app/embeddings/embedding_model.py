from collections.abc import Sequence

from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings


class BGEEmbeddings(Embeddings):
    dimension = 1024

    def __init__(self, model_name: str = "BAAI/bge-m3") -> None:
        self.model_name = model_name
        self._embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
            query_encode_kwargs={"normalize_embeddings": True},
        )

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = self._embeddings.embed_documents(list(texts))
        self._validate(vectors)
        return vectors

    def embed_query(self, text: str) -> list[float]:
        vector = self._embeddings.embed_query(text)
        self._validate([vector])
        return vector

    def _validate(self, vectors: Sequence[Sequence[float]]) -> None:
        if vectors and any(len(vector) != self.dimension for vector in vectors):
            raise ValueError(f"Expected {self.dimension}-dimensional embeddings")
