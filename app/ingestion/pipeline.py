import hashlib
from pathlib import Path

from app.config import Settings
from app.ingestion.loader import load_pdf
from app.ingestion.splitter import split_documents
from app.vectorstore.qdrant_store import QdrantStore


class IngestionPipeline:
    def __init__(self, settings: Settings, store: QdrantStore):
        self.settings = settings
        self.store = store

    def ingest(self, path: str | Path, document_id: str | None = None) -> dict[str, object]:
        source = Path(path)
        document_id = document_id or self.document_id(source)
        if self.store.count_document(document_id):
            return {"document_id": document_id, "status": "already_ingested", "chunks": 0}
        pages = load_pdf(source)
        chunks = split_documents(pages, self.settings.chunk_size, self.settings.chunk_overlap)
        ids = self.store.add_documents(chunks, document_id)
        return {"document_id": document_id, "status": "ingested", "chunks": len(ids)}

    @staticmethod
    def document_id(path: str | Path) -> str:
        digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        return digest[:32]
