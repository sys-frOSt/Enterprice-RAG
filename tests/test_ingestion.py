from pathlib import Path

from langchain_core.documents import Document

from app.ingestion import loader
from app.ingestion.pipeline import IngestionPipeline
from app.ingestion.splitter import split_documents


def test_load_pdf_cleans_text_and_preserves_page_metadata(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "guide.pdf"
    source.write_bytes(b"%PDF")

    class FakeLoader:
        def __init__(self, path: str, mode: str) -> None:
            assert path == str(source)
            assert mode == "page"

        def load(self) -> list[Document]:
            return [Document(page_content="  Hello   world\n\n\n", metadata={"page": 1})]

    monkeypatch.setattr(loader, "PyPDFLoader", FakeLoader)
    documents = loader.load_pdf(source)

    assert documents[0].page_content == "Hello world"
    assert documents[0].metadata["source_name"] == "guide.pdf"
    assert documents[0].metadata["page"] == 2


def test_split_documents_adds_chunk_indexes() -> None:
    documents = [Document(page_content="one two three four five", metadata={"page": 1})]

    chunks = split_documents(documents, chunk_size=10, chunk_overlap=2)

    assert len(chunks) > 1
    assert [chunk.metadata["chunk_index"] for chunk in chunks] == list(range(len(chunks)))


def test_document_id_is_stable(tmp_path: Path) -> None:
    source = tmp_path / "guide.pdf"
    source.write_bytes(b"same content")

    first = IngestionPipeline.document_id(source)
    second = IngestionPipeline.document_id(source)

    assert first == second
    assert len(first) == 32
