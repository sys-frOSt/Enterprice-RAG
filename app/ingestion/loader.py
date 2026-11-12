import re
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_pdf(path: str | Path) -> list[Document]:
    pdf_path = Path(path)
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError("Only PDF documents are supported")
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    try:
        documents = PyPDFLoader(str(pdf_path), mode="page").load()
    except Exception as exc:
        raise RuntimeError(f"Unable to load PDF: {pdf_path.name}") from exc

    result: list[Document] = []
    for document in documents:
        text = clean_text(document.page_content)
        if not text:
            continue
        metadata = dict(document.metadata)
        metadata["source_name"] = pdf_path.name
        metadata["page"] = int(metadata.get("page", 0)) + 1
        result.append(Document(page_content=text, metadata=metadata))
    return result
