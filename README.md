# Enterprise RAG

A modular Retrieval-Augmented Generation service built with Python, LangChain,
the Groq API through `langchain-groq`, BAAI/BGE-M3 embeddings, and Qdrant.

## Status

The complete initial application is implemented: PDF ingestion, BGE-M3
embeddings, Qdrant storage and retrieval, Llama generation through Groq, a LangChain RAG
chain, FastAPI endpoints, structured configuration, and automated tests.

## Requirements

- Python 3.12, 3.13, or 3.14
- [uv](https://docs.astral.sh/uv/)
- A Groq API key
- A reachable Qdrant instance or Qdrant Cloud project

No Docker or Docker Compose setup is required by this project.

## Setup

From the repository root on Windows:

```powershell
uv sync
Copy-Item .env.example .env
```

Edit `.env` and set `GROQ_API_KEY`, `GROQ_MODEL`, and the Qdrant settings. Never
commit `.env` or place credentials in source code.

The dependency lockfile is generated and maintained by uv:

```powershell
uv lock
uv run pytest
uv run ruff check .
```

## Planned structure

```text
app/
  api/          FastAPI routes and request models
  embeddings/   BGE-M3 embedding adapter
  ingestion/   PDF loading and chunking
  llm/          ChatGroq integration
  rag/          Prompt and Runnable chain
  retrieval/    Semantic retrieval
  utils/        Logging and shared utilities
  vectorstore/  Qdrant adapter
tests/
data/           Local document data (ignored by git)
```

The application will use the Groq API for generation and will not use OpenAI
API keys or OpenAI embeddings.

## Run

```powershell
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Qdrant must be reachable at `QDRANT_URL`. The service creates the configured
collection if it does not exist and never drops it during ingestion.
