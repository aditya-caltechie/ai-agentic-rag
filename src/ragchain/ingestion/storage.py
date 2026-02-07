"""Storage utilities: embeddings, vector store, and document ingestion."""

import logging
import time
from pathlib import Path
from urllib.parse import urlparse

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ragchain.config import config
from ragchain.types import IngestResult

logger = logging.getLogger(__name__)


def get_embedder() -> OllamaEmbeddings:
    """Create Ollama embeddings with configured model (e.g., bge-m3)."""
    return OllamaEmbeddings(
        model=config.ollama_embed_model,
        base_url=config.ollama_base_url,
        num_ctx=config.ollama_embed_ctx,
    )


def get_vector_store() -> Chroma:
    """Return Chroma vector store (remote HTTP or local persistent).

    Uses CHROMA_SERVER_URL if set, otherwise CHROMA_PERSIST_DIRECTORY.
    """
    embedder = get_embedder()
    collection_name = "ragchain"

    # Remote Chroma server via HTTP
    if config.chroma_server_url:
        from chromadb import HttpClient

        parsed = urlparse(config.chroma_server_url)
        client = HttpClient(
            host=parsed.hostname or "localhost",
            port=parsed.port or 8000,
        )
        return Chroma(
            collection_name=collection_name,
            embedding_function=embedder,
            client=client,
        )

    # Local persistent Chroma
    persist_dir = Path(config.chroma_persist_directory)
    persist_dir.mkdir(parents=True, exist_ok=True)

    return Chroma(
        collection_name=collection_name,
        embedding_function=embedder,
        persist_directory=str(persist_dir),
    )


async def ingest_documents(docs: list[Document]) -> IngestResult:
    """Chunk, embed, and store documents in vector store.

    Pipeline: docs → split into chunks → embed → store in Chroma

    Args:
        docs: Documents to ingest

    Returns:
        Status dict with count and elapsed time
    """
    # Early return for empty input
    if not docs:
        return {
            "status": "ok",
            "count": 0,
            "message": "No documents to ingest",
            "elapsed_seconds": 0.0,
        }

    start_time = time.perf_counter()

    # Split documents into chunks with overlap for context preservation
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
    )
    chunks = splitter.split_documents(docs)

    # Store chunks in vector database
    store = get_vector_store()
    store.add_documents(chunks)

    # Clear cached retrievers to reflect new data
    from ragchain.inference.retrievers import get_ensemble_retriever

    get_ensemble_retriever.cache_clear()

    elapsed = time.perf_counter() - start_time

    return {
        "status": "ok",
        "count": len(chunks),
        "message": f"Ingested {len(chunks)} chunks in {elapsed:.2f}s",
        "elapsed_seconds": elapsed,
    }
