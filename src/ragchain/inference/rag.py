"""RAG pipeline orchestration using LangChain."""

from ragchain.config import config
from ragchain.inference.retrievers import get_ensemble_retriever
from ragchain.types import SearchResult


async def search(query: str, k: int | None = None) -> SearchResult:
    """
    Perform ensemble retrieval using BM25 and Chroma vector search.

    This is a simple search function (no LLM, no grading, no self-correction).
    For full RAG with adaptive retrieval and self-correction, use `graph.py`.

    Args:
        query: Search query text (e.g., 'Python machine learning')
        k: Number of results to return (default: from config.search_k)

    Returns:
        dict with 'query' and 'results' list of {content, metadata, distance}
    """
    if k is None:
        k = config.search_k

    # Get ensemble retriever (BM25 + Chroma with RRF)
    ensemble_retriever = get_ensemble_retriever(k)

    # Retrieve documents
    results = ensemble_retriever.invoke(query)

    # Limit to k results
    results = results[:k]

    # Format and return
    return {
        "query": query,
        "results": [{"content": r.page_content, "metadata": r.metadata, "distance": 0.0} for r in results],
    }


# ============================================================================
# NOTE: Simple Search vs Full RAG Pipeline
# ============================================================================
#
# This file provides a SIMPLE search function (no grading, no retry).
#
# For the FULL self-correcting RAG pipeline with:
#   - Intent classification
#   - Adaptive retrieval weights
#   - Document grading
#   - Query rewriting & retry
#
# See: src/ragchain/inference/graph.py (LangGraph implementation)
#
# ============================================================================
