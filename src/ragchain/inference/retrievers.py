"""
Ensemble Retriever - Combines keyword search (BM25) + semantic search (Chroma)

This module creates a hybrid retriever that:
1. Searches using BM25 (exact keyword matching)
2. Searches using Chroma (meaning/semantic matching)
3. Combines results using Reciprocal Rank Fusion (RRF)
4. Returns best documents that appear in both rankings
"""

import logging
import time
from collections import defaultdict
from functools import lru_cache

from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStoreRetriever

from ragchain.config import config
from ragchain.utils import timed

logger = logging.getLogger(__name__)


class EnsembleRetriever(BaseRetriever):
    """
    Hybrid retriever that combines keyword + semantic search

    How it works:
    - BM25 finds documents with exact keyword matches (fast)
    - Chroma finds documents with similar meanings (smart)
    - RRF combines both rankings to get the best results
    """

    # Required components
    bm25_retriever: BM25Retriever  # Keyword search engine
    chroma_retriever: VectorStoreRetriever  # Semantic search engine

    # Weights control how much we trust each retriever (must sum to ~1.0)
    bm25_weight: float = 0.4  # 40% trust in keyword matching
    chroma_weight: float = 0.6  # 60% trust in semantic matching

    def _parallel_retrieve(self, query: str) -> tuple[list[Document], list[Document]]:
        """Run BM25 and Chroma searches at the same time (parallel) for speed."""
        import concurrent.futures

        # Create 2 threads: one for BM25, one for Chroma
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            # Start both searches simultaneously
            bm25_future = executor.submit(self.bm25_retriever.invoke, query)
            chroma_future = executor.submit(self.chroma_retriever.invoke, query)

            # Wait for both to finish and return results
            return bm25_future.result(), chroma_future.result()

    def _compute_rrf_scores(self, bm25_docs: list[Document], chroma_docs: list[Document]) -> list[Document]:
        """
        Combine rankings using Reciprocal Rank Fusion (RRF)

        Simple idea: Documents that appear in BOTH rankings get higher scores.
        Formula: score = weight / (rank + 60)
        """

        rrf_k = 60  # Magic number that prevents first result from dominating
        doc_scores: dict[str, float] = defaultdict(float)  # Track scores per document
        doc_map: dict[str, Document] = {}  # Remember which document each text belongs to

        # Score BM25 results (keyword matches)
        for rank, doc in enumerate(bm25_docs):
            content = doc.page_content
            # Higher rank (0, 1, 2...) = lower score
            rrf_score = self.bm25_weight * (1.0 / (rank + rrf_k))
            doc_scores[content] += rrf_score  # Add to existing score if already found
            doc_map[content] = doc

        # Score Chroma results (semantic matches)
        for rank, doc in enumerate(chroma_docs):
            content = doc.page_content
            rrf_score = self.chroma_weight * (1.0 / (rank + rrf_k))
            doc_scores[content] += rrf_score  # Documents in BOTH get higher total score!
            doc_map[content] = doc

        # Sort by score (highest first) and return documents
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        return [doc_map[content] for content, _ in sorted_docs]

    def _get_relevant_documents(self, query: str) -> list[Document]:  # type: ignore[override]
        """
        Main retrieval method - the heart of ensemble search

        Steps:
        1. Search BM25 and Chroma in parallel (faster!)
        2. Combine results using RRF scoring
        3. Return top N documents
        """
        start = time.time()

        # Step 1: Search both retrievers at the same time
        bm25_docs, chroma_docs = self._parallel_retrieve(query)

        # Step 2: Combine rankings with RRF
        sorted_docs = self._compute_rrf_scores(bm25_docs, chroma_docs)

        # Step 3: Take only top N results (keeps context window manageable)
        top_docs = sorted_docs[: config.rrf_max_results]

        # Log performance stats
        elapsed = time.time() - start
        logger.debug(
            f"[EnsembleRetriever] Retrieved {len(bm25_docs)} BM25 + {len(chroma_docs)} semantic, "
            f"RRF returned {len(top_docs)}/{len(sorted_docs)} in {elapsed:.2f}s"
        )

        return top_docs


# ============================================================================
# Helper Functions (Simple building blocks)
# ============================================================================


def _load_documents_from_chroma(store: Chroma) -> list[Document]:
    """Load all documents from Chroma database (needed for BM25 indexing)."""

    # Get raw data from Chroma
    chroma_data = store.get()
    documents = chroma_data.get("documents", [])
    metadatas = chroma_data.get("metadatas", [])

    # Make sure every document has metadata (even if empty)
    if len(metadatas) < len(documents):
        metadatas.extend([{} for _ in range(len(documents) - len(metadatas))])

    # Convert to LangChain Document objects
    return [Document(page_content=doc, metadata=meta if meta else {}) for doc, meta in zip(documents, metadatas, strict=True)]


def _create_bm25_retriever(docs: list[Document], k: int) -> BM25Retriever:
    """Create BM25 keyword search retriever (searches for exact words)."""
    return BM25Retriever.from_documents(docs, k=k)


def _create_chroma_retriever(store: Chroma, k: int) -> VectorStoreRetriever:
    """Create Chroma semantic search retriever (searches by meaning)."""
    return store.as_retriever(search_kwargs={"k": k})


# ============================================================================
# Main Factory Function (Creates the ensemble retriever)
# ============================================================================


@lru_cache(maxsize=32)  # Cache results to avoid rebuilding BM25 index every time
@timed(logger, "get_ensemble_retriever")  # Track how long this takes
def get_ensemble_retriever(k: int, bm25_weight: float = 0.4, chroma_weight: float = 0.6) -> EnsembleRetriever:
    """
    Create an ensemble retriever (BM25 + Chroma)

    This is the main function you call to get a retriever!

    Args:
        k: How many results each retriever should return (e.g., 10)
        bm25_weight: Trust level for keyword search (default: 0.4 = 40%)
        chroma_weight: Trust level for semantic search (default: 0.6 = 60%)

    Returns:
        EnsembleRetriever ready to search

    Performance Note:
        - First call: Slow (~200ms, builds BM25 index)
        - Cached calls: Fast (~5ms, reuses index)
    """
    from ragchain.ingestion.storage import get_vector_store

    # Step 1: Get Chroma database
    store = get_vector_store()

    # Step 2: Load all documents (needed for BM25 indexing)
    docs = _load_documents_from_chroma(store)

    # Step 3: Create BM25 keyword retriever
    bm25_retriever = _create_bm25_retriever(docs, k)

    # Step 4: Create Chroma semantic retriever
    chroma_retriever = _create_chroma_retriever(store, k)

    # Step 5: Combine into ensemble retriever
    retriever = EnsembleRetriever(
        bm25_retriever=bm25_retriever,
        chroma_retriever=chroma_retriever,
        bm25_weight=bm25_weight,
        chroma_weight=chroma_weight,
    )

    logger.debug(f"[get_ensemble_retriever] Initialized with {len(docs)} documents, k={k}")
    return retriever
