"""
Document Grading - Check if retrieved docs can answer the query

Uses statistical scoring (keyword overlap + term frequency) instead of LLM.
Fast and cost-effective quality check for retrieval results.
"""

import logging
import re

from langchain_core.documents import Document

from ragchain.config import config
from ragchain.types import GradeSignal

logger = logging.getLogger(__name__)


# ============================================================================
# Helper Functions
# ============================================================================


def should_skip_grading() -> bool:
    """Check if grading is disabled in config."""
    return not config.enable_grading


def should_accept_docs(retrieved_docs: list[Document], retry_count: int) -> bool:
    """Auto-accept if no docs or already retried (prevents loops)."""
    return not retrieved_docs or retry_count > 0


def extract_keywords(text: str) -> set[str]:
    """
    Extract important words from text (removes common stop words)

    Example: "What is Python?" → {"python"}
    """

    # Common words to ignore (a, the, is, etc.)
    stop_words = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "he",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "that",
        "the",
        "to",
        "was",
        "will",
        "with",
        "what",
        "which",
        "who",
        "how",
        "when",
        "where",
        "why",
        "this",
        "these",
        "those",
        "can",
        "could",
        "should",
        "would",
        "do",
        "does",
        "did",
        "have",
        "had",
        "been",
        "being",
    }

    # Find all words (3+ letters), lowercase, remove stop words
    words = re.findall(r"\b[a-z]{3,}\b", text.lower())
    return {w for w in words if w not in stop_words}


# ============================================================================
# Main Grading Function
# ============================================================================


def grade_with_statistics(query: str, docs: list[Document]) -> GradeSignal:
    """
    Grade if documents can answer the query (using keyword matching)

    Scoring method:
    1. Extract keywords from query and docs
    2. Calculate overlap ratio (how many query words appear in doc)
    3. Calculate term frequency (how often keywords appear)
    4. Combined score = 70% overlap + 30% TF
    5. If top-3 docs have score ≥ 0.25 → YES, else NO

    What happens after grading:
    - YES → Documents are good, proceed to answer generation ✅
    - NO → Documents are poor quality:
        * First time (retry_count=0) → Query rewriter enhances query → Retry
        * Second time (retry_count=1) → Accept docs anyway (max retries)
        * System always returns an answer (may say "I don't have info...")

    Returns YES or NO (GradeSignal)
    """
    try:
        logger.debug(f"[grade_with_statistics] Grading {len(docs)} docs")

        # Step 1: Get keywords from query
        query_keywords = extract_keywords(query)

        if not query_keywords:
            logger.warning("[grade_with_statistics] No keywords, accepting docs")
            return GradeSignal.YES

        # Step 2: Score each document
        doc_scores = []
        for i, doc in enumerate(docs):
            doc_text = doc.page_content.lower()
            doc_keywords = extract_keywords(doc.page_content)

            # How many query keywords appear in doc? (Jaccard similarity)
            overlap = query_keywords & doc_keywords  # Intersection
            overlap_ratio = len(overlap) / len(query_keywords) if query_keywords else 0

            # How many times do keywords appear? (Term Frequency)
            tf_score = sum(doc_text.count(keyword) for keyword in query_keywords) / len(query_keywords)

            # Final score: 70% overlap + 30% term frequency (capped at 1.0)
            score = 0.7 * overlap_ratio + 0.3 * min(tf_score, 1.0)

            doc_scores.append((i, score, overlap_ratio))
            logger.debug(f"Doc {i}: score={score:.3f}")

        # Step 3: Sort by score (best first)
        doc_scores.sort(key=lambda x: x[1], reverse=True)

        # Step 4: Check if top-3 docs are good enough
        relevance_threshold = 0.25  # Minimum score to be considered "relevant"
        top_k = min(3, len(doc_scores))

        for rank, (doc_idx, score, _) in enumerate(doc_scores[:top_k], 1):
            if score >= relevance_threshold:
                reciprocal_rank = 1.0 / rank
                logger.info(f"Grade: YES (doc {doc_idx} rank {rank}, score={score:.3f}, MRR={reciprocal_rank:.3f})")
                return GradeSignal.YES

        # No good documents found
        if doc_scores:
            best_doc, best_score, _ = doc_scores[0]
            logger.info(f"Grade: NO (best doc {best_doc} score={best_score:.3f} < threshold {relevance_threshold})")
        else:
            logger.info("Grade: NO (no documents)")

        return GradeSignal.NO

    except Exception as e:
        logger.error(f"Grading error: {e}", exc_info=True)
        # On error, accept docs (don't block the pipeline)
        return GradeSignal.YES
