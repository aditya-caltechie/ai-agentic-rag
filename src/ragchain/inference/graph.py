"""
LangGraph RAG Pipeline - Self-Correcting Adaptive Retrieval

Flow:
    1. INTENT_ROUTER → Classify query (FACT/CONCEPT/COMPARISON)
    2. ADAPTIVE_RETRIEVER → Get documents with intent-specific weights
    3. RETRIEVAL_GRADER → Check if documents answer the query
    4. If grade fails → QUERY_REWRITER → Retry (max 1 time)
    5. Return documents for answer generation

Usage:
    from ragchain.inference.graph import rag_graph

    state = {"query": "What is Python?"}
    result = rag_graph.invoke(state)
    docs = result["retrieved_docs"]
"""

import logging

from langgraph.graph import END, StateGraph

from ragchain.config import config
from ragchain.inference.grader import grade_with_statistics, should_accept_docs, should_skip_grading
from ragchain.inference.retrievers import get_ensemble_retriever
from ragchain.inference.router import intent_router
from ragchain.prompts import QUERY_REWRITER_PROMPT
from ragchain.types import GradeSignal, Intent, IntentRoutingState, Node
from ragchain.utils import get_llm, timed

logger = logging.getLogger(__name__)

__all__ = ["rag_graph"]


# ============================================================================
# Node Functions (Steps in the RAG Pipeline)
# ============================================================================


@timed(logger, "adaptive_retriever")
def adaptive_retriever(state: IntentRoutingState) -> IntentRoutingState:
    """
    Retrieve documents using intent-specific weights

    Weights by intent:
    - FACT: 80% keyword (BM25), 20% semantic (Chroma)
    - CONCEPT: 40% keyword, 60% semantic
    - COMPARISON: 50% keyword, 50% semantic
    """

    # Use rewritten query if available, otherwise use original
    query = state.get("rewritten_query") or state["query"]

    # Define retrieval weights for each intent type
    weights = {
        Intent.FACT: (0.8, 0.2),  # Keyword-heavy for lists
        Intent.CONCEPT: (0.4, 0.6),  # Semantic-heavy for explanations
        Intent.COMPARISON: (0.5, 0.5),  # Balanced for comparisons
    }
    bm25_weight, chroma_weight = weights.get(state["intent"], (0.5, 0.5))

    try:
        # Get ensemble retriever with intent-specific weights
        retriever = get_ensemble_retriever(config.graph_k, bm25_weight=bm25_weight, chroma_weight=chroma_weight)
        docs = retriever.invoke(query)
        logger.debug(f"[adaptive_retriever] Retrieved {len(docs)} docs for {state['intent'].value}")
    except Exception as e:
        logger.error(f"[adaptive_retriever] Error: {e}")
        docs = []

    return {**state, "retrieved_docs": docs}


@timed(logger, "retrieval_grader")
def retrieval_grader(state: IntentRoutingState) -> IntentRoutingState:
    """
    Grade if retrieved documents can answer the query

    Returns YES if docs are good, NO if they need improvement.
    Auto-accepts if: grading disabled, no docs, or already retried.
    """

    # Fast-path: Skip grading if disabled
    if should_skip_grading():
        return {**state, "retrieval_grade": GradeSignal.YES}

    # Auto-accept if no docs or already retried (prevent loops)
    if should_accept_docs(state["retrieved_docs"], state.get("retry_count", 0)):
        return {**state, "retrieval_grade": GradeSignal.YES}

    # Grade with statistical scoring
    grade_value = grade_with_statistics(state["query"], state["retrieved_docs"])
    logger.debug(f"[retrieval_grader] Grade: {grade_value} ({len(state['retrieved_docs'])} docs)")

    return {**state, "retrieval_grade": grade_value}


@timed(logger, "query_rewriter")
def query_rewriter(state: IntentRoutingState) -> IntentRoutingState:
    """
    Rewrite query for better retrieval (called when initial retrieval fails)

    Example: "OOP" → "What is object-oriented programming?"
    """

    llm = get_llm(purpose="rewriting")

    # Always rewrite from original query (not previous rewrite)
    original = state["original_query"]
    prompt = QUERY_REWRITER_PROMPT.format(query=original)
    rewritten = llm.invoke(prompt).strip()

    logger.debug(f"[query_rewriter] Rewrite #{state.get('retry_count', 0) + 1}")

    return {**state, "rewritten_query": rewritten, "retry_count": state.get("retry_count", 0) + 1}


# ============================================================================
# LangGraph Workflow Construction (5 Simple Steps)
# ============================================================================

# STEP 1: Initialize state graph (what data flows through the pipeline)
workflow = StateGraph(IntentRoutingState)

# STEP 2: Add nodes (the 4 processing steps)
workflow.add_node(Node.INTENT_ROUTER, intent_router)  # Classify query type
workflow.add_node(Node.ADAPTIVE_RETRIEVER, adaptive_retriever)  # Get documents
workflow.add_node(Node.RETRIEVAL_GRADER, retrieval_grader)  # Check quality
workflow.add_node(Node.QUERY_REWRITER, query_rewriter)  # Fix bad queries

# STEP 3: Set starting point
workflow.set_entry_point(Node.INTENT_ROUTER)  # All queries start here

# STEP 4: Connect nodes with edges (define the flow)
# Main path: route → retrieve → grade
workflow.add_edge(Node.INTENT_ROUTER, Node.ADAPTIVE_RETRIEVER)
workflow.add_edge(Node.ADAPTIVE_RETRIEVER, Node.RETRIEVAL_GRADER)


def should_rewrite(state: IntentRoutingState) -> str:
    """
    Decide what to do after grading

    - If grade = YES → END (success!)
    - If grade = NO and already retried → END (give up)
    - If grade = NO and first try → QUERY_REWRITER (try again)
    """
    if state["retrieval_grade"] == GradeSignal.YES:
        return END  # Success

    if state.get("retry_count", 0) >= 1:
        return END  # Already retried, give up

    return Node.QUERY_REWRITER  # First failure, retry


# Add conditional routing: grade → END or rewrite
workflow.add_conditional_edges(
    Node.RETRIEVAL_GRADER,  # After grading...
    should_rewrite,  # This function decides where to go
    {END: END, Node.QUERY_REWRITER: Node.QUERY_REWRITER},
)

# Add retry loop: rewriter → retriever (try again)
workflow.add_edge(Node.QUERY_REWRITER, Node.ADAPTIVE_RETRIEVER)

# STEP 5: Compile into executable graph
rag_graph = workflow.compile()


# ============================================================================
# Graph Flow Summary
# ============================================================================
#
# SUCCESS PATH (no retry):
#   START → INTENT_ROUTER → ADAPTIVE_RETRIEVER → RETRIEVAL_GRADER → END
#
# RETRY PATH (self-correction):
#   START → INTENT_ROUTER → ADAPTIVE_RETRIEVER → RETRIEVAL_GRADER
#         → QUERY_REWRITER → ADAPTIVE_RETRIEVER → RETRIEVAL_GRADER → END
#
# ============================================================================


# ============================================================================
# WHAT HAPPENS WHEN DOCUMENTS ARE GRADED "NO" (Can't Answer Query)
# ============================================================================
#
# ┌─────────────────────────────────────────────────────────────────────┐
# │ ATTEMPT 1 (Initial Retrieval)                                       │
# ├─────────────────────────────────────────────────────────────────────┤
# │ 1. Intent Router → Classify query (FACT/CONCEPT/COMPARISON)         │
# │ 2. Adaptive Retriever → Get documents with intent-specific weights  │
# │ 3. Grader → Check relevance (keyword overlap + TF scoring)          │
# │    └─ Score each doc: 0.7 × overlap + 0.3 × term_frequency          │
# │    └─ Check if top-3 docs have score ≥ 0.25                         │
# │                                                                     │
# │ IF Grade = YES (score ≥ 0.25):                                      │
# │    → Generate answer ✅                                             │
# │                                                                     │
# │ IF Grade = NO (score < 0.25) AND retry_count = 0:                   │
# │    → Continue to QUERY REWRITER ↓                                   │
# └─────────────────────────────────────────────────────────────────────┘
#         ↓
# ┌─────────────────────────────────────────────────────────────────────┐
# │ QUERY REWRITER (Self-Correction)                                    │
# ├─────────────────────────────────────────────────────────────────────┤
# │ - LLM enhances query with better keywords                           │
# │ - Example: "OOP" → "object-oriented programming principles"         │
# │ - retry_count += 1                                                  │
# └─────────────────────────────────────────────────────────────────────┘
#         ↓
# ┌─────────────────────────────────────────────────────────────────────┐
# │ ATTEMPT 2 (Retry with Better Query)                                 │
# ├─────────────────────────────────────────────────────────────────────┤
# │ 1. Adaptive Retriever → Get docs with rewritten query               │
# │ 2. Grader → Check relevance again                                   │
# │                                                                     │
# │ IF Grade = YES:                                                     │
# │    → Generate answer ✅ (improved docs!)                            │
# │                                                                     │
# │ IF Grade = NO AND retry_count = 1:                                  │
# │    → Accept current docs anyway (max retries reached)               │
# │    → Generate answer ⚠️ (may say "I don't have info...")            │
# └─────────────────────────────────────────────────────────────────────┘
#
# KEY POINTS:
# ────────────
# 1. Max 1 retry (prevents infinite loops)
# 2. System always returns an answer (never crashes)
# 3. LLM is instructed to say "I don't know" if context insufficient
# 4. Grading uses statistics (no LLM call), fast and free
#
# EXAMPLE 1: System Self-Corrects Successfully
# ──────────────────────────────────────────────
# Query: "What is OOP?"
#   [ATTEMPT 1]
#   - Retrieved: Generic Python/Java docs
#   - Grade: NO (score 0.15 < 0.25)
#   - Rewrite: "object-oriented programming principles"
#
#   [ATTEMPT 2]
#   - Retrieved: OOP-specific docs
#   - Grade: YES (score 0.78 ≥ 0.25)
#   - Answer: "Object-oriented programming is..." ✅
#
# EXAMPLE 2: System Gives Up Gracefully
# ────────────────────────────────────────
# Query: "What is SuperObscureLanguage?"
#   [ATTEMPT 1]
#   - Retrieved: Generic programming docs
#   - Grade: NO (score 0.08)
#   - Rewrite: Enhanced query
#
#   [ATTEMPT 2]
#   - Retrieved: Still no relevant docs (not in database)
#   - Grade: NO (score 0.10)
#   - retry_count = 1 → Accept anyway
#   - Answer: "Based on the provided context, I don't have
#             specific information about this language..." ⚠️
#
# CONFIGURATION:
# ──────────────
# export ENABLE_GRADING=false         # Disable grading (skip validation)
# export ENABLE_INTENT_ROUTING=false  # Use balanced weights (skip classification)
#
# ============================================================================
