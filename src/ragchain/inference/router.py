"""
Intent Router - Classifies queries to optimize search strategy

Determines query type to adjust keyword vs semantic search weights:
- FACT: "Top 10 languages" → 80% keyword, 20% semantic
- CONCEPT: "What is Python?" → 40% keyword, 60% semantic
- COMPARISON: "Python vs Ruby" → 50% keyword, 50% semantic
"""

import logging

from ragchain.config import config
from ragchain.prompts import INTENT_ROUTER_PROMPT
from ragchain.types import Intent, IntentRoutingState
from ragchain.utils import get_llm, timed

logger = logging.getLogger(__name__)

__all__ = ["intent_router"]


@timed(logger, "intent_router")
def intent_router(state: IntentRoutingState) -> IntentRoutingState:
    """
    Classify query intent to optimize retrieval weights

    Fast-path: Simple queries (e.g., "What is X?") skip LLM call
    Slow-path: Complex queries use LLM classification
    """

    query_lower = state["query"].lower()

    # Patterns that indicate simple CONCEPT queries
    simple_patterns = ["what is", "define", "explain", "who is", "when was", "where is", "how does", "why is"]
    is_simple = any(pattern in query_lower for pattern in simple_patterns) and len(state["query"].split()) <= 8

    # Fast-path: Skip LLM for simple queries or if routing disabled
    if not config.enable_intent_routing or is_simple:
        logger.debug("[intent_router] Fast-path → CONCEPT")
        return {**state, "intent": Intent.CONCEPT, "original_query": state["query"]}

    # Slow-path: Use LLM to classify intent
    llm = get_llm(purpose="routing")
    prompt = INTENT_ROUTER_PROMPT.format(query=state["query"])
    response = llm.invoke(prompt).strip().upper()

    # Extract intent from LLM response (defaults to CONCEPT if unclear)
    valid_intents: list[Intent] = [Intent.FACT, Intent.CONCEPT, Intent.COMPARISON]
    intent_value: Intent = next((i for i in valid_intents if i.value in response), Intent.CONCEPT)

    logger.debug(f"[intent_router] LLM classified → {intent_value}")

    return {**state, "intent": intent_value, "original_query": state["query"]}
