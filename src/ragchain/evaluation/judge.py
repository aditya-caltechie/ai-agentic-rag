"""
LLM-as-Judge Evaluation - Score RAG answer quality

Evaluates generated answers on 3 dimensions (1-5 scale):
- Correctness: Is the answer factually accurate?
- Relevance: Does it answer the question?
- Faithfulness: Is it grounded in the context (no hallucinations)?

Used by: `ragchain evaluate` command
"""

import json
import logging

from langchain_core.prompts import ChatPromptTemplate

from ragchain.config import config
from ragchain.prompts import JUDGE_PROMPT, RAG_ANSWER_TEMPLATE
from ragchain.utils import get_llm, log_with_prefix, timed

logger = logging.getLogger(__name__)


# ============================================================================
# Helper Functions
# ============================================================================


def extract_json_object(text: str) -> dict | None:
    """
    Extract JSON from LLM response (handles markdown wrappers)

    Example: "Here's the score: {\"correctness\": 5}" → {"correctness": 5}
    """
    # Find first { and match closing }
    start_idx = text.find("{")
    if start_idx == -1:
        return None

    # Count braces to find matching }
    brace_count = 0
    end_idx = start_idx
    for i, char in enumerate(text[start_idx:], start_idx):
        if char == "{":
            brace_count += 1
        elif char == "}":
            brace_count -= 1
            if brace_count == 0:
                end_idx = i
                break

    if brace_count != 0:
        return None

    # Parse JSON
    json_str = text[start_idx : end_idx + 1]
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return None


# ============================================================================
# Main Judging Function
# ============================================================================


@timed(logger, "judge_answer")
async def judge_answer(question: str, context: str, answer: str, model: str = config.ollama_model) -> dict:
    """
    Evaluate a RAG answer using LLM-as-judge

    Scores (1-5 scale):
    - Correctness: Factual accuracy
    - Relevance: Answers the question
    - Faithfulness: No hallucinations (grounded in context)

    Returns: {"correctness": {"score": 5, "explanation": "..."}, ...}
    """

    # Step 1: Truncate context (balance speed vs quality)
    max_context_chars = 2500
    if len(context) > max_context_chars:
        truncated_context = context[:max_context_chars] + "\n\n[...truncated...]"
    else:
        truncated_context = context

    # Step 2: Get LLM and create prompt
    llm = get_llm(model=model, purpose="judging")
    prompt = ChatPromptTemplate.from_template(JUDGE_PROMPT)
    judge_input = prompt.format(question=question, context=truncated_context, answer=answer)

    # Step 3: Ask LLM to judge
    raw_response = llm.invoke(judge_input)

    # Step 4: Parse JSON response
    try:
        # Try direct JSON parsing first
        evaluation = json.loads(raw_response.strip())
    except json.JSONDecodeError:
        # Try extracting JSON from markdown/text wrapper
        evaluation = extract_json_object(raw_response)

        if not (evaluation and "correctness" in evaluation):
            # Parsing failed completely
            log_with_prefix(logger, logging.ERROR, "judge_answer", f"Failed to parse response: {raw_response[:500]}")
            return {
                "correctness": {"score": 0, "explanation": "Failed to parse response"},
                "relevance": {"score": 0, "explanation": "Failed to parse response"},
                "faithfulness": {"score": 0, "explanation": "Failed to parse response"},
            }

    # Step 5: Validate scores (must be 1-5)
    for criterion in ["correctness", "relevance", "faithfulness"]:
        if criterion in evaluation and isinstance(evaluation[criterion], dict):
            score = evaluation[criterion].get("score", 0)

            # Check if score is valid (integer between 1-5)
            if not isinstance(score, int) or score < 1 or score > 5:
                logger.warning(f"Invalid {criterion} score {score}, marking as parse error")
                evaluation[criterion]["score"] = 0
                evaluation[criterion]["explanation"] = f"Invalid score: {score}. " + evaluation[criterion].get("explanation", "")

    return evaluation


# ============================================================================
# Batch Evaluation Function
# ============================================================================


async def evaluate_questions(questions: list[str], model: str = config.ollama_model) -> list[dict]:
    """
    Evaluate RAG system on multiple questions

    For each question:
    1. Run RAG pipeline (retrieve docs)
    2. Generate answer
    3. Judge answer quality

    Returns: List of {question, answer, evaluation}
    """
    from ragchain.inference.graph import rag_graph

    llm = get_llm(model=model, purpose="generation")
    evaluations = []

    for question in questions:
        # Step 1: Run RAG pipeline to get documents
        initial_state = {
            "query": question,
            "intent": "CONCEPT",
            "retrieved_docs": [],
            "retrieval_grade": "NO",
            "rewritten_query": "",
            "retry_count": 0,
        }

        final_state = rag_graph.invoke(initial_state)  # type: ignore[arg-type]
        retrieved_docs = final_state["retrieved_docs"]

        logger.info(f"Retrieved {len(retrieved_docs)} docs for: {question[:50]}...")

        # Skip if no documents retrieved
        if not retrieved_docs:
            continue

        # Step 2: Generate answer from documents
        prompt = ChatPromptTemplate.from_template(RAG_ANSWER_TEMPLATE)
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        answer = llm.invoke(prompt.format(context=context, question=question))

        # Step 3: Judge the answer
        evaluation = await judge_answer(question, context, answer, model)

        # Collect results
        evaluations.append({"question": question, "answer": answer, "evaluation": evaluation})

    return evaluations
