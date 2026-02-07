# Why RAGChain is Agentic RAG, Not Traditional RAG

## Executive Summary

**This codebase implements Agentic RAG**, not traditional RAG. This document explains the fundamental differences, provides evidence from the code, and demonstrates why this system exhibits autonomous agent-like behavior.

---

## What Makes It Agentic?

RAGChain implements **4 autonomous agents** that make independent decisions, adapt strategies, self-correct errors, and coordinate via shared state—all hallmarks of agentic systems.

---

## 📊 Visual Comparison

![Traditional vs Agentic RAG](images/traditional-vs-agentic-rag.png)

### Traditional RAG: Linear Pipeline
```
Query → Embed → Search → Retrieve → Generate → Return
```
- **No decision-making**: Fixed path for all queries
- **No adaptation**: Same strategy regardless of query type
- **No self-correction**: Accepts whatever is retrieved
- **No state management**: Stateless pipeline

### Agentic RAG: Adaptive Multi-Agent System
```
Query → [Agent 1: Classify] → [Agent 2: Retrieve Adaptively] 
      → [Agent 3: Evaluate Quality] → Decision: Good? 
      → If NO: [Agent 4: Rewrite] → Retry
      → If YES: Generate → Return
```
- **Decision-making**: 4 agents make autonomous choices
- **Adaptation**: Strategy changes based on query intent
- **Self-correction**: Automatic retry on quality failure
- **State management**: LangGraph orchestrates agent coordination

---

## 🤖 The 4 Autonomous Agents

![Agentic Behavior](images/agentic-behavior.png)

### Agent 1: Intent Classifier (`intent_router`)

**Role:** Analyzes query semantics and routes to optimal retrieval strategy

**Autonomous Decision:** Classifies queries into FACT, CONCEPT, or COMPARISON

**Code Evidence:**
```python
# src/ragchain/inference/router.py:23-53
def intent_router(state: IntentRoutingState) -> IntentRoutingState:
    """
    Classify query intent to optimize retrieval weights
    
    Fast-path: Simple queries (e.g., "What is X?") skip LLM call
    Slow-path: Complex queries use LLM classification
    """
    query_lower = state["query"].lower()
    
    # AUTONOMOUS DECISION: Pattern matching vs LLM classification
    simple_patterns = ["what is", "define", "explain", ...]
    is_simple = any(pattern in query_lower for pattern in simple_patterns)
    
    if not config.enable_intent_routing or is_simple:
        return {**state, "intent": Intent.CONCEPT}  # Fast-path decision
    
    # LLM-based classification for complex queries
    llm = get_llm(purpose="routing")
    response = llm.invoke(prompt).strip().upper()
    intent_value = next((i for i in valid_intents if i.value in response), Intent.CONCEPT)
    
    return {**state, "intent": intent_value}  # Agent's decision
```

**Agentic Properties:**
- ✅ **Autonomy**: Decides whether to use fast-path or LLM
- ✅ **Reactivity**: Responds to query complexity
- ✅ **Goal-oriented**: Aims to classify accurately while minimizing cost

---

### Agent 2: Adaptive Retriever (`adaptive_retriever`)

**Role:** Dynamically adjusts BM25/Chroma weights based on query intent

**Autonomous Decision:** Selects weight configuration from 3 strategies

**Code Evidence:**
```python
# src/ragchain/inference/graph.py:41-71
def adaptive_retriever(state: IntentRoutingState) -> IntentRoutingState:
    """
    Retrieve documents using intent-specific weights
    
    Weights by intent:
    - FACT: 80% keyword (BM25), 20% semantic (Chroma)
    - CONCEPT: 40% keyword, 60% semantic
    - COMPARISON: 50% keyword, 50% semantic
    """
    query = state.get("rewritten_query") or state["query"]
    
    # AUTONOMOUS DECISION: Weight selection based on intent
    weights = {
        Intent.FACT: (0.8, 0.2),        # Keyword-heavy for lists
        Intent.CONCEPT: (0.4, 0.6),     # Semantic-heavy for explanations
        Intent.COMPARISON: (0.5, 0.5),  # Balanced for comparisons
    }
    bm25_weight, chroma_weight = weights.get(state["intent"], (0.5, 0.5))
    
    # Execute retrieval with agent's chosen weights
    retriever = get_ensemble_retriever(config.graph_k, 
                                      bm25_weight=bm25_weight, 
                                      chroma_weight=chroma_weight)
    docs = retriever.invoke(query)
    
    return {**state, "retrieved_docs": docs}
```

**Agentic Properties:**
- ✅ **Autonomy**: Independently selects retrieval strategy
- ✅ **Adaptivity**: Changes behavior based on context (intent)
- ✅ **Tool use**: Configures and uses ensemble retriever
- ✅ **Learning**: Strategy is informed by query characteristics

---

### Agent 3: Quality Grader (`retrieval_grader`)

**Role:** Evaluates if retrieved documents can answer the query

**Autonomous Decision:** Accept or reject retrieval results

**Code Evidence:**
```python
# src/ragchain/inference/graph.py:74-95
def retrieval_grader(state: IntentRoutingState) -> IntentRoutingState:
    """
    Grade if retrieved documents can answer the query
    
    Returns YES if docs are good, NO if they need improvement.
    Auto-accepts if: grading disabled, no docs, or already retried.
    """
    # AUTONOMOUS DECISION: Fast-path logic
    if should_skip_grading():
        return {**state, "retrieval_grade": GradeSignal.YES}
    
    if should_accept_docs(state["retrieved_docs"], state.get("retry_count", 0)):
        return {**state, "retrieval_grade": GradeSignal.YES}
    
    # AUTONOMOUS DECISION: Quality evaluation
    grade_value = grade_with_statistics(state["query"], state["retrieved_docs"])
    
    return {**state, "retrieval_grade": grade_value}  # YES or NO
```

**Supporting Code (Statistical Grading):**
```python
# src/ragchain/inference/grader.py:58-132
def grade_with_statistics(query: str, docs: list[Document]) -> GradeSignal:
    """
    Grade documents using keyword overlap + term frequency
    
    Scoring: 0.7 × overlap + 0.3 × TF
    Threshold: Top-3 docs must have score ≥ 0.25
    """
    query_keywords = extract_keywords(query)
    
    for doc in docs:
        doc_keywords = extract_keywords(doc.page_content)
        overlap_ratio = len(query_keywords & doc_keywords) / len(query_keywords)
        tf_score = sum(doc.count(kw) for kw in query_keywords) / len(query_keywords)
        score = 0.7 * overlap_ratio + 0.3 * min(tf_score, 1.0)
        # ... scoring logic ...
    
    # AUTONOMOUS DECISION: Accept or reject based on threshold
    return GradeSignal.YES if best_score >= 0.25 else GradeSignal.NO
```

**Agentic Properties:**
- ✅ **Autonomy**: Independently evaluates quality
- ✅ **Reactivity**: Responds to document relevance
- ✅ **Goal-oriented**: Aims for high-quality retrieval
- ✅ **Decision-making**: Binary accept/reject decision

---

### Agent 4: Query Rewriter (`query_rewriter`)

**Role:** Improves failed queries for better retrieval

**Autonomous Decision:** Enhances query with better search terms

**Code Evidence:**
```python
# src/ragchain/inference/graph.py:98-115
def query_rewriter(state: IntentRoutingState) -> IntentRoutingState:
    """
    Rewrite query for better retrieval (called when initial retrieval fails)
    
    Example: "OOP" → "What is object-oriented programming?"
    """
    llm = get_llm(purpose="rewriting")
    
    # AUTONOMOUS DECISION: Query enhancement strategy
    original = state["original_query"]
    prompt = QUERY_REWRITER_PROMPT.format(query=original)
    rewritten = llm.invoke(prompt).strip()
    
    return {
        **state, 
        "rewritten_query": rewritten,  # Agent's improved query
        "retry_count": state.get("retry_count", 0) + 1
    }
```

**Prompt for Query Rewriting:**
```python
# src/ragchain/prompts.py:91-107
QUERY_REWRITER_PROMPT = """Your previous retrieval didn't return relevant documents:
Original Query: {query}

Rewrite this query to be more explicit. For comparisons or synthesis questions, 
include BOTH concepts as separate searchable terms.

Examples:
- "What are the top 10 languages?" → "TIOBE index top 10 most popular programming..."
- "Compare Go and Rust" → "Go programming language features Rust programming..."
- "differences between interpreted and compiled" → "interpreted languages definition..."

Rewritten Query:"""
```

**Agentic Properties:**
- ✅ **Autonomy**: Decides how to improve the query
- ✅ **Persistence**: Triggered by failure, tries again
- ✅ **Goal-oriented**: Aims to improve retrieval success
- ✅ **Creativity**: Uses temperature=0.5 for varied rewrites

---

## 🔄 Agent Coordination via LangGraph

### State Management

**Code Evidence:**
```python
# src/ragchain/types.py (simplified)
class IntentRoutingState(TypedDict):
    query: str                          # Original user query
    original_query: str                 # Backup for rewriting
    intent: Intent                      # Agent 1's decision
    retrieved_docs: list[Document]      # Agent 2's output
    retrieval_grade: GradeSignal        # Agent 3's decision
    rewritten_query: str                # Agent 4's output
    retry_count: int                    # Coordination state
```

**LangGraph Orchestration:**
```python
# src/ragchain/inference/graph.py:122-168
workflow = StateGraph(IntentRoutingState)

# Register agents as nodes
workflow.add_node(Node.INTENT_ROUTER, intent_router)        # Agent 1
workflow.add_node(Node.ADAPTIVE_RETRIEVER, adaptive_retriever)  # Agent 2
workflow.add_node(Node.RETRIEVAL_GRADER, retrieval_grader)    # Agent 3
workflow.add_node(Node.QUERY_REWRITER, query_rewriter)       # Agent 4

# Define agent coordination flow
workflow.set_entry_point(Node.INTENT_ROUTER)
workflow.add_edge(Node.INTENT_ROUTER, Node.ADAPTIVE_RETRIEVER)
workflow.add_edge(Node.ADAPTIVE_RETRIEVER, Node.RETRIEVAL_GRADER)

# AUTONOMOUS DECISION: Conditional routing based on quality
def should_rewrite(state: IntentRoutingState) -> str:
    if state["retrieval_grade"] == GradeSignal.YES:
        return END  # Success, proceed to answer generation
    
    if state.get("retry_count", 0) >= 1:
        return END  # Already retried, give up gracefully
    
    return Node.QUERY_REWRITER  # Trigger self-correction

workflow.add_conditional_edges(
    Node.RETRIEVAL_GRADER,
    should_rewrite,  # Agent coordination logic
    {END: END, Node.QUERY_REWRITER: Node.QUERY_REWRITER}
)

# Self-correction loop: rewriter → retriever
workflow.add_edge(Node.QUERY_REWRITER, Node.ADAPTIVE_RETRIEVER)

rag_graph = workflow.compile()
```

---

## 📈 Comparison with Traditional RAG

| Characteristic | Traditional RAG | RAGChain (Agentic RAG) |
|----------------|----------------|------------------------|
| **Architecture** | Linear pipeline | Multi-agent system with orchestration |
| **Decision Making** | None (fixed flow) | 4 autonomous agents |
| **Adaptivity** | Fixed retrieval strategy | Intent-based dynamic weights |
| **Self-Correction** | No retry mechanism | Automatic query rewriting on failure |
| **State Management** | Stateless | LangGraph state across agents |
| **Tool Use** | 1 retriever | Multiple tools (BM25, Chroma, LLM) |
| **Coordination** | N/A | Agent-to-agent via conditional routing |
| **Autonomy Level** | Zero | High (4 decision points) |
| **Error Handling** | Fails or returns bad results | Self-corrects with retry (max 1) |
| **Optimization** | None | Cost-optimized fast-paths |
| **Query Types** | All treated identically | FACT/CONCEPT/COMPARISON strategies |

---

## 🔬 Evidence from Code Structure

### Traditional RAG Pattern (NOT in this codebase):
```python
def traditional_rag(query: str) -> str:
    # Fixed, linear flow
    embedding = embed(query)
    docs = vector_store.search(embedding, k=5)
    context = "\n".join(doc.content for doc in docs)
    answer = llm.generate(context, query)
    return answer
```

### Agentic RAG Pattern (in this codebase):
```python
def agentic_rag(query: str) -> str:
    # Multi-agent orchestration
    state = {
        "query": query,
        "intent": None,
        "retrieved_docs": [],
        "retrieval_grade": GradeSignal.NO,
        "retry_count": 0
    }
    
    # Agent 1: Classify intent
    state = intent_router(state)
    
    # Agent 2: Adaptive retrieval
    state = adaptive_retriever(state)
    
    # Agent 3: Grade quality
    state = retrieval_grader(state)
    
    # Agent 4 (conditional): Self-correct if needed
    if state["retrieval_grade"] == GradeSignal.NO and state["retry_count"] < 1:
        state = query_rewriter(state)
        state = adaptive_retriever(state)  # Retry
        state = retrieval_grader(state)
    
    # Final generation (after agent coordination)
    context = "\n".join(doc.page_content for doc in state["retrieved_docs"])
    answer = llm.generate(context, query)
    return answer
```

---

## 🎯 Why This Matters

### Traditional RAG Limitations:
1. ❌ **One-size-fits-all**: Same strategy for all queries
2. ❌ **No quality control**: Accepts whatever is retrieved
3. ❌ **No retry mechanism**: Fails on poor retrieval
4. ❌ **Rigid pipeline**: Cannot adapt to query characteristics
5. ❌ **No optimization**: Wastes compute on simple queries

### Agentic RAG Advantages (in RAGChain):
1. ✅ **Query-specific optimization**: FACT/CONCEPT/COMPARISON strategies
2. ✅ **Quality assurance**: Automatic relevance grading
3. ✅ **Self-correction**: Query rewriting on failure (max 1 retry)
4. ✅ **Adaptive pipeline**: Dynamic weight adjustment
5. ✅ **Cost optimization**: Fast-path routing for simple queries

---

## 📊 Real-World Example: Query Flow

### Example 1: "Top 10 programming languages" (FACT query)

**Agent Coordination:**
```
1. Intent Router (Agent 1)
   Input: "Top 10 programming languages"
   Decision: Classify as FACT (list query)
   Output: Intent.FACT

2. Adaptive Retriever (Agent 2)
   Input: Intent.FACT
   Decision: Use 80% BM25, 20% Chroma (keyword-heavy)
   Retrieves: 6 documents about language rankings
   Output: [Doc1: TIOBE index, Doc2: IEEE rankings, ...]

3. Quality Grader (Agent 3)
   Input: Query + Retrieved docs
   Score: Keyword overlap = 0.85 (high)
   Decision: ACCEPT (score ≥ 0.25)
   Output: GradeSignal.YES

4. Conditional Router (Orchestrator)
   Input: GradeSignal.YES
   Decision: Skip rewriting, proceed to generation
   Output: END (success path)

5. Answer Generation (LLM)
   Context: 6 high-quality documents
   Output: "According to TIOBE index, the top 10 languages are: 1. Python, 2. C, 3. Java..."
```

**Total LLM Calls:** 2 (intent classification + answer generation)

### Example 2: "OOP benefits" → Poor retrieval → Self-correction

**Agent Coordination with Retry:**
```
1. Intent Router (Agent 1)
   Input: "OOP benefits"
   Decision: Classify as CONCEPT
   Output: Intent.CONCEPT

2. Adaptive Retriever (Agent 2) - Attempt 1
   Input: Intent.CONCEPT
   Decision: Use 40% BM25, 60% Chroma
   Retrieves: 6 generic programming documents
   Output: [Doc1: General programming, Doc2: Python basics, ...]

3. Quality Grader (Agent 3) - Attempt 1
   Input: "OOP benefits" + Retrieved docs
   Score: Keyword overlap = 0.18 (too low)
   Decision: REJECT (score < 0.25)
   Output: GradeSignal.NO

4. Conditional Router (Orchestrator)
   Input: GradeSignal.NO, retry_count=0
   Decision: Trigger query rewriter (self-correction)
   Output: Node.QUERY_REWRITER

5. Query Rewriter (Agent 4)
   Input: "OOP benefits"
   Decision: Expand with synonyms and context
   Output: "object-oriented programming benefits advantages encapsulation inheritance polymorphism"

6. Adaptive Retriever (Agent 2) - Attempt 2 (Retry)
   Input: Rewritten query + Intent.CONCEPT
   Decision: Use 40% BM25, 60% Chroma
   Retrieves: 6 OOP-specific documents
   Output: [Doc1: OOP principles, Doc2: OOP advantages, ...]

7. Quality Grader (Agent 3) - Attempt 2
   Input: Rewritten query + New docs
   Score: Keyword overlap = 0.72 (high)
   Decision: ACCEPT (score ≥ 0.25)
   Output: GradeSignal.YES

8. Conditional Router (Orchestrator)
   Input: GradeSignal.YES, retry_count=1
   Decision: Proceed to generation (max retries reached)
   Output: END (success after retry)

9. Answer Generation (LLM)
   Context: 6 OOP-specific documents
   Output: "Object-oriented programming offers several benefits: encapsulation..."
```

**Total LLM Calls:** 3 (intent + rewriting + answer generation)

---

## 🏗️ Implementation Architecture

### LangGraph as Agent Orchestrator

```python
# RAGChain uses LangGraph for agent coordination
workflow = StateGraph(IntentRoutingState)  # Shared state

# Register 4 autonomous agents
workflow.add_node("intent_router", intent_router)
workflow.add_node("adaptive_retriever", adaptive_retriever)
workflow.add_node("retrieval_grader", retrieval_grader)
workflow.add_node("query_rewriter", query_rewriter)

# Define agent coordination logic
workflow.add_edge("intent_router", "adaptive_retriever")
workflow.add_edge("adaptive_retriever", "retrieval_grader")

# Conditional routing (agent coordination)
workflow.add_conditional_edges(
    "retrieval_grader",
    should_rewrite,  # Decision function
    {
        END: END,                           # Success path
        "query_rewriter": "query_rewriter"  # Self-correction path
    }
)

# Self-correction loop
workflow.add_edge("query_rewriter", "adaptive_retriever")

rag_graph = workflow.compile()  # Compile agent system
```

---

## 📚 References to Advanced RAG Document

This implementation aligns with strategies from `docs/advance-rag.md`:

### Implemented Agentic Strategies:

1. **Query Rewriting** (Strategy #5)
   - Location: `src/ragchain/inference/graph.py:98-115`
   - Status: ✅ Fully implemented
   - Agent: Query Rewriter

2. **Re-ranking** (Strategy #7)
   - Location: `src/ragchain/inference/retrievers.py`
   - Status: ✅ Implemented via RRF (Reciprocal Rank Fusion)
   - Agent: Adaptive Retriever

3. **Agentic RAG** (Strategy #10)
   - Location: `src/ragchain/inference/graph.py`
   - Status: ✅ Fully implemented with LangGraph
   - Agents: All 4 agents (Intent Router, Adaptive Retriever, Quality Grader, Query Rewriter)

### Additional Unique Agentic Features:

4. **Intent-based Adaptive Retrieval**
   - Agent: Intent Router + Adaptive Retriever
   - Unique to this implementation

5. **Self-Correcting RAG**
   - Agent: Quality Grader + Query Rewriter
   - Automatic retry mechanism

6. **Statistical Grading**
   - Agent: Quality Grader
   - Fast, cost-free quality control

---

## 🎯 Conclusion

### Is this Agentic RAG? **Yes, definitively.**

**Evidence:**
1. ✅ **4 autonomous agents** with independent decision-making
2. ✅ **LangGraph orchestration** for agent coordination
3. ✅ **Shared state management** (`IntentRoutingState`)
4. ✅ **Self-correction mechanism** (query rewriting on failure)
5. ✅ **Adaptive behavior** (intent-based weight adjustment)
6. ✅ **Goal-oriented design** (optimize retrieval quality)
7. ✅ **Tool use** (agents use multiple retrievers and LLMs)
8. ✅ **Conditional routing** (agents coordinate via decisions)

### Why It Matters:

**Traditional RAG:**
- Simple but rigid
- One strategy for all queries
- No quality control
- No retry mechanism

**Agentic RAG (RAGChain):**
- Complex but adaptive
- Query-specific strategies
- Automatic quality control
- Self-correcting with retry

### Key Differentiators:

| Traditional RAG | Agentic RAG (RAGChain) |
|----------------|------------------------|
| Fixed pipeline | Multi-agent orchestration |
| No decisions | 4 autonomous decision points |
| No adaptation | Intent-based dynamic weights |
| No retry | Automatic query rewriting |
| Stateless | LangGraph state management |

---

## 📖 Further Reading

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Agentic RAG Research Paper](https://arxiv.org/abs/2310.06117)
- [Multi-Agent Systems (Wooldridge)](https://www.wiley.com/en-us/An+Introduction+to+MultiAgent+Systems%2C+2nd+Edition-p-9780470519462)
- [Advanced RAG Strategies](advance-rag.md)
- [LLM Architecture Flow](llm-usage.md)
- [System Architecture](architecture.md)

---

**Last Updated:** 2026-02-07  
**Author:** Generated from codebase analysis  
**Version:** 1.0.0
