# How LangGraph Orchestrates Everything

Here's the complete flow:

```
┌─────────────────────────────────────────────────────────────┐
│                    LangGraph State Flow                     │
└─────────────────────────────────────────────────────────────┘

User Query: "What is Python?"
         │
         ▼
┌────────────────────┐
│ 1. INTENT_ROUTER   │ ← router.py::intent_router()
│ (router.py)        │   Classifies: FACT/CONCEPT/COMPARISON
└────────┬───────────┘
         │ intent = CONCEPT
         ▼
┌────────────────────┐
│ 2. ADAPTIVE_       │ ← graph.py::adaptive_retriever()
│    RETRIEVER       │   Uses retrievers.py::get_ensemble_retriever()
│ (graph.py)         │   Weights: CONCEPT = 0.4 BM25 / 0.6 Chroma
└────────┬───────────┘
         │ retrieved_docs = [...]
         ▼
┌────────────────────┐
│ 3. RETRIEVAL_      │ ← graph.py::retrieval_grader()
│    GRADER          │   Uses grader.py::grade_with_statistics()
│ (graph.py)         │
└────────┬───────────┘
         │
         ├─→ Grade = YES → END (return docs)
         │
         └─→ Grade = NO & retry_count < 1
                    │
                    ▼
         ┌────────────────────┐
         │ 4. QUERY_REWRITER  │ ← graph.py::query_rewriter()
         │ (graph.py)         │   Rewrites query using LLM
         └────────┬───────────┘
                  │ rewritten_query
                  │
                  └──→ Back to ADAPTIVE_RETRIEVER (retry)
```

State Management
LangGraph uses IntentRoutingState (from types.py) to pass data between nodes:

```
{
    "query": str,              # Current query
    "original_query": str,     # Original (before rewrite)
    "intent": Intent,          # FACT/CONCEPT/COMPARISON
    "retrieved_docs": list,    # Retrieved documents
    "retrieval_grade": str,    # YES/NO
    "rewritten_query": str,    # Rewritten version
    "retry_count": int         # Number of retries
}
```
Invocation from CLI
```
# cli.py line 109
final_state = rag_graph.invoke(initial_state)
retrieved_docs = final_state["retrieved_docs"]
```

🔧 Could You Use Tools Instead of LangGraph?
Yes, absolutely! Here are the alternatives:
```
# Instead of LangGraph
def simple_rag_pipeline(query: str):
    # Step 1: Route
    intent = classify_intent(query)
    
    # Step 2: Retrieve with weights
    weights = get_weights_for_intent(intent)
    docs = ensemble_retrieve(query, weights)
    
    # Step 3: Grade
    if not grade_documents(query, docs):
        # Step 4: Retry
        rewritten = rewrite_query(query)
        docs = ensemble_retrieve(rewritten, weights)
    
    return docs
```

Pros:
- Simpler, easier to understand
- Less dependencies
- Faster for simple cases

Cons:
- Manual state management
- Harder to add conditional paths
- No built-in retry/loop handling
- No visualization tools

# Alternative 2: LangChain LCEL (LangChain Expression Language)

```
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

pipeline = (
    {"query": RunnablePassthrough()}
    | RunnableLambda(intent_router)
    | RunnableLambda(adaptive_retriever)
    | RunnableLambda(retrieval_grader)
)
```

Pros:
Cleaner syntax
Built into LangChain
Good for linear pipelines
Cons:
Difficult for conditional loops (like retry logic)
Limited state inspection
No visual graph representation

# Alternative 3: LangChain Tools + Agent

```
from langchain.agents import Tool, AgentExecutor

tools = [
    Tool(name="retriever", func=adaptive_retriever),
    Tool(name="grader", func=retrieval_grader),
    Tool(name="rewriter", func=query_rewriter)
]

agent = create_react_agent(llm, tools)
```

# 🎯 Why LangGraph is Perfect Here:

- Your codebase uses LangGraph because:
- Conditional Routing: Need to decide "retry or end?" based on grade
- State Persistence: Need to track retry_count, original_query, etc.
- Visual Debugging: Can visualize the graph structure
- Deterministic: Same query → same path (unlike agents)
- Efficient: No extra LLM calls for orchestration
- Type-Safe: IntentRoutingState ensures correct data flow


# Other files in `inference` folder :

## 1. retrievers.py - The Foundation
- What it does: Implements ensemble retrieval combining BM25 (keyword) + Chroma (semantic)
- Key components:
- EnsembleRetriever: Custom retriever using Reciprocal Rank Fusion (RRF) algorithm
- Parallel execution (ThreadPoolExecutor) for BM25 + Chroma simultaneously
- get_ensemble_retriever(): Factory function with LRU cache to avoid rebuilding BM25 index
- Used by: rag.py, graph.py

## 2. rag.py - Simple Search API
- What it does: Provides a straightforward search function without orchestration
- async def search(query: str, k: int) -> SearchResult
- Used by: CLI search command (line 65 in cli.py)
- When used: When you just want semantic search without LLM generation
- Example: uv run ragchain search "Python programming"

## 3. router.py - Intent Classification
- What it does: Classifies queries into 3 categories:
- FACT → Lists, rankings, enumerations (e.g., "top 10 languages")
- CONCEPT → Definitions, explanations (e.g., "What is Python?")
- COMPARISON → Comparing entities (e.g., "Compare Go vs Rust")
- Features:
- Fast-path for simple queries (pattern matching, no LLM call)
- LLM-based classification for complex queries
- Used by: graph.py as the entry point node

## 4. grader.py - Relevance Validation
- What it does: Grades if retrieved documents actually answer the query
- Grading logic:
- Keyword overlap scoring (Jaccard-like + TF scoring)
- Checks top-3 documents for relevance threshold (≥0.25 score)
- Uses MRR (Mean Reciprocal Rank) inspired scoring
- Used by: graph.py in the grading node

## 5. graph.py - LangGraph Orchestrator (THE BRAIN) 🧠
- What it does: Orchestrates the entire adaptive RAG pipeline with state management
- Refer Above diagram / flow
