# RAGChain - LLM Architecture & Flow Diagram

## 🔍 Where LLMs Are Called

This document identifies **ALL** locations where LLMs are invoked in the RAGChain system and provides a visual flow diagram.

---

## 🎨 Visual Architecture Diagrams

### Full System Architecture

![RAGChain Architecture](images/architecture.png)

**Legend:**
- 🔴 **Red Star with "LLM CALL" badge** = LLM invocation point (prompt + generate)
- ✅ **Green Checkmark with "No LLM"** = Statistical/algorithmic operation (no API call)
- 🔵 **Blue** = Data ingestion pipeline
- 🟢 **Green** = Search/retrieval operations
- 🟠 **Orange** = RAG pipeline with adaptive routing
- 🟣 **Purple** = Evaluation workflow

### LLM Invocation Points & Configuration

![LLM Usage Details](images/llm-usage.png)

This diagram shows:
- **4 LLM call points** with detailed configurations (temperature, tokens, context window)
- **Query flow** showing when each LLM is invoked
- **Configuration table** comparing settings across all LLM calls
- **Cost metrics** showing per-query LLM usage (1-3 calls)
- **Conditional paths** highlighting fast-paths and retries

**Key Insights:**
- Only **1 LLM call** is required (if routing is skipped and no retry needed)
- **2 LLM calls** is typical (routing + generation)
- **3 LLM calls** maximum in production (routing + rewriting + generation)
- **Statistical operations** (grader, retriever) avoid LLM costs

---

## 📍 LLM Call Locations

### 1. **Intent Router** (`src/ragchain/inference/router.py`)
- **Function**: `intent_router()`
- **Purpose**: Classify query intent (FACT/CONCEPT/COMPARISON)
- **LLM Model**: `get_llm(purpose="routing")`
- **Settings**: 
  - `temperature=0.0` (deterministic)
  - `num_predict=32` (short output)
  - `num_ctx=config.ollama_routing_ctx`
- **Prompt**: `INTENT_ROUTER_PROMPT`
- **When Called**: Start of RAG pipeline (unless fast-path detected)
- **Output**: Intent classification (FACT/CONCEPT/COMPARISON)

```python
llm = get_llm(purpose="routing")
prompt = INTENT_ROUTER_PROMPT.format(query=state["query"])
response = llm.invoke(prompt).strip().upper()
```

---

### 2. **Query Rewriter** (`src/ragchain/inference/graph.py`)
- **Function**: `query_rewriter()`
- **Purpose**: Enhance failed queries for better retrieval
- **LLM Model**: `get_llm(purpose="rewriting")`
- **Settings**: 
  - `temperature=0.5` (creative)
  - `num_predict=128` (moderate output)
  - `num_ctx=config.ollama_rewriting_ctx`
- **Prompt**: `QUERY_REWRITER_PROMPT`
- **When Called**: When retrieval grading fails (max 1 retry)
- **Output**: Enhanced query string

```python
llm = get_llm(purpose="rewriting")
prompt = QUERY_REWRITER_PROMPT.format(query=original)
rewritten = llm.invoke(prompt).strip()
```

---

### 3. **Answer Generator** (`src/ragchain/cli.py` & `src/ragchain/evaluation/judge.py`)
- **Function**: `cli.ask()` and `evaluate_questions()`
- **Purpose**: Generate natural language answers from retrieved context
- **LLM Model**: `get_llm(model=model, purpose="generation")`
- **Settings**: 
  - `temperature=0.1` (mostly deterministic)
  - `num_predict=1024` (long output)
  - `num_ctx=config.ollama_gen_ctx`
  - `reasoning=True` (enables chain-of-thought)
- **Prompt**: `RAG_ANSWER_TEMPLATE`
- **When Called**: After retrieval, to generate final answer
- **Output**: Natural language answer

```python
llm = get_llm(model=model, purpose="generation")
prompt = ChatPromptTemplate.from_template(RAG_ANSWER_TEMPLATE)
answer = llm.invoke(prompt.format(context=context, question=query))
```

---

### 4. **LLM-as-Judge** (`src/ragchain/evaluation/judge.py`)
- **Function**: `judge_answer()`
- **Purpose**: Evaluate answer quality (correctness, relevance, faithfulness)
- **LLM Model**: `get_llm(model=model, purpose="judging")`
- **Settings**: 
  - `temperature=0.0` (deterministic)
  - `num_predict=512` (JSON output)
  - `num_ctx=config.ollama_judging_ctx`
- **Prompt**: `JUDGE_PROMPT`
- **When Called**: During evaluation (`ragchain evaluate` command)
- **Output**: JSON scores (1-5 scale for 3 dimensions)

```python
llm = get_llm(model=model, purpose="judging")
prompt = ChatPromptTemplate.from_template(JUDGE_PROMPT)
raw_response = llm.invoke(judge_input)
evaluation = json.loads(raw_response.strip())
```

---

## 🚫 Where LLMs Are NOT Called

### Statistical Grading (No LLM)
- **File**: `src/ragchain/inference/grader.py`
- **Function**: `grade_with_statistics()`
- **Method**: Keyword overlap + term frequency scoring
- **Reason**: Fast, cost-free, effective for relevance checking

---

## 📊 Architecture Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         USER INTERACTION (CLI Commands)                          │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                ┌───────────────────────┼───────────────────────┐
                │                       │                       │
                ▼                       ▼                       ▼
        ┌───────────────┐      ┌───────────────┐      ┌───────────────┐
        │  ragchain     │      │  ragchain     │      │  ragchain     │
        │  ingest       │      │  search       │      │  ask          │
        │               │      │               │      │               │
        │  (No LLM)     │      │  (No LLM)     │      │  (Full RAG)   │
        └───────────────┘      └───────────────┘      └───────┬───────┘
                │                       │                      │
                ▼                       ▼                      ▼
        ┌───────────────┐      ┌───────────────┐      ┌───────────────────┐
        │ Load & Chunk  │      │  Ensemble     │      │   RAG GRAPH       │
        │  Documents    │      │  Retrieval    │      │   (LangGraph)     │
        │               │      │  (BM25+Chroma)│      │                   │
        └───────────────┘      └───────────────┘      └───────┬───────────┘
                │                       │                      │
                ▼                       │                      │
        ┌───────────────┐              │                      │
        │  Embed &      │              │                      │
        │  Store to     │              │                      │
        │  Chroma DB    │              │                      │
        └───────────────┘              │                      │
                                       │                      │
┌──────────────────────────────────────┴──────────────────────┴──────────────────┐
│                                                                                  │
│                        FULL RAG PIPELINE (LangGraph Flow)                       │
│                                                                                  │
│  ┌──────────────────────────────────────────────────────────────────────────┐  │
│  │  STEP 1: INTENT ROUTER                                                   │  │
│  │  ═══════════════════════                                                 │  │
│  │  📍 LLM CALL #1: Intent Classification                                   │  │
│  │  ────────────────────────────────────                                    │  │
│  │  • File: src/ragchain/inference/router.py                                │  │
│  │  • Function: intent_router()                                             │  │
│  │  • Purpose: Classify query type                                          │  │
│  │  • Model: get_llm(purpose="routing")                                     │  │
│  │  • Config: temp=0.0, num_predict=32                                      │  │
│  │  • Prompt: INTENT_ROUTER_PROMPT                                          │  │
│  │  • Input: User query                                                     │  │
│  │  • Output: FACT | CONCEPT | COMPARISON                                   │  │
│  │                                                                           │  │
│  │  Fast-path: Simple "What is X?" queries skip LLM                         │  │
│  │  Feature flag: ENABLE_INTENT_ROUTING=false → skip                        │  │
│  └──────────────────────────────────────────────────────────────────────────┘  │
│                                    ↓                                            │
│  ┌──────────────────────────────────────────────────────────────────────────┐  │
│  │  STEP 2: ADAPTIVE RETRIEVER                                              │  │
│  │  ═══════════════════════════                                             │  │
│  │  🔍 NO LLM - Uses Statistical Retrieval                                  │  │
│  │  ──────────────────────────────────                                      │  │
│  │  • File: src/ragchain/inference/graph.py                                 │  │
│  │  • Function: adaptive_retriever()                                        │  │
│  │  • Method: Ensemble (BM25 + Chroma vector search)                        │  │
│  │  • Uses: get_ensemble_retriever() with intent-specific weights:          │  │
│  │    - FACT: 80% BM25, 20% Chroma (keyword-heavy)                          │  │
│  │    - CONCEPT: 40% BM25, 60% Chroma (semantic-heavy)                      │  │
│  │    - COMPARISON: 50% BM25, 50% Chroma (balanced)                         │  │
│  │  • Algorithm: Reciprocal Rank Fusion (RRF)                               │  │
│  │  • Output: List of Document objects                                      │  │
│  └──────────────────────────────────────────────────────────────────────────┘  │
│                                    ↓                                            │
│  ┌──────────────────────────────────────────────────────────────────────────┐  │
│  │  STEP 3: RETRIEVAL GRADER                                                │  │
│  │  ═══════════════════════                                                 │  │
│  │  📊 NO LLM - Uses Statistical Scoring                                    │  │
│  │  ──────────────────────────────────────                                  │  │
│  │  • File: src/ragchain/inference/grader.py                                │  │
│  │  • Function: grade_with_statistics()                                     │  │
│  │  • Method:                                                               │  │
│  │    1. Extract keywords from query and docs                               │  │
│  │    2. Calculate overlap ratio (Jaccard similarity)                       │  │
│  │    3. Calculate term frequency (TF)                                      │  │
│  │    4. Score = 0.7 × overlap + 0.3 × TF                                   │  │
│  │    5. Check if top-3 docs have score ≥ 0.25                              │  │
│  │  • Output: YES | NO (GradeSignal)                                        │  │
│  │                                                                           │  │
│  │  Fast-paths:                                                             │  │
│  │  • ENABLE_GRADING=false → Always YES                                     │  │
│  │  • No docs → Always YES                                                  │  │
│  │  • Already retried → Always YES (prevent loops)                          │  │
│  └──────────────────────────────────────────────────────────────────────────┘  │
│                                    ↓                                            │
│                      ┌─────────────┴─────────────┐                              │
│                      │    Grade Result?          │                              │
│                      └─────────────┬─────────────┘                              │
│                                    │                                            │
│              ┌─────────────────────┼─────────────────────┐                      │
│              │ YES                 │                     │ NO                   │
│              │                     │                     │                      │
│              ▼                     ▼                     ▼                      │
│     ┌────────────────┐    ┌────────────────┐    ┌────────────────┐             │
│     │  END (Success) │    │  Already Retry │    │ QUERY REWRITER │             │
│     │  Documents OK  │    │  retry_count≥1 │    │  retry_count=0 │             │
│     └────────────────┘    └────────┬───────┘    └────────┬───────┘             │
│                                    │                      │                     │
│                                    ▼                      ▼                     │
│                            ┌────────────────┐    ┌────────────────────────┐    │
│                            │  END (Give up) │    │  STEP 4: QUERY_REWRITER│    │
│                            │  Accept anyway │    │  ═══════════════════════│   │
│                            └────────────────┘    │  📍 LLM CALL #2:        │   │
│                                                  │     Query Enhancement   │    │
│                                                  │  ─────────────────────  │    │
│                                                  │  • File: graph.py       │    │
│                                                  │  • Function:            │    │
│                                                  │    query_rewriter()     │    │
│                                                  │  • Model: get_llm(      │    │
│                                                  │    purpose="rewriting") │    │
│                                                  │  • Config: temp=0.5,    │    │
│                                                  │    num_predict=128      │    │
│                                                  │  • Prompt:              │    │
│                                                  │    QUERY_REWRITER_PROMPT│    │
│                                                  │  • Input: Original query│    │
│                                                  │  • Output: Enhanced     │    │
│                                                  │    query string         │    │
│                                                  └────────┬───────────────┘    │
│                                                           │                     │
│                                                           ▼                     │
│                                              ┌────────────────────────┐         │
│                                              │  RETRY: Go back to     │         │
│                                              │  ADAPTIVE_RETRIEVER    │         │
│                                              │  (with rewritten query)│         │
│                                              └────────────────────────┘         │
│                                                           │                     │
│                                              (Loop back to STEP 2, max 1 time)  │
│                                                                                  │
└──────────────────────────────────────────────────────────────────────────────────┘
                                    ↓
                                    ↓
        ┌───────────────────────────────────────────────────────────┐
        │  STEP 5: ANSWER GENERATION                                │
        │  ═════════════════════════                                │
        │  📍 LLM CALL #3: Generate Natural Language Answer         │
        │  ──────────────────────────────────────────────            │
        │  • File: src/ragchain/cli.py (ask command)                │
        │  • Function: cli.ask()                                    │
        │  • Model: get_llm(model=model, purpose="generation")      │
        │  • Config: temp=0.1, num_predict=1024, reasoning=True     │
        │  • Prompt: RAG_ANSWER_TEMPLATE                            │
        │  • Input: Retrieved documents + user query                │
        │  • Output: Natural language answer                        │
        │                                                           │
        │  Rules:                                                   │
        │  • ONLY use information from retrieved context            │
        │  • Say "I don't know" if context insufficient             │
        │  • Direct quotes preferred over summaries                 │
        │  • 150-300 word answers                                   │
        └───────────────────────────────────────────────────────────┘
                                    ↓
                                    ▼
                            ┌───────────────┐
                            │  Return Answer│
                            │  to User      │
                            └───────────────┘
                                    │
                                    │ (Optional: Evaluation)
                                    ▼
        ┌───────────────────────────────────────────────────────────┐
        │  STEP 6: LLM-AS-JUDGE (Optional - ragchain evaluate)     │
        │  ════════════════════════════════════════════════         │
        │  📍 LLM CALL #4: Evaluate Answer Quality                  │
        │  ────────────────────────────────────────                 │
        │  • File: src/ragchain/evaluation/judge.py                 │
        │  • Function: judge_answer()                               │
        │  • Model: get_llm(model=model, purpose="judging")         │
        │  • Config: temp=0.0, num_predict=512                      │
        │  • Prompt: JUDGE_PROMPT                                   │
        │  • Input: Question + Context + Answer                     │
        │  • Output: JSON with 3 scores (1-5 scale):                │
        │    - correctness: Factual accuracy                        │
        │    - relevance: Answers the question                      │
        │    - faithfulness: No hallucinations                      │
        │                                                           │
        │  Used by: ragchain evaluate command                       │
        └───────────────────────────────────────────────────────────┘
```

---

## 🔢 LLM Call Summary

| # | Component | File | Function | Purpose | Model Config | When Called |
|---|-----------|------|----------|---------|--------------|-------------|
| **1** | Intent Router | `router.py` | `intent_router()` | Query classification | `temp=0.0, predict=32` | Start of RAG (unless fast-path) |
| **2** | Query Rewriter | `graph.py` | `query_rewriter()` | Query enhancement | `temp=0.5, predict=128` | After grading fails (max 1×) |
| **3** | Answer Generator | `cli.py`, `judge.py` | `ask()`, `evaluate_questions()` | Generate answer | `temp=0.1, predict=1024` | After retrieval completes |
| **4** | LLM-as-Judge | `judge.py` | `judge_answer()` | Answer evaluation | `temp=0.0, predict=512` | During evaluation only |

---

## 🎯 Key Design Decisions

### Why Some Steps Use LLM, Others Don't

1. **Intent Router (LLM)**: 
   - Complex semantic understanding needed
   - Fast-path optimization for simple queries
   - Can be disabled (`ENABLE_INTENT_ROUTING=false`)

2. **Retrieval Grader (NO LLM)**:
   - Statistical scoring is fast & free
   - Keyword overlap sufficient for relevance
   - Prevents unnecessary API calls

3. **Query Rewriter (LLM)**:
   - Requires creative rephrasing
   - Only called on failure (rare)
   - Higher temperature (0.5) for creativity

4. **Answer Generator (LLM)**:
   - Core RAG functionality
   - Requires natural language synthesis
   - Reasoning mode enabled

5. **Judge (LLM)**:
   - Evaluation only (not production path)
   - Requires nuanced quality assessment
   - Runs in batch mode

---

## 🚀 Optimization Strategies

### Performance Optimizations
1. **Fast-path routing**: Simple queries skip LLM classification
2. **Grading can be disabled**: `ENABLE_GRADING=false`
3. **Max 1 retry**: Prevents infinite loops
4. **Token limits**: All LLMs have `num_predict` caps
5. **Statistical grading**: No LLM cost for quality checks

### Cost Optimizations
1. **Purpose-specific models**: Different context windows per use case
2. **Conditional LLM calls**: Routing/rewriting only when needed
3. **Caching**: Ensemble retriever caches results
4. **Parallel retrieval**: BM25 + Chroma run concurrently

---

## 📝 Configuration Options

All LLM configurations are centralized in `src/ragchain/utils.py::get_llm()`:

```python
purpose_defaults = {
    "generation":  {"temperature": 0.1, "num_ctx": 8192, "num_predict": 1024, "reasoning": True},
    "routing":     {"temperature": 0.0, "num_ctx": 4096, "num_predict": 32,   "reasoning": False},
    "judging":     {"temperature": 0.0, "num_ctx": 4096, "num_predict": 512,  "reasoning": False},
    "rewriting":   {"temperature": 0.5, "num_ctx": 4096, "num_predict": 128,  "reasoning": False},
}
```

---

## 🔍 Tracing LLM Calls in Code

All LLM invocations follow this pattern:

```python
# 1. Get LLM with purpose-specific config
llm = get_llm(purpose="routing")  # or "generation", "judging", "rewriting"

# 2. Format prompt
prompt = SOME_PROMPT_TEMPLATE.format(query=user_query)

# 3. Invoke LLM
response = llm.invoke(prompt)

# 4. Process response
result = response.strip()  # or json.loads(response), etc.
```

Search for `.invoke(` to find all LLM calls:
```bash
rg "\.invoke\(" --type py src/ragchain/
```

---

## 📚 Related Documentation

- **Full codebase walkthrough**: `docs/codewalk.md`
- **LangGraph details**: `docs/langGraph.md`
- **Advanced RAG strategies**: `docs/advanceRAG_strategies.md`
- **Project overview**: `AGENTS.md`

---

## 🎓 Learning Resources

**Key LangChain Concepts Used:**
- `OllamaLLM`: LLM wrapper for local Ollama models
- `ChatPromptTemplate`: Prompt formatting
- `StateGraph`: LangGraph state management
- `EnsembleRetriever`: Custom RRF retrieval

**External Dependencies:**
- **Ollama**: Local LLM runtime
- **Chroma**: Vector database
- **LangChain**: RAG orchestration framework
- **LangGraph**: Agentic workflow framework

---

## 📊 Example Flow: "What is Python?"

```
1. User: "What is Python?"
   ↓
2. Intent Router (LLM Call #1)
   - Fast-path detected: "what is" pattern
   - Skip LLM, return: CONCEPT
   ↓
3. Adaptive Retriever (No LLM)
   - CONCEPT → 40% BM25, 60% Chroma
   - Retrieve 6 documents
   ↓
4. Retrieval Grader (No LLM)
   - Extract keywords: {"python"}
   - Score docs: [0.85, 0.72, 0.68, ...]
   - Top doc score ≥ 0.25 → YES
   ↓
5. Answer Generator (LLM Call #2)
   - Context: 6 document chunks
   - Prompt: RAG_ANSWER_TEMPLATE
   - Output: "Python is a high-level, interpreted..."
   ↓
6. Return answer to user
```

**Total LLM Calls**: **1** (skipped intent routing via fast-path)

---

## 📊 Example Flow: "Compare Go and Rust" (with retry)

```
1. User: "Compare Go and Rust"
   ↓
2. Intent Router (LLM Call #1)
   - LLM classifies: COMPARISON
   ↓
3. Adaptive Retriever (No LLM)
   - COMPARISON → 50% BM25, 50% Chroma
   - Retrieve 6 documents
   - Assume: Got only Go docs, no Rust
   ↓
4. Retrieval Grader (No LLM)
   - Keywords: {"compare", "rust"}
   - Top doc score: 0.18 < 0.25 → NO
   ↓
5. Query Rewriter (LLM Call #2)
   - Original: "Compare Go and Rust"
   - Rewritten: "Go programming language features performance Rust programming..."
   - retry_count = 1
   ↓
6. Adaptive Retriever (Retry, No LLM)
   - Use rewritten query
   - Retrieve 6 documents
   - Now has both Go AND Rust docs
   ↓
7. Retrieval Grader (No LLM)
   - Top doc score: 0.67 ≥ 0.25 → YES
   ↓
8. Answer Generator (LLM Call #3)
   - Context: 6 mixed Go/Rust docs
   - Output: "Go and Rust are both systems languages..."
   ↓
9. Return answer to user
```

**Total LLM Calls**: **3** (routing + rewriting + generation)

---

## 🔧 Debugging LLM Calls

Enable debug logging to see all LLM interactions:

```bash
export LOG_LEVEL=DEBUG
uv run ragchain ask "Your query"
```

Look for these log patterns:
- `[intent_router]` → Intent classification
- `[query_rewriter]` → Query enhancement
- `[adaptive_retriever]` → Document retrieval (no LLM)
- `[retrieval_grader]` → Quality check (no LLM)
- Final answer generation (in CLI output)

---

## 📌 Summary

**Total LLM Call Points: 4**
1. Intent Router (conditional, can skip)
2. Query Rewriter (conditional, only on failure)
3. Answer Generator (always)
4. LLM-as-Judge (evaluation only)

**Per Query Typical LLM Usage:**
- **Minimum**: 1 call (fast-path: skip routing, no retry)
- **Average**: 2 calls (routing + generation)
- **Maximum**: 3 calls (routing + rewriting + generation)
- **Evaluation**: +1 call (judging, separate workflow)

**Cost-Saving Features:**
- Statistical grading (no LLM for quality checks)
- Fast-path routing (skip LLM for simple queries)
- Conditional rewriting (only on failure)
- Token limits (prevent runaway generation)
- Feature flags (disable routing/grading)

---

**Last Updated**: 2026-02-07
**Author**: Generated from codebase analysis
**Version**: 1.0.0
