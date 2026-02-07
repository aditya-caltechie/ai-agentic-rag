
RAGChain is an Intent-Based Adaptive RAG (Retrieval-Augmented Generation) system that uses LangGraph for orchestration. Here's the complete architecture:

```
┌─────────────────────────────────────────────────────────────────────┐
│                         RAGChain System                             │
│                                                                     │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐         │
│  │   CLI Layer  │────▶│ LangGraph    │────▶│  LLM Layer   │         │
│  │  (Click)     │     │ Orchestrator │     │  (Ollama)    │         │
│  └──────────────┘     └──────────────┘     └──────────────┘         │
│         │                     │                     │               │
│         │                     ▼                     ▼               │
│         │            ┌──────────────┐     ┌──────────────┐          │
│         │            │  Retrieval   │────▶│  Embeddings  │          │
│         │            │  Pipeline    │     │  (Ollama)    │          │
│         │            └──────────────┘     └──────────────┘          │
│         │                     │                     │               │
│         ▼                     ▼                     ▼               │
│  ┌──────────────────────────────────────────────────────┐           │
│  │              Storage & Vector Layer                  │           │
│  │  ┌──────────────┐           ┌──────────────┐         │           │
│  │  │ Chroma DB    │           │    BM25      │         │           │
│  │  │ (Semantic)   │◀─── RRF ─▶│  (Keyword)   │         │           │
│  │  └──────────────┘           └──────────────┘         │           │
│  └──────────────────────────────────────────────────────┘           │
└─────────────────────────────────────────────────────────────────────┘
```

# 📦 Component Breakdown

## 1. Configuration Layer (config.py)
Purpose: Centralized singleton configuration management
Key Settings:
Vector store: Chroma (local or remote HTTP)
Ollama models: qwen3-embedding:4b (embeddings), qwen3:8b (generation)
Context windows: 4096 (embedding), 8192 (generation)
Retrieval parameters: k values, RRF settings
Feature flags: grading, intent routing

## 2. CLI Layer (cli.py)
Four main commands:
ingest: Load documents into vector store
search: Direct semantic search
ask: Full RAG pipeline with LLM answer generation
evaluate: LLM-as-judge evaluation framework

## 3. Ingestion Pipeline (ingestion/)
```
┌─────────────────────────────────────────────────────────┐
│              Document Ingestion Workflow                │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
              ┌───────────────────┐
              │  Data Sources     │
              │  - Wikipedia API  │
              │  - TIOBE Index    │
              │  - Conceptual     │
              └────────┬──────────┘
                       │
                       ▼
              ┌───────────────────┐
              │  Document Loader  │
              │  (loaders.py)     │
              └────────┬──────────┘
                       │
                       ▼
              ┌───────────────────┐
              │  Text Splitter    │
              │  Chunk: 2500 ch   │
              │  Overlap: 500 ch  │
              └────────┬──────────┘
                       │
                       ▼
              ┌───────────────────┐
              │  Embeddings       │
              │  qwen3-embed:4b   │
              │  (1024 dims)      │
              └────────┬──────────┘
                       │
                       ▼
              ┌───────────────────┐
              │  Vector Store     │
              │  (Chroma DB)      │
              └───────────────────┘
```
Key Files:
loaders.py: Fetches Wikipedia articles, TIOBE rankings, conceptual pages
storage.py: Manages vector store, embeddings, and document ingestion

## 4. Retrieval Pipeline (inference/)
The core innovation is Ensemble Retrieval with Reciprocal Rank Fusion (RRF):
```
┌─────────────────────────────────────────────────────────┐
│           Ensemble Retrieval Architecture               │
└─────────────────────────────────────────────────────────┘

    User Query
        │
        ├──────────────────┬──────────────────┐
        │                  │                  │
        ▼                  ▼                  ▼
   ┌─────────┐      ┌─────────┐      ┌─────────┐
   │  BM25   │      │ Chroma  │      │ Intent  │
   │ Keyword │      │Semantic │      │ Router  │
   │ Search  │      │ Search  │      │         │
   └────┬────┘      └────┬────┘      └────┬────┘
        │                │                 │
        │                │                 │
        └────────┬───────┘                 │
                 │                         │
                 ▼                         │
        ┌──────────────────┐               │
        │  RRF Fusion      │◀─────────────-┘
        │  score = 1/(r+60)│  Weight Adjustment
        └────────┬─────────┘
                 │
                 ▼
        ┌──────────────────┐
        │ Ranked Results   │
        └──────────────────┘
```

Key Features:

- BM25 Retriever: Keyword-based ranking (traditional IR)
- Chroma Retriever: Semantic vector similarity
- RRF Algorithm: Combines rankings with formula score = 1/(rank + 60)
- Intent-Based Weights:
  - FACT: 0.8 BM25 / 0.2 Chroma (keyword-heavy)
  - CONCEPT: 0.4 BM25 / 0.6 Chroma (balanced)
  - COMPARISON: 0.5 BM25 / 0.5 Chroma (semantic-leaning)
 
5. LangGraph RAG Orchestrator (inference/graph.py)
The heart of the system - a self-correcting agentic RAG workflow:

```
┌──────────────────────────────────────────────────────────────┐
│              LangGraph RAG Workflow                          │
└──────────────────────────────────────────────────────────────┘

                    START
                      │
                      ▼
            ┌─────────────────┐
            │ Intent Router   │
            │ Classify Query  │
            │ FACT/CONCEPT/   │
            │ COMPARISON      │
            └────────┬────────┘
                     │
                     ▼
            ┌─────────────────┐
            │ Adaptive        │
            │ Retriever       │
            │ (Weighted RRF)  │
            └────────┬────────┘
                     │
                     ▼
            ┌─────────────────┐
            │ Retrieval       │
            │ Grader          │
            │ (LLM validates) │
            └────────┬────────┘
                     │
                     ├─── Grade = YES ───▶ END (Success)
                     │
                     ├─── Grade = NO & retry_count = 0
                     │
                     ▼
            ┌─────────────────┐
            │ Query Rewriter  │
            │ Enhance query   │
            └────────┬────────┘
                     │
                     └────▶ Loop back to Adaptive Retriever
                            (Max 1 retry)
```

State Management (IntentRoutingState):

- query: Current query (may be rewritten)
- original_query: Original user query
- intent: FACT/CONCEPT/COMPARISON
- retrieved_docs: Retrieved documents
- retrieval_grade: YES/NO validation
- rewritten_query: Enhanced query if needed
- retry_count: Number of retry attempts (max 1)

## 6. Evaluation Framework (evaluation/judge.py)
```
┌──────────────────────────────────────────────────┐
│          LLM-as-Judge Evaluation                 │
└──────────────────────────────────────────────────┘

   Question ──┐
              │
   Context ───┼──▶ Judge Prompt ──▶ LLM ──▶ Scores
              │                              (JSON)
   Answer  -──┘

   Metrics:
   - Correctness (1-5): Answer accuracy
   - Relevance (1-5): Query alignment
   - Faithfulness (1-5): Context grounding
```

## 🔄 Complete Workflow

### Workflow 1: Document Ingestion (ragchain ingest)
```
1. Fetch Data Sources
   ├─ TIOBE Index → Top 50 languages
   ├─ Wikipedia API → Language articles
   └─ Conceptual Pages → Bridge topics

2. Process Documents
   ├─ Parse HTML/Text
   ├─ Chunk (2500 chars, 500 overlap)
   └─ Add metadata (title, source, etc.)

3. Generate Embeddings
   ├─ Use qwen3-embedding:4b
   └─ 1024-dimensional vectors

4. Store in Chroma
   ├─ Upsert to vector store
   └─ Index for semantic search
```
### Workflow 2: Direct Search (ragchain search)
```
User Query
    │
    ▼
Ensemble Retriever
    ├─ BM25 Retrieval
    ├─ Chroma Retrieval
    └─ RRF Fusion
    │
    ▼
Return Top-K Results
```
### Workflow 3: RAG Answer Generation (ragchain ask)
```
1. User Query
   │
   ▼
2. Intent Classification
   ├─ FACT → Keyword-heavy
   ├─ CONCEPT → Balanced
   └─ COMPARISON → Semantic-heavy
   │
   ▼
3. Adaptive Retrieval
   ├─ Apply intent-specific weights
   └─ Retrieve documents
   │
   ▼
4. Relevance Grading
   ├─ LLM validates relevance
   └─ Decision: YES/NO
   │
   ├─── YES ────▶ 5. Generate Answer
   │                 ├─ Build context
   │                 ├─ Apply RAG template
   │                 └─ LLM generates answer
   │
   └─── NO ─────▶ 4a. Query Rewriting
                     ├─ Enhance with keywords
                     └─ Retry retrieval (once)
```
### Workflow 4: Evaluation (ragchain evaluate)

```
1. Load Test Questions (20 diverse queries)
   │
   ▼
2. For each question:
   ├─ Run full RAG pipeline
   ├─ Generate answer
   └─ Collect context
   │
   ▼
3. LLM-as-Judge Evaluation
   ├─ Score correctness (1-5)
   ├─ Score relevance (1-5)
   └─ Score faithfulness (1-5)
   │
   ▼
4. Aggregate Results
   └─ Display averages
```

