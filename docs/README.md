# RAGChain Documentation

## 📊 System Architecture

### Full System Architecture

![RAGChain Architecture](images/architecture.png)

**Legend:**
- 🔴 **Red Star with "LLM CALL"** = LLM invocation (4 total call points)
- ✅ **Green Checkmark "No LLM"** = Statistical/algorithmic operations
- 🔵 **Blue** = Data ingestion pipeline
- 🟢 **Green** = Search/retrieval operations  
- 🟠 **Orange** = RAG pipeline with adaptive routing
- 🟣 **Purple** = Evaluation workflow

### LLM Invocation Details

![LLM Usage Details](images/llm-usage.png)

### LLM Call Points

The system has **4 distinct LLM invocation points**:

1. **Intent Router** (`router.py`) - Classifies queries as FACT/CONCEPT/COMPARISON
   - Config: `temp=0.0, max_tokens=32, ctx=4096`
   - Fast-path: Skips LLM for simple queries

2. **Query Rewriter** (`graph.py`) - Enhances failed queries for retry (max 1×)
   - Config: `temp=0.5, max_tokens=128, ctx=4096`
   - Conditional: Only runs if grading fails

3. **Answer Generator** (`cli.py`) - Generates natural language answers
   - Config: `temp=0.1, max_tokens=1024, ctx=8192, reasoning=True`
   - Always runs: Core RAG functionality

4. **LLM-as-Judge** (`judge.py`) - Evaluates answer quality (optional)
   - Config: `temp=0.0, max_tokens=512, ctx=4096`
   - Evaluation only: Not in production path

**Per-query LLM usage:**
- Minimum: 1 call (fast-path routing + no retry)
- Average: 2 calls (routing + generation)
- Maximum: 3 calls (routing + rewriting + generation)

See [llm-architecture-flow.md](llm-architecture-flow.md) for detailed flow analysis.

---

# 🚀 How to Run This Code

📸 **See [Demo Screenshots](demo.md)** for a visual walkthrough of the complete setup and usage process.

```
1. Install Ollama (macOS example)
brew install ollama

2. Start Ollama service
ollama serve

3. Pull required models
ollama pull qwen3-embedding:4b  # For embeddings
ollama pull qwen3:8b            # For generation

4. Install Python 3.12+
brew install python@3.12

5. Install uv (fast Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh
```
# Installation
```
# Clone repository
- cd /Users/averma/github/Udemy/02_AI-ML_Courses/01_LLM/ragchain-main

# Install dependencies with uv
- uv sync

# This installs:
# - Runtime dependencies (LangChain, Chroma, etc.)
# - Dev dependencies (pytest, ruff, mypy)
# - CLI entry point: ragchain command
```
# Start Infrastructure

```
# Start Chroma vector database
- docker compose up -d

# Verify Chroma is running
- curl http://localhost:8000/api/v1/heartbeat

# Check Ollama is running
- ollama list
```
Note : Alternatively, if you don't want docker image for chromaDB, you can have local as well

```
# Just install dependencies
uv sync

# Use local file-based Chroma (default behavior)
export CHROMA_PERSIST_DIRECTORY="./chroma_data"

# Don't set CHROMA_SERVER_URL or set it to empty
export CHROMA_SERVER_URL=""

# Now you can run directly
ragchain ingest
ragchain ask "your question"
```
# Usage - All Options

**Note:** All `ragchain` commands must be prefixed with `uv run` since the package is installed in a virtual environment:

```bash
uv run ragchain <command>
```

1. Ingest Documents
```
# Basic ingestion (default settings)
uv run ragchain ingest

# What it does:
# - Fetches top 50 languages from TIOBE
# - Loads Wikipedia articles for each
# - Loads 10 conceptual bridge pages
# - Chunks text (2500 chars, 500 overlap)
# - Generates embeddings
# - Stores in Chroma

# Environment variables (optional):
export CHUNK_SIZE=3000           # Increase chunk size
export CHUNK_OVERLAP=600         # Increase overlap
export CHROMA_SERVER_URL=http://localhost:8000  # Custom Chroma

uv run ragchain ingest
```
2. Search Documents
```
# Basic search
uv run ragchain search "functional programming"

# With custom k (number of results)
uv run ragchain search "Python programming" --k 10

# What it does:
# - Runs ensemble retrieval (BM25 + Chroma)
# - Returns top-k ranked results
# - Default weights: 0.5 BM25 / 0.5 Chroma

# Environment variables:
export SEARCH_K=15  # Default k value
export RRF_MAX_RESULTS=20  # Max results after RRF

uv run ragchain search "your query"
```
3. Ask Questions (Full RAG)
```
# Basic question
uv run ragchain ask "What is Python used for?"

# With custom model
uv run ragchain ask "Compare Go and Rust" --model qwen3:8b

# Complex queries
uv run ragchain ask "What are the top 10 most popular languages?"
uv run ragchain ask "How has Java evolved since its release?"

# What it does:
# 1. Classifies intent (FACT/CONCEPT/COMPARISON)
# 2. Retrieves with adaptive weights
# 3. Grades relevance (optional)
# 4. Rewrites query if needed (max 1 retry)
# 5. Generates answer with LLM

# Environment variables:
export ENABLE_INTENT_ROUTING=true   # Enable/disable routing
export ENABLE_GRADING=true          # Enable/disable grading
export GRAPH_K=8                    # Documents for RAG
export OLLAMA_GEN_CTX=12288         # Larger context window

uv run ragchain ask "your question"
```

4. Evaluate RAG Quality
```
# Basic evaluation (default model)
uv run ragchain evaluate

# With custom model
uv run ragchain evaluate --model qwen3:8b

# What it does:
# - Runs 20 diverse test questions
# - Generates answers via RAG
# - Evaluates with LLM-as-judge
# - Scores: correctness, relevance, faithfulness
# - Displays averages

# Environment variables:
export OLLAMA_JUDGING_CTX=6144  # Larger context for judging

uv run ragchain evaluate
```
# Configuration Options
All configuration is via environment variables (see config.py):

```
# Vector Store
export CHROMA_PERSIST_DIRECTORY="./chroma_data"  # Local storage
export CHROMA_SERVER_URL="http://localhost:8000"  # Remote Chroma

# Ollama Models
export OLLAMA_BASE_URL="http://localhost:11434"
export OLLAMA_EMBED_MODEL="qwen3-embedding:4b"
export OLLAMA_MODEL="qwen3:8b"
export OLLAMA_EMBED_CTX=4096      # Embedding context
export OLLAMA_GEN_CTX=8192        # Generation context
export OLLAMA_ROUTING_CTX=2048    # Routing context
export OLLAMA_JUDGING_CTX=4096    # Judging context
export OLLAMA_REWRITING_CTX=2048  # Rewriting context

# Document Processing
export CHUNK_SIZE=2500
export CHUNK_OVERLAP=500

# Retrieval
export SEARCH_K=10          # Search API k
export GRAPH_K=6            # RAG pipeline k
export RRF_MAX_RESULTS=10   # Max RRF results

# Feature Flags
export ENABLE_GRADING=true          # Document grading
export ENABLE_INTENT_ROUTING=true   # Intent classification

```

Testing

```
# Unit tests (mocked dependencies)
uv run pytest -q

# Integration tests (requires Ollama + Chroma)
docker compose up -d
CHROMA_SERVER_URL= uv run pytest -m integration

# Run specific test file
uv run pytest tests/unit/test_graph.py -v

# Run with coverage
uv run pytest --cov=ragchain --cov-report=html
```

Development

```
# Linting (Ruff)
uv run ruff check src/

# Formatting
uv run ruff format src/

# Type checking (mypy)
uv run mypy src/

# Auto-fix linting issues
uv run ruff check --fix src/
```
# 🎯 Key Design Patterns

- Adaptive Retrieval: Intent-based weight adjustment for BM25/Chroma
- Self-Correction: Automatic query rewriting on validation failure
- Ensemble Search: RRF combines keyword + semantic search
- LangGraph Orchestration: State-based workflow with conditional routing
- Singleton Config: Centralized configuration management
- Prompt Templates: Reusable LLM prompts for different tasks
- Evaluation as Code: Automated LLM-as-judge testing

# 📊 Data Flow Summary
```
Input → Intent Classification → Weighted Retrieval → Grading → 
(Rewrite if needed) → Answer Generation → Output
```

# 📖 Additional Documentation

- **[Why RAGChain is Agentic RAG](agentic-rag.md)** - Comprehensive analysis of why this system implements Agentic RAG, not traditional RAG, with evidence from code and detailed agent behavior
- **[Demo Screenshots](demo.md)** - Visual walkthrough of setup and usage
- **[LLM Architecture & Flow](llm-usage.md)** - Detailed analysis of all 4 LLM invocation points with configuration details
- **[Advanced RAG Strategies](advance-rag.md)** - Guide to advanced RAG techniques implemented in this codebase
- **[Architecture](architecture.md)** - Detailed system architecture and design decisions
