# Agentic-RAG

[![CI](https://github.com/aditya-caltechie/ai-agentic-rag/workflows/CI/badge.svg)](https://github.com/aditya-caltechie/ai-agentic-rag/actions)

Your local Agentic RAG stack — no APIs, no cloud, full control.

**Main focus :**
- Understand and build traditional RAG pipeline using chromaDB and LLM (Ollama platform).
- Understand and perform EVALs.
- Apply Advance RAG techniques to make solution agnetic (use of tools or LangGraph, query rewrite, re-ranking etc).

**Key Features:**
- Intent-based adaptive RAG with self-correcting retrieval (auto-retry on validation failure)
- Ensemble search via Reciprocal Rank Fusion combining BM25 + semantic vectors
- Local-only: Ollama for embeddings/LLM, Chroma for vector store, no external APIs
- Analyze programming languages via Docker Compose demo stack

## Quick start

📸 **[See Demo Screenshots](docs/demo.md)** - Visual walkthrough of setup and usage  
🏗️ **[System Architecture](docs/architecture.md)** - Detailed architecture and design decisions  
🔄 **[Query Flow & LLM Usage](docs/llm-usage.md)** - Step-by-step RAG pipeline execution  
🤖 **[Agentic RAG](docs/agentic-rag.md)** - Self-correcting retrieval and adaptive routing

```bash
# Install all dependencies using pyproject.toml 
uv sync 

# 1. Start Chroma vector database (optional - defaults to local storage)
# Optionally you can run local chroma db without docker.
docker compose up -d

# 2. Ingest programming language and conceptual documents
uv run ragchain ingest

# 3. Search ingested documents
uv run ragchain search "functional programming paradigm" --k 4
uv run ragchain search "memory management" --k 5

# 4. Ask questions with RAG + LLM
uv run ragchain ask "What is Python used for?"
uv run ragchain ask "Compare Go and Rust for systems programming"
uv run ragchain ask "What are the top 10 most popular languages?"

# 5. Evaluate RAG quality with LLM-as-judge
uv run ragchain evaluate

# Clean up
docker compose down -v
```

**Requirements:**
- Docker (for Chroma)
- Ollama (local LLMs and embeddings)
- Python 3.12+

## Architecture

![RAGChain Architecture](docs/images/architecture.png)

The system uses **adaptive retrieval** with 4 LLM call points:
1. **Intent Router** - Query classification (FACT/CONCEPT/COMPARISON)
2. **Query Rewriter** - Self-correction on retrieval failure (max 1 retry)
3. **Answer Generator** - Natural language synthesis from context
4. **LLM-as-Judge** - Answer quality evaluation (optional)

See [docs/llm-architecture-flow.md](docs/llm-architecture-flow.md) for detailed flow and [AGENTS.md](AGENTS.md) for full architecture.

