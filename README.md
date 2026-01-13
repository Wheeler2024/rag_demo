# RAG Demo - PDF Question Answering System

A Retrieval-Augmented Generation (RAG) system built with LangGraph for answering questions based on PDF documents using advanced retrieval techniques.

## Features

- 🔍 **Hybrid Retrieval**: Combines vector embeddings (E5) and BM25 for better document retrieval
- 🔄 **Query Rewriting**: Automatically generates multiple search queries for improved recall
- 📊 **Reciprocal Rank Fusion (RRF)**: Merges results from multiple retrievers
- 🎯 **LLM Reranking**: Uses LLM to select the most relevant chunks
- 📝 **Citation Support**: Automatically generates inline citations with source tracking
- 💾 **Session Logging**: Saves Q&A sessions to JSON for analysis

## Architecture

```
User Question
    ↓
Query Rewriting (3 variants)
    ↓
Parallel Retrieval (Vector + BM25) × 4 queries
    ↓
RRF Fusion (top 20)
    ↓
LLM Reranking (top 5)
    ↓
Answer Generation with Citations
    ↓
Save to logs/qa_log_*.json
```

## Prerequisites

- Python 3.12+
- CUDA-capable GPU (recommended for embeddings)
- [uv](https://docs.astral.sh/uv/) package manager
- Groq API key (free tier available)

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/rag_demo.git
cd rag_demo
```

### 2. Set up environment

```bash
# Copy .env template and fill in your API keys
cp .env.example .env

# Edit .env and configure:
# - MODEL_PROVIDER (groq/openai/anthropic/google)
# - API keys for your chosen provider
```

### 3. Add PDF documents

```bash
# Place your PDF files in the data/raw/ directory
cp /path/to/your/pdfs/*.pdf data/raw/
```

### 4. Build vector store (REQUIRED before Docker)

```bash
# Install dependencies
uv sync

# Build the vector store locally
uv run python scripts/build_vectorstore.py
```

**⏱️ This step may take 2-5 minutes depending on PDF size.**

### 5. Start Docker services

```bash
# Build and start all services
docker-compose build
docker-compose up -d

# Check service health
docker-compose ps
```

### 6. Access the application

- **Agent Chat UI**: Connect at `https://agentchat.vercel.app/`
  - Deployment URL: `http://localhost:2024`
  - Graph ID: `rag_demo`
  
- **API Docs**: http://localhost:2024/docs
- **Health Check**: http://localhost:2024/ok


## Project Structure

```
rag_demo/
├── src/
│   ├── test.py           # Main RAG graph implementation
│   ├── config.py         # Configuration (models, paths)
│   ├── embedding.py      # Vector store & BM25 setup
│   └── nodes/            # Custom graph nodes
├── data/
│   ├── raw/              # Your PDF documents
│   └── vector_store/     # Generated embeddings
├── logs/                 # Q&A session logs
├── scripts/
│   └── check_cuda.py     # CUDA verification
├── .env.example          # Environment template
├── langgraph.json        # LangGraph configuration
├── pyproject.toml        # Python dependencies
└── README.md             # This file
```

## Configuration

Edit `src/config.py` to customize:

- **Models**:
  - `RESPONSE_MODEL`: LLM for answers (default: GPT-OSS-120B via Groq)
  - `RERANK_MODEL`: LLM for reranking (default: Llama-3.3-70B)
  - `EMBEDDING_MODEL`: Embedding model (default: E5-multilingual)
  
- **Retrieval**:
  - `TOP_K_FINAL`: Number of chunks after reranking (default: 5)
  - `EMBEDDING_BATCH_SIZE`: Batch size for embeddings (default: 16)

## How to Use

### Example Questions

```
Q: Who are the authors of the Attention is All You Need paper?
Q: How many attention heads does the big Transformer model use?
Q: What is the BLEU score achieved on WMT 2014 English-to-German?
```

### Session Logs

Every Q&A interaction is saved to `logs/qa_log_YYYYMMDD.jsonl`:

```json
{
  "timestamp": "2026-01-11T14:30:52",
  "question": "Original user question",
  "rewritten_queries": ["Query 1", "Query 2", "Query 3"],
  "answer": "Generated answer with citations",
  "citations": ["Source identifiers"],
  "reranked_context": [
    {
      "chunk_id": "...",
      "content": "..."
    }
  ]
}
```

## Troubleshooting

### Out of memory errors
- Reduce `EMBEDDING_BATCH_SIZE` in `config.py`
- Use CPU-only embeddings (slower but works)

### API rate limits
- Groq free tier has rate limits
- Consider upgrading or using different models

## Development

### Adding New Documents

1. Add PDFs to `data/raw/`
2. Rebuild vector store: `uv run python src/embedding.py`
3. Restart LangGraph server

## Technologies Used

- **LangGraph**: Orchestration and graph workflow
- **LangChain**: RAG components and integrations
- **Groq**: Fast LLM inference
- **ChromaDB**: Vector database
- **Sentence Transformers**: E5 multilingual embeddings
- **BM25**: Keyword-based retrieval

## License

MIT

## Contributing

Pull requests are welcome! Please open an issue first to discuss proposed changes.

## Acknowledgments

- Attention Is All You Need paper (example dataset)
- LangChain and LangGraph teams
- Groq for fast inference
