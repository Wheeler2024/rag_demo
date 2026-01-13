#!/bin/bash
set -e

echo "🚀 Starting RAG Demo Application..."

# Quick validation: check if vector store exists
VECTOR_STORE_PATH="/deps/rag_demo/data/vector_store/chroma"
BM25_PATH="/deps/rag_demo/data/vector_store/bm25_retriever.pkl"

if [ ! -d "$VECTOR_STORE_PATH" ] || [ ! -f "$BM25_PATH" ]; then
    echo "❌ ERROR: Vector store not found!"
    echo ""
    echo "📝 Please build the vector store first:"
    echo "   1. Add PDF files to data/raw/"
    echo "   2. Run: uv run python scripts/build_vectorstore.py"
    echo "   3. Then restart: docker-compose up -d"
    echo ""
    exit 1
fi

echo "✅ Vector store found."
echo ""
echo "🌟 Starting Warming up..."
echo ""

# Execute the command passed to the container
exec "$@"
