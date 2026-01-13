#!/usr/bin/env python
# scripts/build_vectorstore.py
"""
Build vector store and BM25 retriever from PDF documents.

Usage:
    uv run python scripts/build_vectorstore.py

This script:
1. Loads PDF documents from data/raw/
2. Splits them into chunks
3. Creates embeddings using E5 multilingual model
4. Builds and saves Chroma vector store
5. Builds and saves BM25 retriever
"""
from pathlib import Path
import sys
import hashlib
import pickle
import json
from datetime import datetime
from functools import lru_cache

# Add src to path so we can import from it
src_path = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(src_path))

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from transformers import AutoTokenizer

from config import (
    RAW_PDF_DIR,
    VECTOR_STORE_DIR,
    CHROMA_STORE_DIR,
    TOP_K_FINAL,
    EMBEDDING_MODEL,
)
from utils import build_e5_encoder


# Load tokenizer once
@lru_cache(maxsize=1)
def get_tokenizer():
    """Get tokenizer for E5 model (cached)."""
    return AutoTokenizer.from_pretrained(EMBEDDING_MODEL)


# Utility functions for fingerprinting
def pdf_fingerprint(pdf_dir):
    """Generate fingerprint of all PDFs in directory based on name and modification time."""
    h = hashlib.md5()
    for pdf in sorted(pdf_dir.glob("*.pdf")):
        h.update(pdf.name.encode())
        h.update(str(pdf.stat().st_mtime).encode())
    return h.hexdigest()


def save_fingerprint(fingerprint, pdf_files, total_pages, total_chunks):
    """Save fingerprint and metadata to JSON file."""
    cache_path = VECTOR_STORE_DIR / ".build_cache.json"

    cache_data = {
        "fingerprint": fingerprint,
        "created_at": datetime.now().isoformat(),
        "files": [
            {
                "name": pdf.name,
                "size": pdf.stat().st_size,
                "modified": pdf.stat().st_mtime,
            }
            for pdf in sorted(pdf_files)
        ],
        "total_files": len(pdf_files),
        "total_pages": total_pages,
        "total_chunks": total_chunks,
    }

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache_data, f, indent=2, ensure_ascii=False)


def load_build_cache():
    """Load complete build cache from JSON file."""
    cache_path = VECTOR_STORE_DIR / ".build_cache.json"
    if not cache_path.exists():
        return None

    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, KeyError):
        # If cache file is corrupted, return None to trigger rebuild
        return None


# Document loading and splitting
def load_documents_from_directory(pdf_dir: Path):
    """Load all PDF documents from directory."""
    documents = []
    for pdf_path in pdf_dir.glob("*.pdf"):
        loader = PyPDFLoader(str(pdf_path))
        documents.extend(loader.load())
    return documents


def split_documents(documents):
    """
    Split documents into chunks using token-aware splitting.

    Uses E5 model's tokenizer to ensure chunks don't exceed token limits.
    Optimized for multilingual text (English, Japanese, Chinese).

    Configuration:
    - chunk_size: 400 tokens (balanced for all languages)
      * English: ~1600 characters
      * Japanese/Chinese: ~400-800 characters
    - chunk_overlap: 60 tokens (~15% overlap for context preservation)
    """
    tokenizer = get_tokenizer()

    splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer,
        chunk_size=400,  # 400 tokens (balanced for multilingual content)
        chunk_overlap=60,  # 60 tokens overlap (~15%)
        separators=[
            "\n\n\n",  # Multiple newlines (section breaks)
            "\n\n",  # Paragraph breaks
            "\n",  # Single newline
            "。",  # Japanese/Chinese period
            ". ",  # English period with space
            "！",  # Japanese/Chinese exclamation
            "! ",  # English exclamation with space
            "？",  # Japanese/Chinese question mark
            "? ",  # English question mark with space
            "、",  # Japanese comma (読点)
            "，",  # Chinese comma
            ", ",  # English comma with space
            " ",  # Space (word boundary)
            "",  # Character-level (last resort)
        ],
    )
    return splitter.split_documents(documents)


# Vector store and retriever building
def build_vector_store(chunks):
    """Build and persist Chroma vector store."""
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=build_e5_encoder(),
        persist_directory=str(CHROMA_STORE_DIR),
        collection_name="rag_demo",
    )
    return vectordb


def build_bm25_retriever(chunks):
    """Build and save BM25 retriever."""
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = TOP_K_FINAL  # Default top k

    # Save BM25 retriever
    bm25_path = VECTOR_STORE_DIR / "bm25_retriever.pkl"
    with open(bm25_path, "wb") as f:
        pickle.dump(bm25_retriever, f)

    return bm25_retriever


def main():
    """Main build process."""
    # Create necessary directories
    CHROMA_STORE_DIR.mkdir(parents=True, exist_ok=True)

    # Check if PDF directory exists
    if not RAW_PDF_DIR.exists():
        print(f"❌ Error: PDF directory not found: {RAW_PDF_DIR}")
        print(f"   Creating directory...")
        RAW_PDF_DIR.mkdir(parents=True, exist_ok=True)
        print(f"   ✅ Directory created: {RAW_PDF_DIR}")
        print(
            f"\n⚠️  No PDFs found. Please add PDF files to {RAW_PDF_DIR}/ and run this script again."
        )
        sys.exit(1)

    # Check if there are any PDF files
    pdf_files = list(RAW_PDF_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"❌ No PDF files found in {RAW_PDF_DIR}/")
        print(f"\n📝 Please add your PDF documents to this directory:")
        print(f"   {RAW_PDF_DIR.absolute()}")
        print(f"\nExample:")
        print(f"   cp /path/to/your/document.pdf {RAW_PDF_DIR}/")
        print(f"\nThen run this script again:")
        print(f"   uv run python scripts/build_vectorstore.py")
        sys.exit(1)

    print(f"✅ Found {len(pdf_files)} PDF file(s):")
    for pdf in pdf_files:
        print(f"   - {pdf.name}")
    print()

    # Check fingerprint to see if PDFs have changed
    current_fp = pdf_fingerprint(RAW_PDF_DIR)
    cache = load_build_cache()
    stored_fp = cache.get("fingerprint") if cache else None

    if stored_fp == current_fp:
        print("✅ PDFs unchanged. Using existing vector store.")
        print(f"   Vector store location: {CHROMA_STORE_DIR}")

        # Display cache information
        if cache:
            created_at = cache.get("created_at", "Unknown")
            total_chunks = cache.get("total_chunks", "Unknown")
            print(f"   Last built: {created_at}")
            print(f"   Cached chunks: {total_chunks}")

        sys.exit(0)

    print("🔨 Building vector store...")

    # 1. Load pages
    docs = load_documents_from_directory(RAW_PDF_DIR)
    print(f"📄 Total pages loaded: {len(docs)}")

    # 2. Chunk
    print("✂️  Splitting documents into chunks...")
    chunks = split_documents(docs)

    # Add chunk_id to metadata for better tracking and deduplication
    for i, chunk in enumerate(chunks):
        # Extract filename with extension
        file_name = Path(chunk.metadata.get("source", "unknown")).name
        page = chunk.metadata.get("page", 0) + 1  # Page numbers start at 1
        chunk.metadata["chunk_id"] = f"{file_name}, page {page}, chunk {i}"

    print(f"📦 Total chunks created: {len(chunks)}")

    # 3. Embed + persist
    print("🧮 Creating embeddings and building vector store...")
    print(f"   (This may take a few minutes depending on document size)")
    build_vector_store(chunks)
    print(f"   ✅ Vector store saved to: {CHROMA_STORE_DIR}")

    # 4. Build BM25 retriever
    print("🔍 Building BM25 retriever...")
    build_bm25_retriever(chunks)
    bm25_path = VECTOR_STORE_DIR / "bm25_retriever.pkl"
    print(f"   ✅ BM25 retriever saved to: {bm25_path}")

    # 5. Save fingerprint with metadata
    save_fingerprint(current_fp, pdf_files, len(docs), len(chunks))

    print("\n" + "=" * 60)
    print("✨ Success! Vector store and BM25 retriever built successfully.")
    print("=" * 60)
    print(f"\n📊 Summary:")
    print(f"   - PDF files processed: {len(pdf_files)}")
    print(f"   - Total pages: {len(docs)}")
    print(f"   - Total chunks: {len(chunks)}")
    print(f"   - Vector store: {CHROMA_STORE_DIR}")
    print(f"\n🚀 Next step: Start the server")
    print(f"   langgraph dev")
    print(f"\n   Then open: http://localhost:2024")


if __name__ == "__main__":
    main()
