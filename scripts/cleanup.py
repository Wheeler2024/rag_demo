#!/usr/bin/env python
# scripts/cleanup.py
"""
Clean up generated files and directories.

Usage:
    uv run python scripts/cleanup.py

This script removes:
- logs/ directory (Q&A session logs)
- data/vector_store/chroma/ (vector database)
- data/vector_store/bm25_retriever.pkl (BM25 retriever)
- data/vector_store/fingerprint.txt (PDF fingerprint cache)
"""
from pathlib import Path
import sys
import shutil

# Add src to path so we can import config
src_path = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(src_path))

from config import VECTOR_STORE_DIR, CHROMA_STORE_DIR

# Define paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOGS_DIR = PROJECT_ROOT / "logs"
BM25_FILE = VECTOR_STORE_DIR / "bm25_retriever.pkl"
BUILD_CACHE_FILE = (
    VECTOR_STORE_DIR / ".build_cache.json"
)  # Updated from fingerprint.txt


def remove_directory(path: Path, name: str):
    """Remove directory if it exists."""
    if path.exists() and path.is_dir():
        shutil.rmtree(path)
        print(f"✅ Removed {name}: {path}")
        return True
    else:
        print(f"⏭️  {name} not found: {path}")
        return False


def remove_file(path: Path, name: str):
    """Remove file if it exists."""
    if path.exists() and path.is_file():
        path.unlink()
        print(f"✅ Removed {name}: {path}")
        return True
    else:
        print(f"⏭️  {name} not found: {path}")
        return False


def main():
    """Main cleanup process."""
    print("=" * 60)
    print("🧹 Cleanup Script - Remove Generated Files")
    print("=" * 60)
    print("\nThis will delete:")
    print(f"  1. logs/ directory: {LOGS_DIR}")
    print(f"  2. chroma/ directory: {CHROMA_STORE_DIR}")
    print(f"  3. BM25 retriever: {BM25_FILE}")
    print(f"  4. Build cache: {BUILD_CACHE_FILE}")
    print("\n⚠️  This action cannot be undone!")

    # Ask for confirmation
    response = input("\nProceed with cleanup? (y/N): ").strip().lower()

    if response not in ["y", "yes"]:
        print("\n❌ Cleanup cancelled.")
        sys.exit(0)

    print("\n🧹 Starting cleanup...\n")

    # Track what was removed
    removed_count = 0

    # Remove logs directory
    if remove_directory(LOGS_DIR, "logs directory"):
        removed_count += 1

    # Remove chroma directory
    if remove_directory(CHROMA_STORE_DIR, "chroma vector store"):
        removed_count += 1

    # Remove BM25 retriever file
    if remove_file(BM25_FILE, "BM25 retriever"):
        removed_count += 1

    # Remove build cache file
    if remove_file(BUILD_CACHE_FILE, "build cache"):
        removed_count += 1

    # Summary
    print("\n" + "=" * 60)
    if removed_count > 0:
        print(f"✨ Cleanup complete! Removed {removed_count} item(s).")
    else:
        print("✨ Nothing to clean up - all items already removed.")
    print("=" * 60)

    if removed_count > 0:
        print("\n📝 Next steps:")
        print("  1. Add PDF files to data/raw/")
        print("  2. Rebuild vector store:")
        print("     uv run python scripts/build_vectorstore.py")
        print("  3. Start the server:")
        print("     langgraph dev")


if __name__ == "__main__":
    main()
