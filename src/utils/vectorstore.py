# src/utils/vectorstore.py
"""
Utility functions to load pre-built vector store and BM25 retriever.
Run scripts/build_vectorstore.py first to build the stores.
"""
from functools import lru_cache
from sentence_transformers import SentenceTransformer
from langchain_chroma import Chroma
from langchain.embeddings.base import Embeddings
import pickle

from config import (
    EMBEDDING_MODEL,
    EMBEDDING_BATCH_SIZE,
    CHROMA_STORE_DIR,
    VECTOR_STORE_DIR,
)


@lru_cache(maxsize=1)
def build_e5_encoder():
    """Build E5 multilingual embedding encoder (cached)."""
    # Load model with explicit CPU device - most stable approach
    import torch

    model = SentenceTransformer(
        EMBEDDING_MODEL,
        device="cpu" if not torch.cuda.is_available() else "cuda"
    )

    class E5MultilingualEmbeddings(Embeddings):
        def embed_documents(self, texts):
            texts = [f"passage: {t}" for t in texts if t.strip()]
            return model.encode(
                texts,
                batch_size=EMBEDDING_BATCH_SIZE,
                normalize_embeddings=True,
            ).tolist()

        def embed_query(self, text):
            return model.encode(
                f"query: {text}",
                normalize_embeddings=True,
            ).tolist()

    return E5MultilingualEmbeddings()


@lru_cache(maxsize=1)
def load_vector_store():
    """Load pre-built vector store from disk (cached)."""
    return Chroma(
        persist_directory=str(CHROMA_STORE_DIR),
        embedding_function=build_e5_encoder(),
        collection_name="rag_demo",
    )


@lru_cache(maxsize=1)
def load_bm25_retriever():
    """Load BM25 retriever from saved file (cached)."""
    bm25_path = VECTOR_STORE_DIR / "bm25_retriever.pkl"

    if not bm25_path.exists():
        raise FileNotFoundError(
            f"BM25 retriever not found at {bm25_path}.\n"
            f"Please run: uv run python scripts/build_vectorstore.py"
        )

    with open(bm25_path, "rb") as f:
        retriever = pickle.load(f)

    print(f"BM25 retriever loaded from {bm25_path}")
    return retriever
