# src/config.py
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

# Define project directories
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_PDF_DIR = DATA_DIR / "raw"
VECTOR_STORE_DIR = DATA_DIR / "vector_store"
CHROMA_STORE_DIR = VECTOR_STORE_DIR / "chroma"


# Model Provider Configuration
MODEL_PROVIDER = os.getenv(
    "MODEL_PROVIDER", "groq"
).lower()  # groq, openai, anthropic, google

# Define model names (provider-specific)
if MODEL_PROVIDER == "groq":
    RESPONSE_MODEL = os.getenv("RESPONSE_MODEL", "openai/gpt-oss-120b")
    RERANK_MODEL = os.getenv(
        "RERANK_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct"
    )
elif MODEL_PROVIDER == "openai":
    RESPONSE_MODEL = os.getenv("RESPONSE_MODEL", "gpt-4o")
    RERANK_MODEL = os.getenv("RERANK_MODEL", "gpt-4o-mini")
elif MODEL_PROVIDER == "anthropic":
    RESPONSE_MODEL = os.getenv("RESPONSE_MODEL", "claude-3-5-sonnet-20241022")
    RERANK_MODEL = os.getenv("RERANK_MODEL", "claude-3-5-haiku-20241022")
elif MODEL_PROVIDER == "google":
    RESPONSE_MODEL = os.getenv("RESPONSE_MODEL", "gemini-2.0-flash-exp")
    RERANK_MODEL = os.getenv("RERANK_MODEL", "gemini-2.0-flash-exp")
else:
    raise ValueError(f"Unsupported MODEL_PROVIDER: {MODEL_PROVIDER}")

# Embedding model configuration
EMBEDDING_MODEL = "intfloat/multilingual-e5-base"
EMBEDDING_BATCH_SIZE = 16

# Retrieval configuration
TOP_K_FUSION = 15  # After RRF fusion (send to rerank)
TOP_K_FINAL = 5  # After reranking (final answer generation)
