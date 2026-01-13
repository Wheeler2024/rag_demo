# src/nodes/fusion.py
from collections import defaultdict
from config import TOP_K_FINAL, TOP_K_FUSION
from state import RAGState


# Node function: fuse documents using RRF (Reciprocal Rank Fusion)
def fuse_docs(state: RAGState, k: int = 60):
    """Fuse and deduplicate documents from all retrieved results using RRF (Reciprocal Rank Fusion)."""

    rrf_scores = defaultdict(float)
    doc_map = {}
    all_docs = state.get("docs", [])

    # Calculate rank from position (assuming each query returns TOP_K_FINAL docs)
    for idx, doc in enumerate(all_docs):
        # Use chunk_id if available, fallback to hash
        doc_id = doc.metadata.get("chunk_id", hash(doc.page_content))
        # Infer rank from position: rank = (idx % TOP_K_FINAL) + 1
        rank = (idx % TOP_K_FINAL) + 1
        # RRF formula: score = 1 / (k + rank)
        rrf_scores[doc_id] += 1.0 / (k + rank)
        if doc_id not in doc_map:
            doc_map[doc_id] = doc

    # Sort by RRF score (highest first)
    context = sorted(
        doc_map.values(),
        key=lambda d: rrf_scores[d.metadata.get("chunk_id", hash(d.page_content))],
        reverse=True,
    )

    return {"context": context[:TOP_K_FUSION]}  # Keep top 12 for reranking
