# src/state.py
from typing import Annotated, List
from langgraph.graph import MessagesState
from langchain_core.documents import Document


def docs_reducer(left: list, right: list | None) -> list:
    """Custom reducer: if right is None, clear the list; otherwise concatenate."""
    if right is None:
        return []
    return left + right


# RAGState inherits MessagesState (which includes messages with add_messages reducer)
class RAGState(MessagesState):
    question: str  # Extracted from last message (internal field)
    rewritten_queries: List[str]
    docs: Annotated[List[Document], docs_reducer]  # Custom reducer: supports clearing
    context: List[Document]  # Final: deduplicated docs
    reranked_context: List[Document]  # Reranked top docs
    cache_hit: bool  # Flag to indicate cache hit
