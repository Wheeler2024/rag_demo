# src/nodes/retrievers.py
from utils import load_vector_store, load_bm25_retriever
from config import TOP_K_FINAL
from state import RAGState
from langgraph.types import Send


# Node function: retrieve documents using vector embeddings
def retrieve_vector(state: RAGState):
    """Search and return information about PDFs for a single query using vector embeddings."""
    vectorstore = load_vector_store()
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K_FINAL})

    query = state["question"]
    docs = retriever.invoke(query)
    return {"docs": docs}


# Node function: retrieve documents using BM25
def retrieve_bm25(state: RAGState):
    """Search and return information about PDFs for a single query using BM25."""
    bm25_retriever = load_bm25_retriever()
    bm25_retriever.k = TOP_K_FINAL

    query = state["question"]
    docs = bm25_retriever.invoke(query)
    return {"docs": docs}


# Dispatch function: send all queries for parallel retrieval
def send_all_queries(state: RAGState):
    """Send all rewritten queries and original question for parallel retrieval (both embedding and BM25)."""

    queries = state.get("rewritten_queries", []) + [state["question"]]

    sends = []
    # For each query, send both embedding and BM25 retrieval
    for query in queries:
        sends.append(Send("retrieve_vector", {"question": query}))
        sends.append(Send("retrieve_bm25", {"question": query}))

    return sends
