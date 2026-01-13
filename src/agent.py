# src/agent.py
from state import RAGState
from nodes import (
    check_cache,
    route_from_cache,
    rewrite_query,
    retrieve_vector,
    retrieve_bm25,
    send_all_queries,
    fuse_docs,
    rerank_docs,
    generate_answer,
)
from langgraph.graph import StateGraph, START, END

# Build the agent workflow
workflow = StateGraph(RAGState)

workflow.add_node("check_database", check_cache)
workflow.add_node("rewrite_query", rewrite_query)
workflow.add_node("retrieve_vector", retrieve_vector)
workflow.add_node("retrieve_bm25", retrieve_bm25)
workflow.add_node("fuse_docs", fuse_docs)
workflow.add_node("rerank", rerank_docs)
workflow.add_node("generate_answer", generate_answer)

# Add cache layer at the beginning
workflow.add_edge(START, "check_database")
workflow.add_conditional_edges(
    "check_database",
    route_from_cache,
    {
        "__end__": END,
        "rewrite_query": "rewrite_query",
    },
)
workflow.add_conditional_edges(
    "rewrite_query", send_all_queries, ["retrieve_vector", "retrieve_bm25"]
)
workflow.add_edge("retrieve_vector", "fuse_docs")
workflow.add_edge("retrieve_bm25", "fuse_docs")
workflow.add_edge("fuse_docs", "rerank")
workflow.add_edge("rerank", "generate_answer")
workflow.add_edge("generate_answer", END)

agent = workflow.compile()
