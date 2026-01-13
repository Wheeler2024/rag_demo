# src/nodes/rewriter.py
from utils import get_response_model
from langchain_core.messages import HumanMessage
from state import RAGState


# Prompt template for query rewriting
QUERY_REWRITE_PROMPT = """
You are a search query optimization expert for a Retrieval-Augmented Generation (RAG) system.

Your task is to rewrite the user's question into multiple high-quality search queries
to improve document retrieval from a vector database.

CRITICAL CONSTRAINTS:
- Generate EXACTLY 3 rewritten queries
- EVERY rewritten query MUST preserve the core entity, paper title, or subject explicitly mentioned in the original question
- Do NOT generalize to related topics or broader fields
- Do NOT answer the question
- Do NOT include explanations

Rewrite strategy:
1. One precise factual query explicitly targeting the requested information
2. One semantically complete query phrased as a natural question
3. One alternative phrasing using different wording while preserving the same meaning

User Question:
{question}

Output Format (JSON):
{{
  "queries": [
    "rewritten query 1",
    "rewritten query 2",
    "rewritten query 3"
  ]
}}

Output valid JSON only. Do not include any additional text.
"""


# Node function: rewrite user query into multiple optimized search queries
def rewrite_query(state: RAGState):
    """Extract question from messages and rewrite into multiple optimized search queries."""
    import json

    # Extract question from last message
    question = state["question"]

    prompt = QUERY_REWRITE_PROMPT.format(question=question)

    # Use response_format instead of with_structured_output to preserve thinking process
    response = (
        get_response_model()
        .bind(response_format={"type": "json_object"})
        .invoke([HumanMessage(content=prompt)])
    )

    # Parse JSON from response content
    result = json.loads(response.content)
    queries = result.get("queries", [])

    # Clear previous docs by returning None for docs
    return {"question": question, "rewritten_queries": queries, "docs": None}
