# src/nodes/reranker.py
from typing import List
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage
from config import TOP_K_FINAL
from state import RAGState
from utils import get_rerank_model


# Prompt template for reranking
RERANK_PROMPT = """
You are a relevance evaluator. Given a question and document chunks, identify the TOP {top_k} most relevant chunks to answer the question.

Question: {question}

Available Chunks:
{chunks}

Instructions:
1. Focus on chunks with specific facts, numbers, or direct answers.
2. Prioritize chunks that contain the exact information requested.
3. For list-based questions (authors, references, steps), prioritize chunks with complete lists over partial mentions.
4. Avoid chunks with only background or loosely related content.
5. Return up to {top_k} chunk IDs, ordered by relevance (most relevant first).

Output valid JSON only. Do not include any additional text.
"""


# Pydantic schema for reranking results
class RerankedState(BaseModel):
    """Schema for reranking results."""

    selected_ids: List[int] = Field(
        ...,
        description="List of selected chunk IDs in order of relevance (most relevant first)",
    )


# Node function: rerank documents using LLM
def rerank_docs(state: RAGState):
    """Rerank retrieved documents using LLM to select most relevant ones."""

    question = state["question"]
    docs = state["context"]

    # Format chunks with IDs for the LLM (dynamic truncation based on content type)
    chunks_text = ""

    # Keywords that indicate structured/important content
    TABLE_KEYWORDS = ["table", "figure", "parameter", "coefficient", "matrix", "value"]
    LIST_KEYWORDS = ["author", "reference", "step", "equation", "formula"]
    CODE_KEYWORDS = ["def ", "class ", "function", "import", "return"]

    for i, doc in enumerate(docs):
        # Clean content
        text = doc.page_content.replace("\n", " ")
        text_lower = text.lower()

        # Determine content type and set appropriate char limit
        if any(keyword in text_lower for keyword in TABLE_KEYWORDS):
            # Tables/parameters: preserve more content (they're dense with info)
            max_chars = 1200
            head_size = 600
            tail_size = 600
        elif any(keyword in text_lower for keyword in LIST_KEYWORDS):
            # Lists: preserve full content when possible (need completeness)
            max_chars = 1000
            head_size = 500
            tail_size = 500
        elif any(keyword in text_lower for keyword in CODE_KEYWORDS):
            # Code: preserve more (syntax matters)
            max_chars = 1000
            head_size = 500
            tail_size = 500
        elif len(text) < 600:
            # Short chunks: don't truncate at all
            max_chars = float("inf")
            head_size = 0
            tail_size = 0
        else:
            # Regular text: standard truncation
            max_chars = 800
            head_size = 400
            tail_size = 400

        # Apply truncation if needed
        if len(text) > max_chars:
            text = text[:head_size] + " ... " + text[-tail_size:]

        chunks_text += f"[{i}] {text}\n\n"

    prompt = RERANK_PROMPT.format(
        top_k=TOP_K_FINAL, question=question, chunks=chunks_text
    )

    # Use RERANK_MODEL with structured output
    try:
        response = (
            get_rerank_model()
            .with_structured_output(RerankedState)
            .invoke([HumanMessage(content=prompt)])
        )

        # Get selected IDs from structured output
        selected_ids = response.selected_ids[:TOP_K_FINAL]
        # Select corresponding documents
        reranked_docs = [docs[i] for i in selected_ids if i < len(docs)]
        # Ensure we have at least some documents
        if not reranked_docs:
            reranked_docs = docs[:TOP_K_FINAL]

    except Exception as e:
        # Fallback: if reranking fails, just take first TOP_K_FINAL
        print(f"Reranking failed: {e}, using fallback")
        reranked_docs = docs[:TOP_K_FINAL]

    return {"reranked_context": reranked_docs}
