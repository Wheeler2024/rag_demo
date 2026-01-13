# src/nodes/generator.py
from langchain_core.messages import HumanMessage, AIMessage
from state import RAGState
from utils import get_response_model
import json
from datetime import datetime
from pathlib import Path
import logging


# Setup structured logging (JSON format, daily files)
def setup_logger(name: str, log_prefix: str, level=logging.INFO):
    """Setup logger with JSON formatting and daily file naming."""
    # Use date-based filename
    today = datetime.now().strftime("%Y%m%d")
    log_file = f"{log_prefix}_{today}.jsonl"

    logger = logging.getLogger(f"{name}_{today}")  # Unique logger per day
    logger.setLevel(level)

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    # Create logs directory
    log_dir = Path(__file__).resolve().parents[2] / "logs"
    log_dir.mkdir(exist_ok=True)

    # FileHandler: one file per day
    handler = logging.FileHandler(
        filename=log_dir / log_file,
        mode="a",
        encoding="utf-8",
    )

    # JSON formatter with readable field order
    class JsonFormatter(logging.Formatter):
        def format(self, record):
            # Build log entry with clean field order
            log_data = {"timestamp": datetime.fromtimestamp(record.created).isoformat()}

            # Add extra fields if present (Q&A data)
            if hasattr(record, "extra"):
                log_data.update(record.extra)

            # Add message last (usually empty for structured logs)
            if record.getMessage():
                log_data["message"] = record.getMessage()

            # Compact JSON (JSONL standard), but with readable field order
            return json.dumps(log_data, ensure_ascii=False)

    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)

    return logger


# Initialize loggers (recreated daily)
qa_logger = setup_logger("qa_logger", "qa_log", level=logging.INFO)
error_logger = setup_logger("error_logger", "error_log", level=logging.ERROR)


# Prompt template for answer generation
GENERATE_PROMPT = """
You are answering questions using ONLY the retrieved context below.

Your task:
1) Write a concise factual answer (3-5 sentences).
2) Do NOT include source identifiers in the answer text.
3) Collect all cited sources into a separate citations list.

Fact & Listing Rules:
- Each DISTINCT FACT or FACT GROUP must be supported by a citation.
- A FACT GROUP is a set of related facts that:
  • Are logically inseparable, AND
  • Come from the SAME source.
- When listing multiple items (e.g., authors, parameters, components) from the same source, You MUST group them.
- Do NOT repeat the same citation for every item if they share the same source.

Citation Rules:
- Copy the EXACT identifier from [Source: ...] shown in the context.
- Each source identifier may appear at most once in the citations list.
- Do NOT invent or paraphrase source identifiers.
- Do NOT use outside knowledge or assumptions.

Special cases:
- If the context lacks the answer: answer = "There is no relevant content in the PDFs you uploaded." and citations = []
- If calculation needed: Cite both the formula source and the parameter source.

Question: {question}

Context:
{context}

Output Format (JSON):
{{
  "answer": "Your concise answer here (3-5 sentences)",
  "citations": [
    "source identifier 1",
    "source identifier 2"
  ]
}}

Output valid JSON only. Do not include any additional text.
"""


# Node function: generate answer with inline citations
def generate_answer(state: RAGState):
    """Generate answer with inline citations in one step."""

    question = state["question"]
    docs = state["reranked_context"]

    # Handle edge case: no documents
    if not docs:
        no_answer_msg = AIMessage(content="I don't know.")
        return {"messages": [no_answer_msg]}

    # Format context for the LLM (do NOT truncate)
    numbered_context = ""
    for doc in docs:
        source = doc.metadata.get("chunk_id", "Unknown")
        numbered_context += f"[Source: {source}]\n{doc.page_content.strip()}\n\n---\n\n"

    prompt = GENERATE_PROMPT.format(question=question, context=numbered_context)

    try:
        # Use response_format to preserve thinking process
        response = (
            get_response_model()
            .bind(response_format={"type": "json_object"})
            .invoke([HumanMessage(content=prompt)])
        )

        # Parse JSON from response content
        result = json.loads(response.content)
        answer = result.get("answer", "")
        citations = result.get("citations", [])

        # Format answer with citations for display
        answer_text = answer
        if citations:
            answer_text += "\n\nSources:\n" + "\n".join(f"{cite}" for cite in citations)

        # Create AI message and append to messages
        ai_message = AIMessage(content=answer_text)

        # Log Q&A session using structured logging
        qa_logger.info(
            "Q&A session completed",
            extra={
                "extra": {
                    "question": question,
                    "rewritten_queries": state.get("rewritten_queries", []),
                    "answer": answer,
                    "citations": citations,
                    "reranked_context": [
                        {
                            "chunk_id": doc.metadata.get("chunk_id", "Unknown"),
                            "content": doc.page_content,
                        }
                        for doc in docs
                    ],
                }
            },
        )

        return {"messages": [ai_message]}

    except Exception as e:
        # Log error using structured logging
        error_logger.error(
            f"Answer generation failed: {e}",
            extra={
                "extra": {
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "question": question,
                }
            },
            exc_info=True,  # Include traceback
        )

        # Fallback: return simple error message
        error_msg = "I don't know. (Error during answer generation)"
        ai_message = AIMessage(content=error_msg)

        return {"messages": [ai_message]}
