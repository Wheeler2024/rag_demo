# src/nodes/cache.py
import json
from pathlib import Path
from langchain_core.messages import AIMessage
from state import RAGState


def check_cache(state: RAGState):
    """Check if question exists in historical logs and return cached answer if found."""

    # Extract question from last message
    question = (
        state["messages"][-1].content[0]["text"]
        if isinstance(state["messages"][-1].content, list)
        else state["messages"][-1].content
    )

    # Search through all qa_log files
    log_dir = Path(__file__).resolve().parents[2] / "logs"

    if not log_dir.exists():
        # No logs yet, continue to normal flow
        return {"question": question}

    # Search all qa_log files (sorted by date, newest first)
    qa_log_files = sorted(log_dir.glob("qa_log_*.jsonl"), reverse=True)

    for log_file in qa_log_files:
        try:
            with open(log_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        log_entry = json.loads(line.strip())

                        # Extract question and answer from log
                        logged_question = log_entry.get("question", "")

                        # Exact match (case-insensitive, stripped)
                        if logged_question.strip().lower() == question.strip().lower():
                            # Cache hit! Return cached answer
                            cached_answer = log_entry.get("answer", "")
                            cached_citations = log_entry.get("citations", [])

                            # Format answer with citations
                            answer_text = cached_answer
                            if cached_citations:
                                answer_text += "\n\nSources:\n" + ";\n".join(
                                    f"{cite}" for cite in cached_citations
                                )

                            # Add cache indicator
                            answer_text = "[Cached Response]\n\n" + answer_text

                            ai_message = AIMessage(content=answer_text)

                            print(f"✓ Cache hit for question: {question[:50]}...")

                            return {
                                "question": question,
                                "messages": [ai_message],
                                "cache_hit": True,  # Flag to skip rest of workflow
                            }

                    except json.JSONDecodeError:
                        continue  # Skip malformed lines

        except Exception as e:
            print(f"Warning: Could not read log file {log_file}: {e}")
            continue

    # Cache miss, continue to normal workflow
    print(f"✗ Cache miss for question: {question[:50]}...")
    return {"question": question, "cache_hit": False}


# Router function: skip workflow if cache hit
def route_from_cache(state: RAGState):
    """Route to __end__ if cache hit, otherwise continue to rewrite_query."""
    if state.get("cache_hit", False):
        return "__end__"
    else:
        return "rewrite_query"
