"""System A: Standard single-pass RAG baseline and closed-book answering."""

from typing import Any, Dict
from src.llm.prompts import answer_messages, closed_book_messages


def run_standard(
    item: Dict[str, Any],
    retriever: Any,
    llm: Any,
    cfg: Dict[str, Any],
    tracker: Any,
) -> Dict[str, Any]:
    """Execute Standard RAG pipeline: retrieve top-k context -> LLM generation."""
    k = cfg["retrieval"].get("k", 3)
    paras = retriever.search(item["question"], k=k)
    tracker.retrievals += 1

    msgs = answer_messages(item["question"], paras)
    answer = llm.chat(msgs, tracker, temperature=0.0)

    return {
        "answer": answer,
        "pool_titles": [p["title"] for p in paras],
        "hops": 0,
    }


def run_closed_book(
    item: Dict[str, Any],
    retriever: Any,
    llm: Any,
    cfg: Dict[str, Any],
    tracker: Any,
) -> Dict[str, Any]:
    """Execute closed-book baseline without retrieval context."""
    msgs = closed_book_messages(item["question"])
    answer = llm.chat(msgs, tracker, temperature=0.0)

    return {
        "answer": answer,
        "pool_titles": [],
        "hops": 0,
    }
