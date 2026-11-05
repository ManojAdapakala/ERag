"""Prompt definitions and message formatters for LLM generation."""

ANSWER_SYSTEM = (
    "Answer the question using only the context. Reply with only the short answer, "
    "no explanation. If the context does not contain the answer, reply exactly: unknown"
)


def context_block(paras: list) -> str:
    """Format a list of paragraph dicts into a numbered context block."""
    return "\n\n".join(
        f"[{i+1}] {p['title']}: " + (" ".join(p["sentences"]) if isinstance(p.get("sentences"), list) else p.get("text", ""))
        for i, p in enumerate(paras)
    )


def answer_messages(question: str, paras: list) -> list:
    """Messages for context-grounded short-answer question answering."""
    return [
        {"role": "system", "content": ANSWER_SYSTEM},
        {
            "role": "user",
            "content": f"Context:\n{context_block(paras)}\n\nQuestion: {question}\nAnswer:",
        },
    ]


CLOSED_BOOK_SYSTEM = (
    "Answer the question with only a short answer, no explanation. If you do not know, reply exactly: unknown"
)


def closed_book_messages(question: str) -> list:
    """Messages for closed-book question answering without context."""
    return [
        {"role": "system", "content": CLOSED_BOOK_SYSTEM},
        {"role": "user", "content": f"Question: {question}\nAnswer:"},
    ]

