"""System B: IRCoT-inspired iterative retrieval comparator with interleaved reasoning and retrieval."""

from typing import Any, Dict, List
from src.llm.client import Tracker
from src.llm.prompts import answer_messages, context_block

STEP_SYSTEM = (
    "You are solving a multi-hop question step by step. You get the question, "
    "retrieved paragraphs and the reasoning so far. Write ONLY the next single reasoning sentence. "
    "If you can now answer, write a sentence that begins exactly with 'So the answer is:'."
)


def next_step(
    llm: Any,
    question: str,
    pool: List[Dict[str, Any]],
    steps: List[str],
    tracker: Tracker,
) -> str:
    """Generate the next single CoT reasoning sentence based on current evidence pool."""
    user = (
        f"Question: {question}\n\nParagraphs:\n{context_block(pool)}\n\n"
        f"Reasoning so far: {' '.join(steps) if steps else '(none)'}\n\nNext sentence:"
    )
    msgs = [
        {"role": "system", "content": STEP_SYSTEM},
        {"role": "user", "content": user},
    ]
    out = llm.chat(msgs, tracker, max_tokens=256, temperature=0.0)
    # Take first non-empty line as the next single reasoning sentence
    lines = [line.strip() for line in out.strip().splitlines() if line.strip()]
    return lines[0] if lines else ""


def run_ircot(
    item: Dict[str, Any],
    retriever: Any,
    llm: Any,
    cfg: Dict[str, Any],
    tracker: Tracker,
) -> Dict[str, Any]:
    """Execute IRCoT-inspired iterative reasoning and retrieval pipeline."""
    c = cfg.get("ircot", {})
    max_steps = c.get("max_steps", 4)
    pool_cap = c.get("pool_cap", 12)
    k = cfg["retrieval"].get("k", 3)
    q = item["question"]

    # Step 1: Initial retrieval
    pool = retriever.search(q, k=k)
    tracker.retrievals += 1
    steps: List[str] = []

    # Step 2: Interleaved reasoning and retrieval loop
    for _ in range(max_steps):
        s = next_step(llm, q, pool, steps, tracker)
        steps.append(s)

        # Stopping condition: reached final answer or empty sentence
        if "so the answer is" in s.lower() or not s:
            break

        # Next retrieval query is the reasoning sentence itself
        seen = {p["title"] for p in pool}
        new = [p for p in retriever.search(s, k=k) if p["title"] not in seen]
        tracker.retrievals += 1
        pool = (pool + new)[:pool_cap]

    # Step 3: Final reader answer using shared context-grounded prompt
    answer = llm.chat(answer_messages(q, pool), tracker, temperature=0.0)

    return {
        "answer": answer,
        "pool_titles": [p["title"] for p in pool],
        "hops": len(steps),
        "steps": steps,
    }

