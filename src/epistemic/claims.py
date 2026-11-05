"""Claim decomposition: extract atomic factual claims and dependency indices from generated answers."""

import json
import re
from typing import Any, Dict, List
from src.llm.prompts import context_block

CLAIM_SYSTEM = (
    "You list the short factual claims that must be true for an answer to be correct. "
    "Use at most {n} claims. Each claim is one self-contained sentence that names its entities "
    '(no pronouns). Mark exactly one claim that states the final answer with "answer": true. '
    '"depends_on" lists the 0-based indexes of EARLIER claims it relies on. '
    'Return only JSON: {{"claims": [{{"text": "...", "answer": false, "depends_on": []}}]}}'
)


def _parse(raw: str, max_claims: int) -> List[Dict[str, Any]]:
    """Parse structured JSON claim decomposition from LLM response."""
    if not raw or not isinstance(raw, str):
        return []
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return []
    try:
        rows = json.loads(m.group(0)).get("claims", [])
    except Exception:
        return []

    out: List[Dict[str, Any]] = []
    for r in rows[:max_claims]:
        if not isinstance(r, dict) or not str(r.get("text", "")).strip():
            continue
        # Dependencies must point only to EARLIER claims in the list (enforces DAG acyclicity)
        deps = [
            d for d in r.get("depends_on", [])
            if isinstance(d, int) and 0 <= d < len(out)
        ]
        out.append({
            "text": str(r["text"]).strip(),
            "answer_bearing": bool(r.get("answer")),
            "depends_on": deps,
            "fallback": False,
        })

    # Ensure at least one claim is marked as the answer-bearing claim
    if out and not any(c["answer_bearing"] for c in out):
        out[-1]["answer_bearing"] = True
    return out


def extract_claims(
    llm: Any,
    question: str,
    answer: str,
    paras: List[dict],
    tracker: Any,
    max_claims: int = 5,
) -> List[Dict[str, Any]]:
    """Prompt the LLM to decompose answer and context into atomic verifiable claims."""
    msgs = [
        {"role": "system", "content": CLAIM_SYSTEM.format(n=max_claims)},
        {
            "role": "user",
            "content": f"Question: {question}\nAnswer: {answer}\nContext:\n{context_block(paras)}\n\nJSON:",
        },
    ]

    raw_response = llm.chat(msgs, tracker, temperature=0.0)
    claims = _parse(raw_response, max_claims)

    if not claims:
        # Fallback to single atomic claim when structured generation fails
        claims = [
            {
                "text": f'The answer to "{question}" is {answer}.',
                "answer_bearing": True,
                "depends_on": [],
                "fallback": True,
            }
        ]
    return claims
