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
        raw_deps = r.get("depends_on", [])
        deps = [
            d for d in raw_deps
            if isinstance(d, int) and 0 <= d < len(out)
        ]
        missing_deps = any(
            isinstance(d, int) and (d < 0 or d >= len(out))
            for d in raw_deps
        )
        out.append({
            "text": str(r["text"]).strip(),
            "answer_bearing": bool(r.get("answer")),
            "depends_on": deps,
            "missing_deps": bool(missing_deps),
            "fallback": False,
        })

    # Ensure at least one claim is marked as the answer-bearing claim
    if out and not any(c["answer_bearing"] for c in out):
        out[-1]["answer_bearing"] = True
    return out


def _detect_relational_bridge(
    question: str,
    answer: str,
    paras: List[dict],
) -> Optional[Dict[str, str]]:
    """Detect ungrounded relational bridge pattern (e.g. director of Inception)."""
    # Matches relational patterns like "director of Inception", "author of The Hobbit", etc.
    m = re.search(
        r'\b(?:the\s+)?(director|writer|author|creator|inventor|founder|president|leader|actor|singer|star)\s+of\s+([A-Za-z0-9\s\'\-]+?)(?:\s+(?:born|die|died|live|lived|win|won|start|started|release|released|create|created|first|begin)|\?|$)',
        question,
        re.IGNORECASE,
    )
    if not m:
        return None

    relation = m.group(1).lower()
    q_entity = m.group(2).strip().strip("?,.")
    if not q_entity:
        return None

    # Check if question entity is already mentioned in evidence pool
    pool_texts = []
    for p in paras:
        pool_texts.append(p.get("title", ""))
        s = p.get("sentences", [])
        if isinstance(s, list):
            pool_texts.extend(s)
        elif "text" in p:
            pool_texts.append(p["text"])
    full_pool = " ".join(pool_texts).lower()

    if q_entity.lower() in full_pool:
        return None

    # Extract subject entity candidate from draft answer:
    # 1. Prefer subject clause prior to primary verbs
    verb_re = re.compile(
        r'\b(is|was|were|are|has|have|had|served|became|directed|starred|defeated|won|played|wrote|invented|created|lived|started|born)\b',
        re.IGNORECASE,
    )
    m_verb = verb_re.search(answer)
    subj_clause = answer[:m_verb.start()].strip() if m_verb else answer.strip().strip(".")
    subj_clause = re.sub(r'^(The|A|An)\s+', '', subj_clause).strip()

    cand_entity = ""
    if subj_clause and subj_clause.lower() != q_entity.lower():
        cand_entity = subj_clause
    else:
        cap_phrases = re.findall(r'\b(?:[A-Z0-9\.\'-]+(?:\s+[A-Z0-9\.\'-]+)*)\b', answer)
        for p in cap_phrases:
            cleaned = p.strip("., ")
            if cleaned.lower() not in (q_entity.lower(), "the", "a", "an"):
                cand_entity = cleaned
                break

    if not cand_entity or cand_entity.lower() == q_entity.lower():
        return None

    if relation == "director":
        bridge_text = f"{cand_entity} directed {q_entity}."
    elif relation in ("writer", "author"):
        bridge_text = f"{cand_entity} wrote {q_entity}."
    elif relation in ("creator", "inventor"):
        bridge_text = f"{cand_entity} invented {q_entity}."
    else:
        bridge_text = f"{cand_entity} is the {relation} of {q_entity}."

    ans_text = answer.strip()
    if not ans_text.endswith("."):
        ans_text += "."

    return {
        "relation": relation,
        "q_entity": q_entity,
        "cand_entity": cand_entity,
        "bridge_text": bridge_text,
        "ans_text": ans_text,
    }


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

    bridge_info = _detect_relational_bridge(question, answer, paras)
    if bridge_info:
        q_ent_lower = bridge_info["q_entity"].lower()
        has_bridge = any(q_ent_lower in c["text"].lower() for c in claims)
        if not has_bridge:
            bridge_claim = {
                "text": bridge_info["bridge_text"],
                "answer_bearing": False,
                "depends_on": [],
                "missing_deps": False,
                "fallback": False,
            }
            if not claims:
                claims = [
                    bridge_claim,
                    {
                        "text": bridge_info["ans_text"],
                        "answer_bearing": True,
                        "depends_on": [0],
                        "missing_deps": False,
                        "fallback": False,
                    },
                ]
            else:
                new_claims = [bridge_claim]
                for c in claims:
                    shifted_deps = [d + 1 for d in c["depends_on"]]
                    if c["answer_bearing"] and 0 not in shifted_deps:
                        shifted_deps.append(0)
                    c["depends_on"] = sorted(shifted_deps)
                    new_claims.append(c)
                claims = new_claims[:max_claims]

    if not claims:
        # Fallback to single atomic claim when structured generation fails
        claims = [
            {
                "text": f'The answer to "{question}" is {answer}.',
                "answer_bearing": True,
                "depends_on": [],
                "missing_deps": False,
                "fallback": True,
            }
        ]
    return claims

