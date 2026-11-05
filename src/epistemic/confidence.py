"""Self-consistency confidence estimation over sampled model completions."""

from collections import Counter
from typing import Any, Dict, List
from src.eval.metrics import normalize_answer
from src.llm.prompts import answer_messages


def self_consistency(
    llm: Any,
    question: str,
    paras: List[dict],
    draft: str,
    tracker: Any,
    k: int = 5,
    temperature: float = 0.7,
) -> Dict[str, Any]:
    """Sample k independent completions at temperature > 0 and calculate consensus confidence."""
    msgs = answer_messages(question, paras)
    samples = [
        llm.chat(
            msgs,
            tracker,
            temperature=temperature,
            sample_idx=i,
        )
        for i in range(k)
    ]

    norm_samples = [normalize_answer(s) for s in samples]
    norm_draft = normalize_answer(draft)

    counts = Counter(norm_samples)
    top_answer, top_count = counts.most_common(1)[0]
    n_draft_match = sum(1 for s in norm_samples if s == norm_draft)

    return {
        "majority": top_answer,
        "conf_majority": float(top_count / k),
        "conf_draft_agreement": float(n_draft_match / k),
        "samples": samples,
    }
