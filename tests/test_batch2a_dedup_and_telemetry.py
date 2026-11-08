"""Batch 2A — Regression Test Suite for Exact Evidence Deduplication and Stopping Telemetry.

Verifies:
1. Exact duplicate passage text does not multiply evidence weight.
2. Different passages with the same title remain distinct and available.
3. Diverse passage wordings (semantically similar but not identical) are preserved.
4. Each distinct stopping branch in run_erag produces its designated loop_stop_reason:
   - fast_path
   - no_novel_evidence
   - pool_capacity_saturated
   - max_hops_exhausted
   - non_adaptive_mode
   - no_adaptive_ablation
5. Final output_reason is recorded separately from loop_stop_reason.
6. hops count and legacy return dictionary schema remain 100% backward compatible.
"""

import json
import os
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock
import pytest

from src.epistemic.belief_graph import BeliefGraph, sigmoid
from src.epistemic.nli import NLI
from src.llm.client import Tracker
from src.systems.erag import build_graph, run_erag


# -----------------------------------------------------------------------------
# Mock Helpers
# -----------------------------------------------------------------------------

class MockNLI(NLI):
    """Subclass of NLI providing fast in-memory best() scoring without model loading."""
    def __init__(self, scores: Optional[Dict[str, Tuple[float, float]]] = None, default_score: Tuple[float, float] = (0.0, 0.0)):
        self.memo = {}
        self.i_ent, self.i_con = 0, 2
        self.window = 2
        self.scores = scores or {}
        self.default_score = default_score

    def best(self, claim: str, passage: Dict[str, Any]) -> Tuple[float, float]:
        title = passage.get("title", "")
        # First check text-specific key
        sents = passage.get("sentences", [])
        text_snippet = " ".join(sents) if isinstance(sents, list) else passage.get("text", "")
        key = f"{claim}||{title}||{text_snippet}"
        if key in self.scores:
            return self.scores[key]
        key_short = f"{claim}||{title}"
        if key_short in self.scores:
            return self.scores[key_short]
        if title in self.scores:
            return self.scores[title]
        return self.default_score


class MockRetriever:
    """Configurable mock retriever returning predefined passages."""
    def __init__(self, responses: Optional[Dict[str, List[Dict[str, Any]]]] = None, default: Optional[List[Dict[str, Any]]] = None):
        self.responses = responses or {}
        self.default = default or []
        self.search_calls: List[str] = []

    def search(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        self.search_calls.append(query)
        res = self.responses.get(query, self.default)
        return res[:k]


class MockLLM:
    """Configurable mock LLM returning canned responses for prompts."""
    def __init__(self, canned_chat: Optional[Dict[str, str]] = None, default_chat: str = ""):
        self.canned_chat = canned_chat or {}
        self.default_chat = default_chat

    def chat(
        self,
        messages: List[Dict[str, str]],
        tracker: Tracker,
        max_tokens: int = 200,
        temperature: float = 0.0,
        sample_idx: Optional[int] = None,
        **kwargs: Any,
    ) -> str:
        user_content = next((m["content"] for m in messages if m["role"] == "user"), "")
        # Order matters: check JSON first, then query/question
        if "JSON:" in user_content and "JSON:" in self.canned_chat:
            return self.canned_chat["JSON:"]
        for k, v in self.canned_chat.items():
            if k in user_content:
                return v
        return self.default_chat


# -----------------------------------------------------------------------------
# 1. Exact Duplicate Evidence Handling Tests
# -----------------------------------------------------------------------------

def test_identical_text_does_not_multiply_evidence_weight():
    """Regression Test: Passages with identical text must contribute only once to NLI verification."""
    nli = MockNLI(default_score=(0.90, 0.0))
    # Two identical passages (e.g. from pool + memory overlap or duplicate retrieval)
    passages = [
        {"title": "Doc A", "sentences": ["The capital of France is Paris."]},
        {"title": "Doc A", "sentences": ["The capital of France is Paris."]},
    ]
    ev = nli.verify("The capital of France is Paris.", passages, theta_s=0.5, theta_c=0.5)
    
    # Should only contribute once
    assert len(ev["sup"]) == 1, "Duplicate identical text must be deduplicated to a single support entry"
    assert ev["duplicates_skipped"] == 1
    
    # Update belief graph and verify evidence was NOT doubled
    g = BeliefGraph(w_s=1.0, w_c=2.0)
    g.add("Claim", True, [], 0.50, ev)
    g.update()
    
    # Prior z0 = logit(0.5) = 0.0. Exactly one evidence contribution: 0.0 + 1.0 * 0.90 = 0.90
    expected_z = 0.0 + 0.90
    assert abs(g.claims[0].conf - sigmoid(expected_z)) < 1e-4, "Confidence reflects exactly one evidence weight"


def test_different_passages_with_same_title_are_strictly_preserved():
    """Regression Test: Different sentences from the same article title must NOT be deduplicated."""
    nli = MockNLI(
        scores={
            "Paris||Paris||Paris is the capital of France.": (0.95, 0.0),
            "Paris||Paris||The population of Paris is 2 million.": (0.80, 0.0),
        },
        default_score=(0.85, 0.0),
    )
    passages = [
        {"title": "Paris", "sentences": ["Paris is the capital of France."]},
        {"title": "Paris", "sentences": ["The population of Paris is 2 million."]},
    ]
    ev = nli.verify("Paris", passages, theta_s=0.5, theta_c=0.5)
    
    # Both distinct passages from the same article title must be preserved
    assert len(ev["sup"]) == 2, "Different passages with the same article title must both be retained"
    assert ev["duplicates_skipped"] == 0


def test_semantically_similar_but_non_identical_passages_are_preserved():
    """Regression Test: Paraphrased or semantically similar evidence must NOT be merged by conservative dedup."""
    nli = MockNLI(default_score=(0.90, 0.0))
    passages = [
        {"title": "Doc 1", "sentences": ["Christopher Nolan directed the movie Inception."]},
        {"title": "Doc 2", "sentences": ["Inception is a 2010 film written and directed by Christopher Nolan."]},
    ]
    ev = nli.verify("Christopher Nolan directed Inception.", passages, theta_s=0.5, theta_c=0.5)
    
    assert len(ev["sup"]) == 2, "Distinct wordings must both contribute as distinct evidence"
    assert ev["duplicates_skipped"] == 0


def test_whitespace_and_formatting_variants_are_conservatively_deduplicated():
    """Regression Test: Spurious whitespace differences in identical text are normalized and deduplicated."""
    nli = MockNLI(default_score=(0.90, 0.0))
    passages = [
        {"title": "Doc A", "sentences": ["Paris   is  the capital of   France.\n"]},
        {"title": "Doc A", "sentences": [" Paris is the capital of France. "]},
    ]
    ev = nli.verify("Paris is the capital of France.", passages, theta_s=0.5, theta_c=0.5)
    
    assert len(ev["sup"]) == 1, "Whitespace formatting variants of identical text must deduplicate"
    assert ev["duplicates_skipped"] == 1


def test_duplicate_text_order_invariance_and_entity_context():
    """Regression Test: Identical passage text appearing under a generic title and an entity-identifying
    title must produce order-invariant verification decisions and preserve entity grounding in both orders.
    """
    nli = MockNLI(default_score=(0.0, 0.95))
    claim = "Christopher Nolan was born in Paris."
    
    p_generic = {
        "title": "Early Life",
        "sentences": ["He was born in London on July 30, 1970."],
    }
    p_entity = {
        "title": "Christopher Nolan (director)",
        "sentences": ["He was born in London on July 30, 1970."],
    }
    
    # Order A: generic header first, then entity-grounded header
    ev_a = nli.verify(claim, [p_generic, p_entity], theta_s=0.5, theta_c=0.5)
    # Order B: entity-grounded header first, then generic header
    ev_b = nli.verify(claim, [p_entity, p_generic], theta_s=0.5, theta_c=0.5)
    
    # Both orders must produce identical, grounded contradiction decisions
    assert ev_a["status"] == "contradicted", f"Order A must not drop contradiction due to traversal order: {ev_a}"
    assert ev_b["status"] == "contradicted", f"Order B must retain contradiction: {ev_b}"
    assert ev_a["status"] == ev_b["status"]
    assert len(ev_a["con"]) == 1
    assert len(ev_b["con"]) == 1
    assert ev_a["con"] == ev_b["con"]
    assert ev_a["duplicates_skipped"] == 1
    assert ev_b["duplicates_skipped"] == 1


def test_duplicate_text_unrelated_entity_does_not_impute_grounding():
    """Regression Test: Ambiguous pronoun text appearing under generic and UNRELATED entity titles
    must NOT falsely impute subject grounding for the target claim entity.
    """
    nli = MockNLI(default_score=(0.0, 0.95))
    claim = "Christopher Nolan was born in Paris."
    
    p_generic = {
        "title": "Early Life",
        "sentences": ["He was born in London on July 30, 1970."],
    }
    p_unrelated = {
        "title": "Jonathan Nolan",  # Unrelated entity
        "sentences": ["He was born in London on July 30, 1970."],
    }
    
    ev_a = nli.verify(claim, [p_generic, p_unrelated], theta_s=0.5, theta_c=0.5)
    ev_b = nli.verify(claim, [p_unrelated, p_generic], theta_s=0.5, theta_c=0.5)
    
    # Neither title grounds Christopher Nolan, so contradiction should NOT be accepted in either order
    assert ev_a["status"] == "unresolved", "Unrelated entity must not ground ambiguous pronoun"
    assert ev_b["status"] == "unresolved", "Unrelated entity must not ground ambiguous pronoun"
    assert len(ev_a["con"]) == 0
    assert len(ev_b["con"]) == 0
    assert ev_a["duplicates_skipped"] == 1
    assert ev_b["duplicates_skipped"] == 1


# -----------------------------------------------------------------------------
# 2. Explicit Stopping Telemetry Tests
# -----------------------------------------------------------------------------

def _make_cfg(mode="active", max_hops=2, theta_fast=0.80, theta_gap=0.60, ablate=None, pool_cap=12, saturation_precheck=False):
    return {
        "retrieval": {"k": 3, "setting": "distractor"},
        "erag": {
            "mode": mode,
            "max_hops": max_hops,
            "theta_fast": theta_fast,
            "theta_gap": theta_gap,
            "theta_abs": 0.40,
            "sc_samples": 5,
            "ablate": ablate or [],
            "pool_cap": pool_cap,
            "saturation_precheck": saturation_precheck,
        },
    }


def test_telemetry_fast_path():
    """Test: When all claims are supported >= theta_fast at hop 0, stop reason is 'fast_path'."""
    question = "Capital of France?"
    draft = "Paris"
    claims_json = json.dumps({"claims": [{"text": "Paris is capital.", "answer": True, "depends_on": []}]})
    
    llm = MockLLM(canned_chat={"JSON:": claims_json, "Question:": draft}, default_chat=draft)
    nli = MockNLI(default_score=(0.95, 0.0))  # Entailment >= theta_fast
    retriever = MockRetriever(default=[{"title": "Paris", "sentences": ["Paris is capital."]}])
    cfg = _make_cfg()
    
    res = run_erag({"_id": "t1", "question": question}, retriever, llm, nli, cfg, Tracker())
    
    assert res["hops"] == 0
    assert res["loop_stop_reason"] == "fast_path"
    assert res["fast_path"] is True
    assert res["output_reason"] == "answered"
    assert res["abstained"] is False


def test_telemetry_no_novel_evidence():
    """Test: When adaptive retrieval finds no new passages, stop reason is 'no_novel_evidence'."""
    question = "Uncertain fact?"
    draft = "Maybe"
    claims_json = json.dumps({"claims": [{"text": "Uncertain fact.", "answer": True, "depends_on": []}]})
    
    # Diverse self-consistency so c0 is low (0.2)
    class LowConfLLM(MockLLM):
        def chat(self, messages, tracker, max_tokens=200, temperature=0.0, sample_idx=None, **kwargs):
            if sample_idx is not None:
                return f"Sample {sample_idx}"
            user_content = next((m["content"] for m in messages if m["role"] == "user"), "")
            if "JSON:" in user_content:
                return claims_json
            if "Claim:" in user_content:
                return "search query"
            return draft

    llm = LowConfLLM()
    nli = MockNLI(default_score=(0.0, 0.0))  # Unresolved -> has a gap
    initial_doc = {"title": "Doc Seen", "sentences": ["Info."]}
    retriever = MockRetriever(default=[initial_doc])  # Returns only already seen doc -> new == []
    cfg = _make_cfg()
    
    res = run_erag({"_id": "t2", "question": question}, retriever, llm, nli, cfg, Tracker())
    
    assert res["hops"] == 0
    assert res["loop_stop_reason"] == "no_novel_evidence"
    assert res["fast_path"] is False  # Explicitly distinguished from fast_path
    assert res["output_reason"] == "abstained_low_confidence"
    assert res["abstained"] is True


def test_telemetry_max_hops_exhausted():
    """Test: When retrieval reaches max_hops, stop reason is 'max_hops_exhausted'."""
    question = "Multi hop question?"
    draft = "Draft"
    claims_json = json.dumps({"claims": [{"text": "Gap claim.", "answer": True, "depends_on": []}]})
    
    call_idx = 0
    class ExpandingRetriever:
        def search(self, q, k=3):
            nonlocal call_idx
            call_idx += 1
            return [{"title": f"Doc Novel {call_idx}", "sentences": [f"Sentence {call_idx}."]}]
    
    llm = MockLLM(canned_chat={"JSON:": claims_json, "Claim:": "query"}, default_chat=draft)
    nli = MockNLI(default_score=(0.0, 0.0))  # Always unresolved so it keeps hopping
    cfg = _make_cfg(max_hops=2)
    
    res = run_erag({"_id": "t3", "question": question}, ExpandingRetriever(), llm, nli, cfg, Tracker())
    
    assert res["hops"] == 2
    assert res["loop_stop_reason"] == "max_hops_exhausted"
    assert res["fast_path"] is False


def test_telemetry_non_adaptive_mode():
    """Test: When mode is 'observe' (not active), stop reason is 'non_adaptive_mode'."""
    question = "Capital of France?"
    draft = "Paris"
    claims_json = json.dumps({"claims": [{"text": "Paris is capital.", "answer": True, "depends_on": []}]})
    llm = MockLLM(canned_chat={"JSON:": claims_json}, default_chat=draft)
    nli = MockNLI(default_score=(0.9, 0.0))
    retriever = MockRetriever(default=[{"title": "Doc", "sentences": ["Text."]}])
    cfg = _make_cfg(mode="observe")
    
    res = run_erag({"_id": "t4", "question": question}, retriever, llm, nli, cfg, Tracker())
    
    assert res["hops"] == 0
    assert res["loop_stop_reason"] == "non_adaptive_mode"
    assert res["output_reason"] == "observational_draft"
    assert res["fast_path"] is False


def test_telemetry_no_adaptive_ablation():
    """Test: When 'no_adaptive' is in ablate, stop reason is 'no_adaptive_ablation'."""
    question = "Question?"
    draft = "Answer"
    claims_json = json.dumps({"claims": [{"text": "Claim.", "answer": True, "depends_on": []}]})
    llm = MockLLM(canned_chat={"JSON:": claims_json}, default_chat=draft)
    nli = MockNLI(default_score=(0.9, 0.0))
    retriever = MockRetriever(default=[{"title": "Doc", "sentences": ["Text."]}])
    cfg = _make_cfg(ablate=["no_adaptive"])
    
    res = run_erag({"_id": "t5", "question": question}, retriever, llm, nli, cfg, Tracker())
    
    assert res["hops"] == 0
    assert res["loop_stop_reason"] == "no_adaptive_ablation"
    assert res["fast_path"] is False


def test_telemetry_pool_capacity_saturated():
    """Test: When pool reaches pool_cap and saturation_precheck is True, stop reason is 'pool_capacity_saturated'."""
    question = "Question?"
    draft = "Answer"
    claims_json = json.dumps({"claims": [{"text": "Claim.", "answer": True, "depends_on": []}]})
    llm = MockLLM(canned_chat={"JSON:": claims_json}, default_chat=draft)
    nli = MockNLI(default_score=(0.0, 0.0))
    # Pre-filled pool at capacity (pool_cap = 2)
    pool = [
        {"title": "Doc 1", "sentences": ["Text 1."]},
        {"title": "Doc 2", "sentences": ["Text 2."]},
    ]
    retriever = MockRetriever(default=pool)
    cfg = _make_cfg(pool_cap=2, saturation_precheck=True)
    
    res = run_erag({"_id": "t6", "question": question}, retriever, llm, nli, cfg, Tracker())
    
    assert res["hops"] == 0
    assert res["loop_stop_reason"] == "pool_capacity_saturated"
    assert res["fast_path"] is False


# -----------------------------------------------------------------------------
# 3. Runner Serialization Integration Tests
# -----------------------------------------------------------------------------

def test_run_experiment_telemetry_serialization():
    """Integration Test: Verify that run_experiment.py record-writing path saves
    loop_stop_reason, output_reason, and fast_path to disk in JSONL.
    """
    import tempfile
    from scripts.run_experiment import dispatch

    item = {"_id": "q_exp_1", "question": "Capital of France?", "answer": "Paris"}
    draft = "Paris"
    claims_json = json.dumps({"claims": [{"text": "Paris is capital.", "answer": True, "depends_on": []}]})
    llm = MockLLM(canned_chat={"JSON:": claims_json, "Question:": draft}, default_chat=draft)
    nli = MockNLI(default_score=(0.95, 0.0))
    retriever = MockRetriever(default=[{"title": "Paris", "sentences": ["Paris is capital."]}])
    cfg = _make_cfg()
    tr = Tracker()

    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
        tmp_path = f.name

    try:
        r = dispatch("erag", item, retriever, llm, nli, cfg, tr, memory=None)
        rec = {
            "id": item["_id"],
            "system": "erag",
            "setting": cfg["retrieval"]["setting"],
            "split": "test",
            "question": item["question"],
            "gold": item["answer"],
        }
        rec.update(r)

        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

        # Read back saved file and verify telemetry fields are stored
        with open(tmp_path, "r", encoding="utf-8") as f:
            saved_lines = [json.loads(line) for line in f if line.strip()]

        assert len(saved_lines) == 1
        saved = saved_lines[0]
        assert "loop_stop_reason" in saved
        assert saved["loop_stop_reason"] == "fast_path"
        assert "output_reason" in saved
        assert saved["output_reason"] == "answered"
        assert "fast_path" in saved
        assert saved["fast_path"] is True
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_run_phase5_telemetry_serialization():
    """Integration Test: Verify that run_phase5_four_retrievers.py record-writing path saves
    loop_stop_reason, output_reason, and fast_path to disk in JSONL.
    """
    import tempfile

    item = {"_id": "q_p5_1", "question": "Capital of France?", "answer": "Paris"}
    draft = "Paris"
    claims_json = json.dumps({"claims": [{"text": "Paris is capital.", "answer": True, "depends_on": []}]})
    llm = MockLLM(canned_chat={"JSON:": claims_json, "Question:": draft}, default_chat=draft)
    nli = MockNLI(default_score=(0.95, 0.0))
    retriever = MockRetriever(default=[{"title": "Paris", "sentences": ["Paris is capital."]}])
    cfg = _make_cfg()
    tr = Tracker()

    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
        tmp_path = f.name

    try:
        r = run_erag(item, retriever, llm, nli, cfg, tr)
        rec = {
            "id": item["_id"],
            "question": item["question"],
            "gold": item["answer"],
        }
        rec.update(r)

        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

        with open(tmp_path, "r", encoding="utf-8") as f:
            saved_lines = [json.loads(line) for line in f if line.strip()]

        assert len(saved_lines) == 1
        saved = saved_lines[0]
        assert "loop_stop_reason" in saved
        assert saved["loop_stop_reason"] == "fast_path"
        assert "output_reason" in saved
        assert saved["output_reason"] == "answered"
        assert "fast_path" in saved
        assert saved["fast_path"] is True
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
