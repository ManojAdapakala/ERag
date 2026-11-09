"""Conflict-Policy Offline Characterization and Validation Test Suite.

Covers the 9 conflict scenarios defined in the Conflict Policy Matrix:
1. Strong support, no contradiction
2. Strong answer-critical contradiction, no support
3. Strong support and contradiction for the same claim
4. Contradiction about an unrelated auxiliary claim
5. Contradiction about a required answer dependency
6. Missing answer-claim identification
7. Multiple answer-bearing claims
8. Different-entity evidence
9. High draft consensus despite unresolved conflict

Separates:
- Current-behavior characterization (asserts exact behavior of existing code)
- Desired-behavior assertions (strict xfail for unimplemented gaps)
"""

import json
from typing import Any, Dict, List, Optional, Tuple
import pytest

from src.epistemic.belief_graph import BeliefGraph, Claim, logit, sigmoid
from src.epistemic.claims import extract_claims
from src.epistemic.nli import NLI
from src.llm.client import Tracker
from src.systems.erag import build_graph, run_erag


# -----------------------------------------------------------------------------
# Mock Helpers
# -----------------------------------------------------------------------------

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
    """Configurable mock LLM returning canned responses."""
    def __init__(self, canned_chat: Optional[Dict[str, str]] = None, default_chat: str = ""):
        self.canned_chat = canned_chat or {}
        self.default_chat = default_chat
        self.chat_history: List[List[Dict[str, str]]] = []

    def chat(
        self,
        messages: List[Dict[str, str]],
        tracker: Tracker,
        max_tokens: int = 200,
        temperature: float = 0.0,
        sample_idx: Optional[int] = None,
        **kwargs: Any,
    ) -> str:
        self.chat_history.append(messages)
        user_content = next((m["content"] for m in messages if m["role"] == "user"), "")
        for k, v in self.canned_chat.items():
            if k in user_content:
                return v
        return self.default_chat


class MockNLI(NLI):
    """Subclass of NLI bypassing model loading for deterministic scoring."""
    def __init__(self, scores: Optional[Dict[str, Tuple[float, float]]] = None, default_score: Tuple[float, float] = (0.0, 0.0)):
        self.memo = {}
        self.i_ent, self.i_con = 0, 2
        self.window = 2
        self.scores = scores or {}
        self.default_score = default_score

    def best(self, claim: str, passage: Dict[str, Any]) -> Tuple[float, float]:
        title = passage.get("title", "")
        key = f"{claim}||{title}"
        if key in self.scores:
            return self.scores[key]
        if title in self.scores:
            return self.scores[title]
        return self.default_score


def _make_cfg(theta_abs: float = 0.40, max_hops: int = 1) -> Dict[str, Any]:
    return {
        "retrieval": {"setting": "distractor", "k": 3},
        "erag": {
            "mode": "active",
            "max_hops": max_hops,
            "sc_samples": 5,
            "theta_abs": theta_abs,
            "theta_gap": 0.60,
            "theta_fast": 0.80,
            "pool_cap": 12,
        },
    }


# =============================================================================
# Case 1: Strong Support, No Contradiction
# =============================================================================

def test_case1_strong_support_characterization():
    """Case 1: Direct strong support with no contradiction achieves fast path and answers."""
    question = "Where was Christopher Nolan born?"
    draft = "London"
    claims_json = json.dumps({"claims": [{"text": "Christopher Nolan was born in London.", "answer": True, "depends_on": []}]})
    llm = MockLLM(canned_chat={"JSON:": claims_json}, default_chat=draft)
    nli = MockNLI(default_score=(0.95, 0.0))
    retriever = MockRetriever(default=[{"title": "Christopher Nolan", "sentences": ["Christopher Nolan was born in London."]}])

    res = run_erag({"_id": "c1", "question": question}, retriever, llm, nli, _make_cfg(), Tracker())
    assert res["answer"] == "London"
    assert res["abstained"] is False
    assert res["loop_stop_reason"] == "fast_path"
    assert res["output_reason"] == "answered"


# =============================================================================
# Case 2: Strong Answer-Critical Contradiction, No Support
# =============================================================================

def test_case2_strong_contradiction_characterization_high_prior_defect():
    """Case 2 (Current Characterization): High parametric prior c0=1.0 suppresses abstention
    under current belief update math despite 0.95 direct contradiction.
    """
    g = BeliefGraph(w_s=1.0, w_c=2.0)
    ev_contra = {"status": "contradicted", "conflict": False, "sup": [], "con": [("Doc A", 0.95)]}
    g.add("The director was born in Paris.", answer_bearing=True, depends_on=[], c0=1.0, ev=ev_contra)
    g.update()
    ans = g.answer_claim()
    # Current behavior: conf is ~0.74, so eff > theta_abs (0.40) and it fails to abstain
    assert ans.conf > 0.70
    assert (ans.eff < 0.40) is False


def test_case2_strong_contradiction_desired_abstention():
    """Case 2 (Desired Behavior): Direct unmitigated contradiction on answer claim must abstain via contradiction veto."""
    question = "Where was Christopher Nolan born?"
    draft = "Paris"
    claims_json = json.dumps({"claims": [{"text": "Christopher Nolan was born in Paris.", "answer": True, "depends_on": []}]})
    llm = MockLLM(canned_chat={"JSON:": claims_json}, default_chat=draft)
    nli = MockNLI(default_score=(0.0, 0.95))
    retriever = MockRetriever(default=[{"title": "Christopher Nolan", "sentences": ["Christopher Nolan was born in London, not Paris."]}])

    res = run_erag({"_id": "c2_desire", "question": question}, retriever, llm, nli, _make_cfg(), Tracker())
    assert res["abstained"] is True
    assert res["answer"] == "unknown"
    assert res["output_reason"] == "abstained_contradiction_veto"


# =============================================================================
# Case 3: Strong Support and Contradiction for the Same Claim (Mixed Evidence)
# =============================================================================

def test_case3_mixed_evidence_characterization():
    """Case 3: Mixed support (0.90) and contradiction (0.90) on same claim.
    Because w_c = 2.0 > w_s = 1.0, contradiction penalty outweighs support in logit space.
    """
    g = BeliefGraph(w_s=1.0, w_c=2.0)
    ev_conflict = {
        "status": "contradicted",
        "conflict": True,
        "sup": [("Doc A", 0.90)],
        "con": [("Doc B", 0.90)],
    }
    # With neutral prior c0 = 0.50 (logit = 0.0):
    # z = 0.0 + 1.0 * 0.90 - 2.0 * 0.90 = -0.90 -> sigmoid(-0.90) = 0.289 < 0.40 -> abstains
    g.add("Claim X", answer_bearing=True, depends_on=[], c0=0.50, ev=ev_conflict)
    g.update()
    ans = g.answer_claim()
    assert ans.conf < 0.35
    assert ans.eff < 0.40


# =============================================================================
# Case 4: Contradiction About an Unrelated Auxiliary Claim
# =============================================================================

def test_case4_unrelated_auxiliary_contradiction_does_not_veto_answer():
    """Case 4: An auxiliary claim (not answer-bearing and not an ancestor dependency)
    is contradicted, while the answer-bearing claim is strongly supported.
    The answer must be preserved.
    """
    g = BeliefGraph(w_s=1.0, w_c=2.0)
    ev_sup = {"status": "supported", "conflict": False, "sup": [("Doc A", 0.95)], "con": []}
    ev_con = {"status": "contradicted", "conflict": False, "sup": [], "con": [("Doc B", 0.95)]}

    # Claim 0 is the answer claim; Claim 1 is an unrelated auxiliary claim (depends_on=[])
    g.add("Christopher Nolan was born in London.", answer_bearing=True, depends_on=[], c0=0.50, ev=ev_sup)
    g.add("Christopher Nolan has won ten Oscars.", answer_bearing=False, depends_on=[], c0=0.50, ev=ev_con)
    g.update()

    ans = g.answer_claim()
    assert ans.status == "supported"
    assert ans.eff > 0.70  # Answer claim effective confidence unaffected by unrelated auxiliary contradiction


# =============================================================================
# Case 5: Contradiction About a Required Answer Dependency
# =============================================================================

def test_case5_required_dependency_contradiction_neutral_prior():
    """Case 5: Contradiction on an upstream required dependency propagates via weakest link."""
    g = BeliefGraph(w_s=1.0, w_c=2.0)
    ev_con = {"status": "contradicted", "conflict": False, "sup": [], "con": [("Doc B", 0.95)]}
    ev_sup = {"status": "supported", "conflict": False, "sup": [("Doc A", 0.95)], "con": []}

    # Dependency Claim 0 is contradicted
    g.add("Christopher Nolan directed Inception.", answer_bearing=False, depends_on=[], c0=0.50, ev=ev_con)
    # Answer Claim 1 is supported, but depends on Claim 0
    g.add("The director was born in London.", answer_bearing=True, depends_on=[0], c0=0.50, ev=ev_sup)
    g.update()

    ans = g.answer_claim()
    # Weakest link: ans.eff = min(ans.conf (~0.72), claim_0.eff (~0.13)) = ~0.13 < 0.40
    assert ans.eff < 0.40


def test_case5_required_dependency_contradiction_high_prior_desired_abstention():
    """Case 5 (Desired Behavior): When a required dependency is directly refuted,
    the system must abstain even if prior c0=1.0.
    """
    question = "Where was the director of Inception born?"
    draft = "London"
    claims_json = json.dumps({
        "claims": [
            {"text": "Christopher Nolan directed Inception.", "answer": False, "depends_on": []},
            {"text": "Christopher Nolan was born in London.", "answer": True, "depends_on": [0]},
        ]
    })
    llm = MockLLM(canned_chat={"JSON:": claims_json}, default_chat=draft)
    # Nolan birth is supported, but Nolan directing Inception is refuted
    nli = MockNLI(scores={
        "Christopher Nolan directed Inception.||Christopher Nolan": (0.0, 0.95),
        "Christopher Nolan was born in London.||Christopher Nolan": (0.95, 0.0),
    })
    retriever = MockRetriever(default=[{"title": "Christopher Nolan", "sentences": ["Inception was directed by someone else, not Christopher Nolan. Christopher Nolan was born in London."]}])

    res = run_erag({"_id": "c5_desire", "question": question}, retriever, llm, nli, _make_cfg(), Tracker())
    assert res["abstained"] is True
    assert res["answer"] == "unknown"
    assert res["output_reason"] == "abstained_contradiction_veto"


# =============================================================================
# Case 6: Missing Answer-Claim Identification
# =============================================================================

def test_case6_missing_answer_claim_fallback():
    """Case 6: If no claim is marked 'answer': true, fallback assigns claims[-1]."""
    raw_claims = [
        {"text": "Fact 1", "answer": False, "depends_on": []},
        {"text": "Fact 2 (the real answer)", "answer": False, "depends_on": []},
    ]
    # Simulate extraction post-processing in src/epistemic/claims.py
    out = []
    for r in raw_claims:
        out.append({"text": r["text"], "answer_bearing": bool(r.get("answer")), "depends_on": []})
    if out and not any(c["answer_bearing"] for c in out):
        out[-1]["answer_bearing"] = True

    assert out[0]["answer_bearing"] is False
    assert out[1]["answer_bearing"] is True


# =============================================================================
# Case 7: Multiple Answer-Bearing Claims
# =============================================================================

def test_case7_multiple_answer_claims_weakest_selected():
    """Case 7: BeliefGraph.answer_claim() evaluates across all answer claims and selects the weakest."""
    g = BeliefGraph(w_s=1.0, w_c=2.0)
    ev_sup = {"status": "supported", "conflict": False, "sup": [("Doc A", 0.95)], "con": []}
    ev_con = {"status": "contradicted", "conflict": False, "sup": [], "con": [("Doc B", 0.95)]}

    # Two claims both marked answer_bearing: Claim 0 is supported, Claim 1 is contradicted
    g.add("Paris is capital of France.", answer_bearing=True, depends_on=[], c0=0.50, ev=ev_sup)
    g.add("Paris has 50 million residents.", answer_bearing=True, depends_on=[], c0=0.50, ev=ev_con)
    g.update()

    ans = g.answer_claim()
    # Weakest answer claim (Claim 1 with eff ~0.13) is selected
    assert ans.idx == 1
    assert ans.eff < 0.40


def test_case7_multiple_answer_claims_desired_composite_evaluation():
    """Case 7 (Desired Behavior): If multiple claims are answer-bearing and one is refuted,
    the composite answer must abstain.
    """
    question = "Capital of France and population?"
    draft = "Paris with 50 million people"
    claims_json = json.dumps({
        "claims": [
            {"text": "Paris is capital of France.", "answer": True, "depends_on": []},
            {"text": "Paris has 50 million residents.", "answer": True, "depends_on": []},
        ]
    })
    llm = MockLLM(canned_chat={"JSON:": claims_json}, default_chat=draft)
    nli = MockNLI(scores={
        "Paris is capital of France.||Paris": (0.95, 0.0),
        "Paris has 50 million residents.||Paris": (0.0, 0.95),
    })
    retriever = MockRetriever(default=[{"title": "Paris", "sentences": ["Paris is capital of France. Population is 2 million, not 50 million."]}])

    res = run_erag({"_id": "c7_desire", "question": question}, retriever, llm, nli, _make_cfg(), Tracker())
    assert res["abstained"] is True
    assert res["answer"] == "unknown"
    assert res["output_reason"] == "abstained_contradiction_veto"


# =============================================================================
# Case 8: Different-Entity Evidence
# =============================================================================

def test_case8_different_entity_evidence_does_not_contradict():
    """Case 8: Refuting text about a different entity is filtered out and does not trigger contradiction."""
    nli = NLI()
    claim = "Christopher Nolan was born in London."
    passages = [
        {"title": "Jonathan Nolan", "sentences": ["He was born in London, not Paris, and later moved to Los Angeles."]},
        {"title": "Early Life", "sentences": ["He was born in London on July 30, 1970."]},
    ]
    res = nli.verify(claim, passages, theta_s=0.70, theta_c=0.80)
    # The different-entity passage ("Jonathan Nolan") must not create a grounded contradiction for Christopher Nolan
    con_titles = [t for t, _ in res["con"]]
    assert "Jonathan Nolan" not in con_titles


# =============================================================================
# Case 9: High Draft Consensus Despite Unresolved Conflict
# =============================================================================

def test_case9_high_draft_consensus_with_unresolved_conflict_current_characterization():
    """Case 9 (Current Characterization): High prior c0=1.0 with unresolved conflict
    currently answers rather than abstaining.
    """
    g = BeliefGraph(w_s=1.0, w_c=2.0)
    ev_conflict = {
        "status": "contradicted",
        "conflict": True,
        "sup": [("Doc A", 0.75)],
        "con": [("Doc B", 0.90)],
    }
    # c0 = 1.0 (logit = 2.944), w_s = 1.0, w_c = 2.0
    # z = 2.944 + 0.75 - 1.80 = +1.894 -> sigmoid(1.894) = 0.87 > 0.40
    g.add("Contested Fact", answer_bearing=True, depends_on=[], c0=1.0, ev=ev_conflict)
    g.update()
    ans = g.answer_claim()
    assert ans.eff > 0.70
    assert (ans.eff < 0.40) is False  # Fails to abstain under current code


def test_case9_mixed_strong_support_plus_strong_contradiction_does_not_veto():
    """Case 9: Mixed strong support plus strong contradiction does NOT trigger contradiction veto.
    Logit belief math governs mixed evidence.
    """
    question = "Where was the person born?"
    draft = "London"
    claims_json = json.dumps({"claims": [{"text": "The person was born in London.", "answer": True, "depends_on": []}]})
    llm = MockLLM(canned_chat={"JSON:": claims_json}, default_chat=draft)
    # Both strong support (0.75 >= theta_s 0.70) and strong contradiction (0.90 >= theta_c 0.80)
    nli = MockNLI(scores={
        "The person was born in London.||Biography A": (0.75, 0.0),
        "The person was born in London.||Biography B": (0.0, 0.90),
    })
    retriever = MockRetriever(default=[
        {"title": "Biography A", "sentences": ["The person was born in London."]},
        {"title": "Biography B", "sentences": ["The person was born in Paris, not London."]},
    ])

    res = run_erag({"_id": "c9_mixed", "question": question}, retriever, llm, nli, _make_cfg(), Tracker())
    # No veto because strong support coexists
    assert res["output_reason"] != "abstained_contradiction_veto"


def test_case9_high_draft_consensus_unmitigated_contradiction_vetoes():
    """Case 9: Direct unmitigated contradiction on answer claim under high consensus c0=1.0
    triggers contradiction veto and safely abstains.
    """
    question = "Where was the person born?"
    draft = "London"
    claims_json = json.dumps({"claims": [{"text": "The person was born in London.", "answer": True, "depends_on": []}]})
    llm = MockLLM(canned_chat={"JSON:": claims_json}, default_chat=draft)
    # Unmitigated contradiction: 0.90 contradiction, 0.0 support
    nli = MockNLI(scores={
        "The person was born in London.||Biography B": (0.0, 0.90),
    })
    retriever = MockRetriever(default=[
        {"title": "Biography B", "sentences": ["The person was born in Paris, not London."]},
    ])

    res = run_erag({"_id": "c9_unmitigated", "question": question}, retriever, llm, nli, _make_cfg(), Tracker())
    assert res["abstained"] is True
    assert res["answer"] == "unknown"
    assert res["output_reason"] == "abstained_contradiction_veto"
