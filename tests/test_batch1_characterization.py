"""Batch 1 — Characterization and Regression Test Suite.

Covers:
1. Missing dependency / Fast Path false early-exit
2. High prior + contradiction logit dynamics & abstention
3. Subject filtering on explicit entities vs pronouns
4. IRCoT empty responses, whitespace, stopping phrases, and provider exceptions
5. Evidence accounting (duplicate passages, recomputation, double-counting)
6. Reporting checks (hops == 0 zero-hop execution vs true Fast Path)
"""

import json
import math
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock
import pytest

from src.epistemic.belief_graph import BeliefGraph, Claim, logit, sigmoid
from src.epistemic.nli import NLI
from src.llm.client import Tracker
from src.systems.erag import build_graph, run_erag
from src.systems.ircot import next_step, run_ircot


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
    """Configurable mock LLM returning canned responses for prompts."""
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
        # Match based on user message content
        user_content = next((m["content"] for m in messages if m["role"] == "user"), "")
        for k, v in self.canned_chat.items():
            if k in user_content:
                return v
        return self.default_chat


class MockNLI(NLI):
    """Subclass of NLI that bypasses heavyweight model loading for fast unit testing."""
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


# -----------------------------------------------------------------------------
# 1. Missing Dependency / Fast Path Characterization Tests
# -----------------------------------------------------------------------------

def test_missing_dependency_causes_false_fast_path_exit():
    """Characterization Test: Draft answers question without establishing the bridge premise.
    
    Question: Where was the director of Inception born?
    Draft: Christopher Nolan was born in London.
    Missing Premise: Christopher Nolan directed Inception (never extracted or verified).
    Current code treats the question as complete and exits at Hop 0 (Fast Path).
    """
    question = "Where was the director of Inception born?"
    draft_answer = "Christopher Nolan was born in London."
    
    # Initial pool supports Nolan's birthplace but says NOTHING about Inception
    initial_pool = [{
        "title": "Christopher Nolan (biography)",
        "sentences": ["Christopher Nolan was born in London on July 30, 1970."],
        "text": "Christopher Nolan was born in London on July 30, 1970.",
    }]
    
    # LLM extracts only the asserted claim from the draft (in valid JSON)
    claims_json = json.dumps({
        "claims": [{
            "text": "Christopher Nolan was born in London.",
            "answer": True,
            "depends_on": [],
        }]
    })
    
    llm = MockLLM(
        canned_chat={
            "JSON:": claims_json,
            "Question: " + question: draft_answer,
        },
        default_chat=draft_answer,
    )
    
    # NLI verifies that London birthplace is entailed by the pool doc
    nli = MockNLI(
        scores={
            "Christopher Nolan was born in London.||Christopher Nolan (biography)": (0.95, 0.01),
        }
    )
    
    retriever = MockRetriever(default=initial_pool)
    tracker = Tracker()
    cfg = {
        "retrieval": {"k": 3},
        "erag": {
            "mode": "active",
            "max_hops": 2,
            "theta_fast": 0.80,
            "theta_gap": 0.60,
            "theta_abs": 0.40,
            "sc_samples": 5,
        },
    }
    
    item = {"_id": "test_dep_1", "question": question}
    result = run_erag(item, retriever, llm, nli, cfg, tracker)
    
    # CURRENT BEHAVIOR: Exits at hop 0 (Fast Path) despite missing the director bridge premise
    assert result["hops"] == 0, "Characterization: current code exits at hop 0 without retrieving the bridge premise"
    assert result["answer"] == draft_answer
    assert result["abstained"] is False
    # The retriever was only called once (initial retrieval for question), never for follow-up
    assert tracker.retrievals == 1


@pytest.mark.xfail(reason="E-RAG currently lacks bridge-premise dependency extraction; intended behavior is to reject Fast-Path on missing bridge premises")
def test_missing_dependency_should_not_fast_path():
    """Regression / Desired Behavior Test: Incomplete draft missing intermediate bridge
    entity must NOT exit at Hop 0.
    """
    question = "Where was the director of Inception born?"
    draft_answer = "Christopher Nolan was born in London."
    
    initial_pool = [{
        "title": "Christopher Nolan",
        "sentences": ["Christopher Nolan was born in London."],
    }]
    
    # Under desired behavior, missing bridge ("Who directed Inception?") must trigger hop > 0
    # or identify an ungrounded premise gap.
    llm = MockLLM(default_chat=draft_answer)
    nli = MockNLI(default_score=(0.95, 0.01))
    retriever = MockRetriever(default=initial_pool)
    cfg = {"retrieval": {"k": 3}, "erag": {"mode": "active", "max_hops": 2}}
    item = {"_id": "test_dep_2", "question": question}
    
    result = run_erag(item, retriever, llm, nli, cfg, Tracker())
    # Desired assertion: Should NOT exit at hop 0 when the question's bridge relation is unverified
    assert result["hops"] > 0, "Desired: system should retrieve bridge entity before completing"


# -----------------------------------------------------------------------------
# 2. High Prior + Contradiction Tests
# -----------------------------------------------------------------------------

def test_high_prior_with_strong_contradiction_suppresses_abstention():
    """Characterization Test: High parametric consensus (c0=1.0, z0=+2.944) overrides
    strong retrieved contradiction under current update math.
    """
    g = BeliefGraph(w_s=1.0, w_c=2.0)
    ev_contra = {
        "status": "contradicted",
        "conflict": False,
        "sup": [],
        "con": [("Refuting Doc", 0.95)],
    }
    # Prior c0 = 1.0 -> clipped to 0.95 -> logit = +2.9444
    g.add("The director was born in Paris.", answer_bearing=True, depends_on=[], c0=1.0, ev=ev_contra)
    g.update()
    
    ans = g.answer_claim()
    # z = +2.9444 - 2.0 * 0.95 = +1.0444 -> sigmoid(1.0444) = 0.7397
    assert ans.conf > 0.70, "Current math: c.conf remains ~0.74 despite strong direct contradiction"
    assert ans.eff > 0.70
    
    # Under current abstention rule: abstain only if eff < 0.40
    theta_abs = 0.40
    should_abstain = ans.eff < theta_abs
    assert should_abstain is False, "Current defect: does NOT abstain despite 0.95 contradiction"


def test_high_prior_with_strong_contradiction_final_decision_should_abstain():
    """Desired Behavior Test: A question whose draft answer is strongly refuted by retrieved evidence
    must safely abstain in final decision (e.g. answer == 'unknown' or abstained is True),
    either via confidence drop or an explicit contradiction veto policy.
    """
    question = "Where was the director born?"
    draft_answer = "Paris"
    claims_json = json.dumps({"claims": [{"text": "The director was born in Paris.", "answer": True, "depends_on": []}]})
    
    llm = MockLLM(canned_chat={"JSON:": claims_json}, default_chat=draft_answer)
    # Refuting passage entails contradiction
    nli = MockNLI(default_score=(0.0, 0.95))
    retriever = MockRetriever(default=[{"title": "Biography", "sentences": ["The director was born in London, not Paris."]}])
    cfg = {
        "retrieval": {"k": 3},
        "erag": {"mode": "active", "max_hops": 1, "sc_samples": 5, "theta_abs": 0.40},
    }
    item = {"_id": "test_contra_abs", "question": question}
    result = run_erag(item, retriever, llm, nli, cfg, Tracker())
    
    # Desired assertion: System should safely abstain from confirming the refuted draft
    assert result["abstained"] is True or result["answer"] == "unknown", "Desired: safe abstention policy on refuted draft"


def test_strong_support_only():
    """Characterization: Strong support with neutral prior raises confidence."""
    g = BeliefGraph(w_s=1.0, w_c=2.0)
    ev_sup = {"status": "supported", "conflict": False, "sup": [("Doc A", 0.95)], "con": []}
    g.add("Fact A", answer_bearing=True, depends_on=[], c0=0.5, ev=ev_sup)
    g.update()
    ans = g.answer_claim()
    # logit(0.5) = 0.0 + 1.0 * 0.95 = 0.95 -> sigmoid(0.95) = 0.721
    assert abs(ans.conf - sigmoid(0.95)) < 1e-4
    assert ans.eff >= 0.40


def test_both_support_and_contradiction():
    """Characterization: Mixed evidence update with balanced weights."""
    g = BeliefGraph(w_s=1.0, w_c=2.0)
    ev_mixed = {
        "status": "contradicted",
        "conflict": True,
        "sup": [("Doc Sup", 0.90)],
        "con": [("Doc Con", 0.90)],
    }
    # Prior c0 = 0.5 (logit = 0.0)
    # z = 0.0 + 1.0 * 0.90 - 2.0 * 0.90 = -0.90 -> sigmoid(-0.90) = 0.289
    g.add("Ambiguous Fact", answer_bearing=True, depends_on=[], c0=0.5, ev=ev_mixed)
    g.update()
    ans = g.answer_claim()
    assert ans.eff < 0.40, "With c0=0.5 and w_c=2.0, contradiction outweighs support"
    
    # BUT with high prior c0 = 0.90 (logit = 2.197):
    # z = 2.197 - 0.90 = 1.297 -> sigmoid(1.297) = 0.785
    g_high = BeliefGraph(w_s=1.0, w_c=2.0)
    g_high.add("Ambiguous Fact", answer_bearing=True, depends_on=[], c0=0.90, ev=ev_mixed)
    g_high.update()
    assert g_high.answer_claim().eff > 0.70, "High prior preserves confidence despite conflicting evidence"


def test_irrelevant_contradicted_claim_ignored_by_abstention():
    """Characterization: A contradicted non-answer claim does NOT trigger abstention
    if it is not an explicit dependency of the answer claim.
    """
    g = BeliefGraph(w_s=1.0, w_c=2.0)
    # Claim 0 is the answer claim, uncontradicted
    g.add("The answer is London.", answer_bearing=True, depends_on=[], c0=0.80, ev={"status": "unresolved"})
    # Claim 1 is an auxiliary claim that was severely contradicted
    g.add("The film was directed in 1920.", answer_bearing=False, depends_on=[], c0=0.50, ev={"status": "contradicted", "con": [("Doc", 0.99)]})
    g.update()
    
    ans = g.answer_claim()
    assert ans.text == "The answer is London."
    assert ans.eff >= 0.40
    # System evaluates abstention on ans.eff only; Claim 1's contradiction is ignored
    assert g.claims[1].eff < 0.20
    assert (ans.eff < 0.40) is False, "Irrelevant or unlinked contradiction does not trigger abstention"


def test_missing_or_ambiguous_answer_claim():
    """Characterization: Fallback behaviors when answer-bearing flag is missing or duplicate."""
    # Case 1: No claim marked answer_bearing -> falls back to claims[-1]
    g1 = BeliefGraph()
    g1.add("Claim 0", answer_bearing=False, depends_on=[], c0=0.5, ev={})
    g1.add("Claim 1", answer_bearing=False, depends_on=[], c0=0.9, ev={})
    g1.update()
    assert g1.answer_claim().text == "Claim 1", "Falls back to the last claim"
    
    # Case 2: Multiple claims marked answer_bearing -> returns the FIRST one encountered
    g2 = BeliefGraph()
    g2.add("Claim A", answer_bearing=True, depends_on=[], c0=0.5, ev={})
    g2.add("Claim B", answer_bearing=True, depends_on=[], c0=0.9, ev={})
    g2.update()
    assert g2.answer_claim().text == "Claim A", "Takes the first answer-bearing claim, ignoring subsequent ones"


# -----------------------------------------------------------------------------
# 3. Subject Filtering Characterization Tests
# -----------------------------------------------------------------------------

def test_subject_filtering_explicit_entity():
    """Test: Explicit entity match retains contradiction."""
    nli = MockNLI(default_score=(0.0, 0.95))
    claim = "Christopher Nolan was born in London."
    passages = [{
        "title": "Christopher Nolan (director)",
        "sentences": ["Christopher Nolan was born in Hertfordshire."],
    }]
    ev = nli.verify(claim, passages, theta_s=0.5, theta_c=0.5)
    assert len(ev["con"]) == 1, "Explicit entity in title retains contradiction"
    assert ev["status"] == "contradicted"


def test_subject_filtering_pronoun_with_context_blindspot():
    """Characterization Test: Passage containing valid contradiction using pronoun
    is discarded by subject filter because entity name is absent from the window.
    """
    nli = MockNLI(default_score=(0.0, 0.95))
    claim = "Christopher Nolan was born in London."
    # Passage has generic section title and pronoun reference
    passages = [{
        "title": "Early Life and Family",
        "sentences": ["He was born in Hertfordshire, not London."],
    }]
    ev = nli.verify(claim, passages, theta_s=0.5, theta_c=0.5)
    # CURRENT BEHAVIOR: mentions_subject evaluates to False, contradiction is dropped
    assert len(ev["con"]) == 0, "Current limitation: pronoun window without entity string drops contradiction"
    assert ev["status"] == "unresolved"


def test_subject_filtering_different_entity_properly_rejected():
    """Test: Passage about a different entity is correctly rejected for contradiction."""
    nli = MockNLI(default_score=(0.0, 0.95))
    claim = "Christopher Nolan was born in London."
    passages = [{
        "title": "Jonathan Nolan",
        "sentences": ["Jonathan Nolan was born in London."],
    }]
    ev = nli.verify(claim, passages, theta_s=0.5, theta_c=0.5)
    # Title 'Jonathan Nolan' does not match 'Christopher Nolan' entity
    assert len(ev["con"]) == 0, "Different entity properly excluded from contradiction"


def test_subject_filtering_paraphrased_support_accepted():
    """Test: Paraphrased support is accepted when subject entity matches."""
    nli = MockNLI(default_score=(0.90, 0.0))
    claim = "The director of Inception is Christopher Nolan."
    passages = [{
        "title": "Inception (film)",
        "sentences": ["The film was written and directed by Christopher Nolan."],
    }]
    ev = nli.verify(claim, passages, theta_s=0.5, theta_c=0.5)
    assert len(ev["sup"]) == 1
    assert ev["status"] == "supported"


# -----------------------------------------------------------------------------
# 4. IRCoT Empty Responses & Stopping Logic Tests
# -----------------------------------------------------------------------------

def test_ircot_empty_content_causes_premature_break():
    """Characterization Test: Empty content from LLM (e.g. max_tokens exhaustion)
    causes IRCoT loop to abort immediately after step 0.
    """
    retriever = MockRetriever(default=[{"title": "Doc 1", "sentences": ["Text 1"]}])
    tracker = Tracker()
    llm = MockLLM(default_chat="")  # Returns empty text
    cfg = {"retrieval": {"k": 3}, "ircot": {"max_steps": 4}}
    item = {"_id": "ircot_1", "question": "Multi-hop question?"}
    
    res = run_ircot(item, retriever, llm, cfg, tracker)
    assert res["hops"] == 1
    assert res["steps"] == [""]
    # Only the initial retrieval was performed; multi-hop retrieval was aborted
    assert tracker.retrievals == 1


def test_ircot_whitespace_content_causes_premature_break():
    """Characterization Test: Whitespace-only content also results in immediate break."""
    retriever = MockRetriever(default=[{"title": "Doc 1", "sentences": ["Text 1"]}])
    tracker = Tracker()
    llm = MockLLM(default_chat="   \n\n\t   ")
    cfg = {"retrieval": {"k": 3}, "ircot": {"max_steps": 4}}
    item = {"_id": "ircot_2", "question": "Multi-hop question?"}
    
    res = run_ircot(item, retriever, llm, cfg, tracker)
    assert res["hops"] == 1
    assert res["steps"] == [""]
    assert tracker.retrievals == 1


def test_ircot_normal_reasoning_sentence_continues():
    """Test: Valid reasoning sentence triggers subsequent retrieval hop."""
    retriever = MockRetriever(default=[{"title": "Doc Novel", "sentences": ["Novel Text"]}])
    tracker = Tracker()
    # Step 1: reasoning sentence; Step 2: stopping sentence
    call_count = 0
    def mock_chat(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return "First step: find the director."
        elif call_count == 2:
            return "So the answer is: Christopher Nolan."
        return "Christopher Nolan"
    
    llm = MagicMock()
    llm.chat.side_effect = mock_chat
    
    cfg = {"retrieval": {"k": 3}, "ircot": {"max_steps": 4}}
    item = {"_id": "ircot_3", "question": "Multi-hop question?"}
    res = run_ircot(item, retriever, llm, cfg, tracker)
    
    assert res["hops"] == 2
    assert "First step: find the director." in res["steps"][0]
    # Retrieval occurred for initial step + step 1
    assert tracker.retrievals == 2


def test_ircot_explicit_answer_stop_phrase():
    """Test: Sentence beginning with 'So the answer is:' halts further reasoning hops."""
    retriever = MockRetriever(default=[{"title": "Doc 1", "sentences": ["Text 1"]}])
    tracker = Tracker()
    llm = MockLLM(default_chat="So the answer is: 42.")
    cfg = {"retrieval": {"k": 3}, "ircot": {"max_steps": 4}}
    item = {"_id": "ircot_4", "question": "What is the answer?"}
    
    res = run_ircot(item, retriever, llm, cfg, tracker)
    assert res["hops"] == 1
    assert "so the answer is" in res["steps"][0].lower()
    assert tracker.retrievals == 1


def test_ircot_provider_exception_propagates_unhandled():
    """Characterization Test: Provider API error raises unhandled exception out of run_ircot."""
    retriever = MockRetriever(default=[{"title": "Doc 1", "sentences": ["Text 1"]}])
    llm = MagicMock()
    llm.chat.side_effect = RuntimeError("Groq rate limit exceeded")
    cfg = {"retrieval": {"k": 3}, "ircot": {"max_steps": 4}}
    item = {"_id": "ircot_5", "question": "Question?"}
    
    with pytest.raises(RuntimeError, match="Groq rate limit exceeded"):
        run_ircot(item, retriever, llm, cfg, Tracker())


# -----------------------------------------------------------------------------
# 5. Evidence Accounting & Rebuilding Tests
# -----------------------------------------------------------------------------

def test_duplicate_passage_in_single_update_does_not_double_evidence_weight():
    """Regression Test: Passages with identical text must not contribute repeatedly within one update."""
    nli = MockNLI(default_score=(0.90, 0.0))
    # Passages contains two identical entries (e.g. from pool + memory overlap)
    passages = [
        {"title": "Doc A", "sentences": ["Sentence 1"]},
        {"title": "Doc A", "sentences": ["Sentence 1"]},
    ]
    ev = nli.verify("Claim A", passages, theta_s=0.5, theta_c=0.5)
    assert len(ev["sup"]) == 1, "Exact duplicate passage text must contribute exactly once"
    assert ev.get("duplicates_skipped") == 1
    
    g = BeliefGraph(w_s=1.0, w_c=2.0)
    g.add("Claim A", answer_bearing=True, depends_on=[], c0=0.5, ev=ev)
    g.update()
    
    # logit(0.5) = 0.0 + 1.0 * 0.90 = 0.90 (single contribution)
    expected_z = 0.0 + 0.90
    assert abs(g.claims[0].conf - sigmoid(expected_z)) < 1e-4


def test_fresh_graph_rebuilt_across_hops():
    """Test: Graph is completely rebuilt at each hop from c0 prior, not accumulating old logits."""
    g1 = BeliefGraph(w_s=1.0, w_c=2.0)
    ev_hop0 = {"status": "supported", "sup": [("Doc 1", 0.50)], "con": []}
    g1.add("Claim A", True, [], 0.5, ev_hop0)
    g1.update()
    conf_hop0 = g1.claims[0].conf  # sigmoid(0.0 + 0.50) = sigmoid(0.50)
    
    # Hop 1: New evidence pool with Doc 1 and Doc 2
    g2 = BeliefGraph(w_s=1.0, w_c=2.0)
    ev_hop1 = {"status": "supported", "sup": [("Doc 1", 0.50), ("Doc 2", 0.50)], "con": []}
    g2.add("Claim A", True, [], 0.5, ev_hop1)
    g2.update()
    conf_hop1 = g2.claims[0].conf  # sigmoid(0.0 + 1.00) = sigmoid(1.00)
    
    assert conf_hop1 > conf_hop0
    assert abs(conf_hop1 - sigmoid(1.00)) < 1e-4, "Rebuilt from c0, not accumulated onto conf_hop0"


# -----------------------------------------------------------------------------
# 6. Reporting Checks: Zero-Hop Execution vs Fast Path
# -----------------------------------------------------------------------------

def test_hops_zero_does_not_uniquely_identify_fast_path():
    """Characterization Test: hops == 0 occurs in non-Fast-Path situations,
    such as when targeted retrieval finds zero new documents.
    """
    question = "Question with unconfident claim?"
    draft_answer = "Draft answer"
    claims_json = json.dumps({
        "claims": [{
            "text": "Weak assertion.",
            "answer": True,
            "depends_on": [],
        }]
    })

    # Return varied responses for self-consistency so c0 is low (e.g. 0.20)
    class LowConfLLM(MockLLM):
        def chat(self, messages, tracker, max_tokens=200, temperature=0.0, sample_idx=None, **kwargs):
            if sample_idx is not None:
                return f"Disagreement sample {sample_idx}"
            user_content = next((m["content"] for m in messages if m["role"] == "user"), "")
            if "JSON:" in user_content:
                return claims_json
            if "Claim: Weak assertion" in user_content:
                return "search query for weak assertion"
            return draft_answer

    llm = LowConfLLM()
    
    # Claim is UNRESOLVED (c0 = 0.2, eff = 0.2 < theta_gap 0.60)
    # This is NOT a Fast Path case!
    nli = MockNLI(default_score=(0.0, 0.0))
    
    # Initial pool contains Doc Seen
    initial_doc = {"title": "Doc Seen", "sentences": ["Some info."]}
    
    # Retriever returns the SAME document (no novel titles)
    retriever = MockRetriever(default=[initial_doc])
    
    cfg = {
        "retrieval": {"k": 3},
        "erag": {
            "mode": "active",
            "max_hops": 2,
            "theta_fast": 0.80,
            "theta_gap": 0.60,
            "theta_abs": 0.40,
            "sc_samples": 5,
        },
    }
    
    item = {"_id": "test_zero_hop", "question": question}
    result = run_erag(item, retriever, llm, nli, cfg, Tracker())
    
    # The claim had a gap (eff = 0.2 < 0.60), so Fast Path did NOT trigger
    # But targeted_retrieve found no new documents (new == []), breaking the loop at hop 0
    assert result["hops"] == 0, "hops == 0 despite NOT being a Fast Path exit"
    assert result["answer_conf"] < 0.60, "Claim was never fast-path supported"
