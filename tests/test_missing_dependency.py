"""Focused offline tests for missing-dependency and fast-path safety.

Covers:
1. Missing prerequisite for a multi-hop answer
2. Unresolved prerequisite
3. Unsupported prerequisite
4. Contradicted prerequisite
5. Auxiliary unresolved claim that is not answer-critical (allows fast path)
6. Fast path with all required claims supported
7. Fast path with a missing/absent dependency
8. Composite answer with one unsupported component
"""

import json
from typing import Any, Dict, List, Optional, Tuple
import pytest

from src.epistemic.belief_graph import BeliefGraph, Claim
from src.epistemic.nli import NLI
from src.llm.client import Tracker
from src.systems.erag import build_graph, run_erag


class MockRetriever:
    """Configurable mock retriever returning predefined passages."""
    def __init__(
        self,
        responses: Optional[Dict[str, List[Dict[str, Any]]]] = None,
        default: Optional[List[Dict[str, Any]]] = None,
    ):
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
        user_content = next((m["content"] for m in messages if m["role"] == "user"), "")
        for k, v in self.canned_chat.items():
            if k in user_content:
                return v
        return self.default_chat


class MockNLI(NLI):
    """Subclass of NLI bypassing model loading for fast unit testing."""
    def __init__(
        self,
        scores: Optional[Dict[str, Tuple[float, float]]] = None,
        default_score: Tuple[float, float] = (0.0, 0.0),
    ):
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


def _make_cfg() -> Dict[str, Any]:
    return {
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


# -----------------------------------------------------------------------------
# 1. Missing Prerequisite for a Multi-Hop Answer
# -----------------------------------------------------------------------------

def test_missing_prerequisite_multihop_blocks_fast_path():
    """Multi-hop question missing bridge relation must block fast path and continue retrieval."""
    question = "Where was the director of Inception born?"
    draft = "Christopher Nolan was born in London."
    initial_pool = [{
        "title": "Christopher Nolan (biography)",
        "sentences": ["Christopher Nolan was born in London."],
    }]
    bridge_doc = [{
        "title": "Inception (film)",
        "sentences": ["Inception was directed by Christopher Nolan."],
    }]

    # When question is searched at hop 0, retriever returns initial_pool;
    # on follow-up bridge retrieval, retriever returns bridge_doc.
    retriever = MockRetriever(
        responses={
            question: initial_pool,
        },
        default=bridge_doc,
    )
    llm = MockLLM(default_chat=draft)
    nli = MockNLI(
        scores={
            "Christopher Nolan was born in London.||Christopher Nolan (biography)": (0.95, 0.0),
            "Christopher Nolan directed Inception.||Inception (film)": (0.95, 0.0),
        },
        default_score=(0.0, 0.0),
    )

    res = run_erag({"_id": "test_mhop_1", "question": question}, retriever, llm, nli, _make_cfg(), Tracker())
    assert res["fast_path"] is False, "Must not take fast path when bridge premise is missing"
    assert res["hops"] > 0, "Retrieval must continue to resolve the bridge entity"


# -----------------------------------------------------------------------------
# 2. Unresolved Prerequisite
# -----------------------------------------------------------------------------

def test_unresolved_prerequisite_blocks_fast_path():
    """When an answer claim depends on an unresolved premise, fast path is blocked."""
    question = "Where was the author of the manuscript born?"
    draft = "London"
    claims_json = json.dumps({
        "claims": [
            {"text": "John Doe wrote the manuscript.", "answer": False, "depends_on": []},
            {"text": "John Doe was born in London.", "answer": True, "depends_on": [0]},
        ]
    })
    llm = MockLLM(canned_chat={"JSON:": claims_json}, default_chat=draft)
    # Claim 0 is unresolved (0, 0); Claim 1 is supported (0.95, 0.0)
    nli = MockNLI(scores={
        "John Doe wrote the manuscript.||Biography": (0.0, 0.0),
        "John Doe was born in London.||Biography": (0.95, 0.0),
    })
    retriever = MockRetriever(default=[{"title": "Biography", "sentences": ["John Doe was born in London."]}])

    res = run_erag({"_id": "test_unres_1", "question": question}, retriever, llm, nli, _make_cfg(), Tracker())
    assert res["fast_path"] is False
    assert res["loop_stop_reason"] != "fast_path"


# -----------------------------------------------------------------------------
# 3. Unsupported Prerequisite (Eff < theta_fast)
# -----------------------------------------------------------------------------

def test_unsupported_prerequisite_blocks_fast_path():
    """When a prerequisite has weak support (below theta_fast), fast path is blocked."""
    question = "Capital of ancient Gaul?"
    draft = "Lyon"
    claims_json = json.dumps({
        "claims": [
            {"text": "Lugdunum was the capital of Gaul.", "answer": False, "depends_on": []},
            {"text": "Lugdunum corresponds to modern Lyon.", "answer": True, "depends_on": [0]},
        ]
    })
    llm = MockLLM(canned_chat={"JSON:": claims_json}, default_chat=draft)
    # Claim 0 has weak support giving eff < 0.80; Claim 1 has strong support
    nli = MockNLI(scores={
        "Lugdunum was the capital of Gaul.||Gaul History": (0.40, 0.0),
        "Lugdunum corresponds to modern Lyon.||Gaul History": (0.95, 0.0),
    })
    retriever = MockRetriever(default=[{"title": "Gaul History", "sentences": ["Historical notes on ancient Gaul."]}])

    res = run_erag({"_id": "test_unsup_1", "question": question}, retriever, llm, nli, _make_cfg(), Tracker())
    assert res["fast_path"] is False


# -----------------------------------------------------------------------------
# 4. Contradicted Prerequisite
# -----------------------------------------------------------------------------

def test_contradicted_prerequisite_blocks_fast_path_and_triggers_veto():
    """When a required prerequisite is contradicted, fast path is blocked and contradiction veto triggers."""
    question = "Where was the director of Inception born?"
    draft = "London"
    claims_json = json.dumps({
        "claims": [
            {"text": "Christopher Nolan directed Inception.", "answer": False, "depends_on": []},
            {"text": "Christopher Nolan was born in London.", "answer": True, "depends_on": [0]},
        ]
    })
    llm = MockLLM(canned_chat={"JSON:": claims_json}, default_chat=draft)
    # Nolan directing Inception is refuted
    nli = MockNLI(scores={
        "Christopher Nolan directed Inception.||Christopher Nolan": (0.0, 0.95),
        "Christopher Nolan was born in London.||Christopher Nolan": (0.95, 0.0),
    })
    retriever = MockRetriever(default=[{"title": "Christopher Nolan", "sentences": ["Inception was directed by someone else, not Christopher Nolan. Christopher Nolan was born in London."]}])

    res = run_erag({"_id": "test_contra_dep", "question": question}, retriever, llm, nli, _make_cfg(), Tracker())
    assert res["fast_path"] is False
    assert res["abstained"] is True
    assert res["output_reason"] == "abstained_contradiction_veto"


# -----------------------------------------------------------------------------
# 5. Auxiliary Unresolved Claim (Not Answer-Critical) Allows Fast Path
# -----------------------------------------------------------------------------

def test_auxiliary_unresolved_claim_does_not_block_fast_path():
    """An unresolved auxiliary claim that is NOT answer-critical must NOT block fast path."""
    question = "Capital of France?"
    draft = "Paris"
    claims_json = json.dumps({
        "claims": [
            {"text": "Paris is the capital of France.", "answer": True, "depends_on": []},
            {"text": "Paris was founded in the 3rd century BC.", "answer": False, "depends_on": []},
        ]
    })
    llm = MockLLM(canned_chat={"JSON:": claims_json}, default_chat=draft)
    # Claim 0 is strongly supported; Claim 1 is completely unresolved
    nli = MockNLI(scores={
        "Paris is the capital of France.||Paris Doc": (0.95, 0.0),
        "Paris was founded in the 3rd century BC.||Paris Doc": (0.0, 0.0),
    })
    retriever = MockRetriever(default=[{"title": "Paris Doc", "sentences": ["Paris is the capital of France."]}])

    res = run_erag({"_id": "test_aux_unres", "question": question}, retriever, llm, nli, _make_cfg(), Tracker())
    assert res["fast_path"] is True
    assert res["loop_stop_reason"] == "fast_path"
    assert res["output_reason"] == "answered"
    assert res["abstained"] is False


# -----------------------------------------------------------------------------
# 6. Fast Path with All Required Claims Supported
# -----------------------------------------------------------------------------

def test_fast_path_all_required_claims_supported():
    """When all answer-critical claims and dependencies are supported >= theta_fast, fast path exits at Hop 0."""
    question = "Capital of France?"
    draft = "Paris"
    claims_json = json.dumps({
        "claims": [
            {"text": "Paris is in France.", "answer": False, "depends_on": []},
            {"text": "Paris is the capital of France.", "answer": True, "depends_on": [0]},
        ]
    })
    llm = MockLLM(canned_chat={"JSON:": claims_json}, default_chat=draft)
    nli = MockNLI(scores={
        "Paris is in France.||Paris Doc": (0.95, 0.0),
        "Paris is the capital of France.||Paris Doc": (0.95, 0.0),
    })
    retriever = MockRetriever(default=[{"title": "Paris Doc", "sentences": ["Paris is in France and is the capital."]}])

    res = run_erag({"_id": "test_fast_all_sup", "question": question}, retriever, llm, nli, _make_cfg(), Tracker())
    assert res["hops"] == 0
    assert res["fast_path"] is True
    assert res["loop_stop_reason"] == "fast_path"
    assert res["output_reason"] == "answered"


# -----------------------------------------------------------------------------
# 7. Fast Path with Missing / Absent Dependency
# -----------------------------------------------------------------------------

def test_missing_absent_dependency_blocks_fast_path():
    """A claim declaring an absent dependency index outside graph bounds must block fast path."""
    question = "Who invented the telephone?"
    draft = "Alexander Graham Bell"
    # Claim 0 declares depends_on: [99] which does not exist in claims
    claims_json = json.dumps({
        "claims": [
            {"text": "Alexander Graham Bell was an inventor.", "answer": False, "depends_on": []},
            {"text": "Alexander Graham Bell invented the telephone.", "answer": True, "depends_on": [99]},
        ]
    })
    llm = MockLLM(canned_chat={"JSON:": claims_json}, default_chat=draft)
    nli = MockNLI(default_score=(0.95, 0.0))
    retriever = MockRetriever(default=[{"title": "Bell Bio", "sentences": ["Bell was an inventor who invented the telephone."]}])

    res = run_erag({"_id": "test_absent_dep", "question": question}, retriever, llm, nli, _make_cfg(), Tracker())
    assert res["fast_path"] is False
    assert res["loop_stop_reason"] != "fast_path"


# -----------------------------------------------------------------------------
# 8. Composite Answer with One Unsupported Component
# -----------------------------------------------------------------------------

def test_composite_answer_one_unsupported_component_blocks_fast_path():
    """When a composite answer has one supported and one unsupported claim, fast path is blocked."""
    question = "What are the two official languages of Canada?"
    draft = "English and French"
    claims_json = json.dumps({
        "claims": [
            {"text": "English is an official language of Canada.", "answer": True, "depends_on": []},
            {"text": "French is an official language of Canada.", "answer": True, "depends_on": []},
        ]
    })
    llm = MockLLM(canned_chat={"JSON:": claims_json}, default_chat=draft)
    # English is supported; French is unresolved
    nli = MockNLI(scores={
        "English is an official language of Canada.||Canada Info": (0.95, 0.0),
        "French is an official language of Canada.||Canada Info": (0.0, 0.0),
    })
    retriever = MockRetriever(default=[{"title": "Canada Info", "sentences": ["English is widely spoken in Canada."]}])

    res = run_erag({"_id": "test_composite_unsup", "question": question}, retriever, llm, nli, _make_cfg(), Tracker())
    assert res["fast_path"] is False
    assert res["loop_stop_reason"] != "fast_path"
    assert res["abstained"] is True
    assert res["output_reason"] == "abstained_low_confidence"
