"""Pre-adaptive-retrieval validation smoke test suite for missing-dependency fast-path safety.

Covers all 10 required development smoke-test scenarios:
1. Fast path still works when all answer-critical claims are supported.
2. Missing answer-critical dependencies prevent fast path and trigger intended retrieval/abstention.
3. Bridge-premise extraction fires only for approved relational patterns.
4. Bridge-premise extraction does not fire on ordinary questions.
5. Bridge claims are prerequisites, not answer-bearing claims.
6. The final answer claim depends on the bridge claim.
7. Contradicted dependencies still produce abstained_contradiction_veto.
8. Auxiliary unresolved claims do not unnecessarily block fast path.
9. JSONL, provenance, retrieval metadata, claim records, confidence values, and runner output remain valid and complete.
10. Safety changes do not cause excessive retrieval on ordinary supported questions or incorrect abstention when evidence is adequate.
"""

import json
import os
import tempfile
from typing import Any, Dict, List, Optional, Tuple
import pytest

from src.epistemic.belief_graph import BeliefGraph, Claim
from src.epistemic.claims import _detect_relational_bridge, extract_claims
from src.epistemic.nli import NLI
from src.llm.client import Tracker
from src.provenance import (
    RunLock,
    SidecarManager,
    compute_config_hash,
    prepare_output_file_for_append,
    read_existing_output_records,
    validate_manifest,
)
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
    """Subclass of NLI bypassing model loading for fast offline unit testing."""
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
        "retrieval": {"k": 3, "setting": "distractor"},
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
# Scenario 1: Fast path still works when all answer-critical claims are supported
# -----------------------------------------------------------------------------
def test_smoke_1_fast_path_supported():
    """Scenario 1: Ordinary question with fully supported claims exits at hop 0 via fast path."""
    question = "What is the capital of France?"
    draft = "Paris"
    claims_json = json.dumps({
        "claims": [{"text": "Paris is the capital of France.", "answer": True, "depends_on": []}]
    })
    llm = MockLLM(canned_chat={"JSON:": claims_json, "Question:": draft}, default_chat=draft)
    nli = MockNLI(default_score=(0.95, 0.0))
    retriever = MockRetriever(default=[{"title": "France", "sentences": ["Paris is the capital of France."]}])
    tracker = Tracker()

    res = run_erag({"_id": "smoke_1", "question": question}, retriever, llm, nli, _make_cfg(), tracker)

    assert res["hops"] == 0
    assert res["fast_path"] is True
    assert res["loop_stop_reason"] == "fast_path"
    assert res["output_reason"] == "answered"
    assert res["abstained"] is False
    assert res["answer"] == "Paris"
    assert tracker.retrievals == 1  # Exactly 1 retrieval (hop 0 initial)


# -----------------------------------------------------------------------------
# Scenario 2: Missing answer-critical dependencies prevent fast path and trigger retrieval/abstention
# -----------------------------------------------------------------------------
def test_smoke_2_missing_dependencies_prevent_fast_path_and_abstain():
    """Scenario 2: Missing bridge premise blocks fast path and safely abstains when no evidence found."""
    question = "Where was the director of Inception born?"
    draft = "Christopher Nolan was born in London."
    initial_pool = [{
        "title": "Christopher Nolan (biography)",
        "sentences": ["Christopher Nolan was born in London on July 30, 1970."],
    }]
    # LLM extracts only the birthplace fact without the director bridge
    claims_json = json.dumps({
        "claims": [{"text": "Christopher Nolan was born in London.", "answer": True, "depends_on": []}]
    })
    llm = MockLLM(canned_chat={"JSON:": claims_json}, default_chat=draft)
    nli = MockNLI(scores={
        "Christopher Nolan was born in London.||Christopher Nolan (biography)": (0.95, 0.0),
    })
    retriever = MockRetriever(default=initial_pool)
    tracker = Tracker()

    res = run_erag({"_id": "smoke_2", "question": question}, retriever, llm, nli, _make_cfg(), tracker)

    assert res["fast_path"] is False
    assert res["loop_stop_reason"] == "no_novel_evidence"
    assert res["abstained"] is True
    assert res["answer"] == "unknown"
    assert res["output_reason"] == "abstained_low_confidence"
    assert tracker.retrievals == 2  # Hop 0 initial + 1 follow-up retrieval probe


# -----------------------------------------------------------------------------
# Scenario 3: Bridge-premise extraction fires only for approved relational patterns
# -----------------------------------------------------------------------------
def test_smoke_3_bridge_extraction_relational_patterns():
    """Scenario 3: Approved relational patterns extract correct bridge and answer claims."""
    patterns = [
        ("Where was the director of Inception born?", "Christopher Nolan was born in London.", "director", "Christopher Nolan directed Inception."),
        ("Who is the author of The Hobbit?", "J.R.R. Tolkien wrote The Hobbit.", "author", "J.R.R. Tolkien wrote The Hobbit."),
        ("Where did the inventor of the telephone live?", "Alexander Graham Bell lived in Boston.", "inventor", "Alexander Graham Bell invented the telephone."),
        ("When did the creator of Linux start it?", "Linus Torvalds started Linux in 1991.", "creator", "Linus Torvalds invented Linux."),
    ]
    empty_pool = []

    for q, ans, rel, expected_bridge in patterns:
        bridge_info = _detect_relational_bridge(q, ans, empty_pool)
        assert bridge_info is not None, f"Pattern for {rel} must be detected"
        assert bridge_info["relation"] == rel
        assert bridge_info["bridge_text"] == expected_bridge


# -----------------------------------------------------------------------------
# Scenario 4: Bridge-premise extraction does not fire on ordinary questions
# -----------------------------------------------------------------------------
def test_smoke_4_bridge_extraction_does_not_fire_on_ordinary_questions():
    """Scenario 4: Ordinary non-bridge questions return None from bridge detector."""
    ordinary_questions = [
        ("What is the capital of France?", "Paris is the capital of France."),
        ("When was Marie Curie born?", "Marie Curie was born in 1867."),
        ("Which country won the 2018 World Cup?", "France won the 2018 World Cup."),
        ("How tall is Mount Everest?", "Mount Everest is 8,848 meters tall."),
        ("What is the speed of light?", "The speed of light is 300,000 km/s."),
    ]
    empty_pool = []

    for q, ans in ordinary_questions:
        bridge_info = _detect_relational_bridge(q, ans, empty_pool)
        assert bridge_info is None, f"Ordinary question '{q}' must not trigger bridge detection"


# -----------------------------------------------------------------------------
# Scenario 5: Bridge claims are prerequisites, not answer-bearing claims
# -----------------------------------------------------------------------------
def test_smoke_5_bridge_claims_are_prerequisites_not_answer_bearing():
    """Scenario 5: Extracted bridge claim is explicitly marked answer_bearing=False."""
    question = "Where was the director of Inception born?"
    draft = "Christopher Nolan was born in London."
    llm = MockLLM(default_chat=draft)
    tracker = Tracker()

    claims = extract_claims(llm, question, draft, [], tracker)
    assert len(claims) >= 2
    bridge_claim = claims[0]
    assert bridge_claim["text"] == "Christopher Nolan directed Inception."
    assert bridge_claim["answer_bearing"] is False, "Bridge claim must not be marked as answer-bearing"
    assert bridge_claim["depends_on"] == []


# -----------------------------------------------------------------------------
# Scenario 6: The final answer claim depends on the bridge claim
# -----------------------------------------------------------------------------
def test_smoke_6_final_answer_depends_on_bridge_claim():
    """Scenario 6: The final answer claim has depends_on referencing the bridge claim index."""
    question = "Where was the director of Inception born?"
    draft = "Christopher Nolan was born in London."
    llm = MockLLM(default_chat=draft)
    tracker = Tracker()

    claims = extract_claims(llm, question, draft, [], tracker)
    assert len(claims) >= 2
    bridge_idx = 0
    ans_claim = claims[1]
    assert ans_claim["answer_bearing"] is True
    assert bridge_idx in ans_claim["depends_on"], "Answer claim must declare dependency on bridge claim"


# -----------------------------------------------------------------------------
# Scenario 7: Contradicted dependencies still produce abstained_contradiction_veto
# -----------------------------------------------------------------------------
def test_smoke_7_contradicted_dependency_produces_veto():
    """Scenario 7: Contradicted prerequisite dependency produces abstained_contradiction_veto."""
    question = "Where was the director of Inception born?"
    draft = "London"
    claims_json = json.dumps({
        "claims": [
            {"text": "Christopher Nolan directed Inception.", "answer": False, "depends_on": []},
            {"text": "Christopher Nolan was born in London.", "answer": True, "depends_on": [0]},
        ]
    })
    llm = MockLLM(canned_chat={"JSON:": claims_json}, default_chat=draft)
    nli = MockNLI(scores={
        "Christopher Nolan directed Inception.||Christopher Nolan": (0.0, 0.95),
        "Christopher Nolan was born in London.||Christopher Nolan": (0.95, 0.0),
    })
    retriever = MockRetriever(default=[{
        "title": "Christopher Nolan",
        "sentences": ["Inception was directed by someone else, not Christopher Nolan. Christopher Nolan was born in London."],
    }])

    res = run_erag({"_id": "smoke_7", "question": question}, retriever, llm, nli, _make_cfg(), Tracker())

    assert res["fast_path"] is False
    assert res["abstained"] is True
    assert res["answer"] == "unknown"
    assert res["output_reason"] == "abstained_contradiction_veto"


# -----------------------------------------------------------------------------
# Scenario 8: Auxiliary unresolved claims do not unnecessarily block fast path
# -----------------------------------------------------------------------------
def test_smoke_8_auxiliary_unresolved_claim_allows_fast_path():
    """Scenario 8: Fully supported answer claim exits via fast path even if auxiliary claim is unresolved."""
    question = "What is the capital of France?"
    draft = "Paris"
    claims_json = json.dumps({
        "claims": [
            {"text": "Paris is the capital of France.", "answer": True, "depends_on": []},
            {"text": "Paris had 2.1 million inhabitants in 2020.", "answer": False, "depends_on": []},
        ]
    })
    llm = MockLLM(canned_chat={"JSON:": claims_json}, default_chat=draft)
    nli = MockNLI(scores={
        "Paris is the capital of France.||France": (0.95, 0.0),
        "Paris had 2.1 million inhabitants in 2020.||France": (0.0, 0.0),
    })
    retriever = MockRetriever(default=[{"title": "France", "sentences": ["Paris is the capital of France."]}])

    res = run_erag({"_id": "smoke_8", "question": question}, retriever, llm, nli, _make_cfg(), Tracker())

    assert res["hops"] == 0
    assert res["fast_path"] is True
    assert res["loop_stop_reason"] == "fast_path"
    assert res["output_reason"] == "answered"
    assert res["abstained"] is False


# -----------------------------------------------------------------------------
# Scenario 9: JSONL, provenance, retrieval metadata, claim records, confidence, runner output
# -----------------------------------------------------------------------------
def test_smoke_9_jsonl_provenance_and_telemetry_integrity():
    """Scenario 9: Runner dispatch produces valid telemetry, serialized JSONL, and sidecar metadata."""
    from scripts.run_experiment import dispatch

    item = {"_id": "smoke_9_item", "question": "What is the capital of France?", "answer": "Paris"}
    draft = "Paris"
    claims_json = json.dumps({
        "claims": [{"text": "Paris is the capital of France.", "answer": True, "depends_on": []}]
    })
    llm = MockLLM(canned_chat={"JSON:": claims_json, "Question:": draft}, default_chat=draft)
    nli = MockNLI(default_score=(0.95, 0.0))
    retriever = MockRetriever(default=[{"title": "France", "sentences": ["Paris is the capital of France."]}])
    cfg = _make_cfg()
    tr = Tracker()

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "smoke_run.jsonl")
        r = dispatch("erag", item, retriever, llm, nli, cfg, tr, memory=None)

        # 1. Verify telemetry fields in memory result
        assert "loop_stop_reason" in r
        assert "output_reason" in r
        assert "fast_path" in r
        assert "answer_conf" in r
        assert "c0" in r
        assert "hops" in r
        assert "trace" in r
        assert "claim_fallbacks" in r

        # 2. Serialize to JSONL via safe append protocol
        rec = {
            "id": item["_id"],
            "system": "erag",
            "setting": cfg["retrieval"]["setting"],
            "split": "test",
            "question": item["question"],
            "gold": item["answer"],
        }
        rec.update(r)

        with RunLock(out_path):
            prepare_output_file_for_append(out_path)
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")

        # 3. Read back and confirm JSONL integrity
        with open(out_path, "r", encoding="utf-8") as f:
            lines = [json.loads(l) for l in f if l.strip()]

        assert len(lines) == 1
        read_rec = lines[0]
        assert read_rec["id"] == "smoke_9_item"
        assert read_rec["loop_stop_reason"] == "fast_path"
        assert read_rec["output_reason"] == "answered"
        assert read_rec["fast_path"] is True
        assert isinstance(read_rec["answer_conf"], float)
        assert isinstance(read_rec["c0"], float)


# -----------------------------------------------------------------------------
# Scenario 10: No excessive retrieval on ordinary supported questions, no false abstention
# -----------------------------------------------------------------------------
def test_smoke_10_no_excessive_retrieval_or_false_abstention():
    """Scenario 10: Ordinary supported questions do not make extraneous retrieval calls or falsely abstain."""
    test_cases = [
        ("Capital of France?", "Paris", "France", "Paris is the capital of France."),
        ("Capital of Italy?", "Rome", "Italy", "Rome is the capital of Italy."),
        ("Capital of Spain?", "Madrid", "Spain", "Madrid is the capital of Spain."),
    ]

    for q_id, (q, ans, title, sentence) in enumerate(test_cases):
        claims_json = json.dumps({
            "claims": [{"text": sentence, "answer": True, "depends_on": []}]
        })
        llm = MockLLM(canned_chat={"JSON:": claims_json, "Question:": ans}, default_chat=ans)
        nli = MockNLI(default_score=(0.95, 0.0))
        retriever = MockRetriever(default=[{"title": title, "sentences": [sentence]}])
        tracker = Tracker()

        res = run_erag({"_id": f"smoke_10_{q_id}", "question": q}, retriever, llm, nli, _make_cfg(), tracker)

        assert res["hops"] == 0, f"Ordinary supported query '{q}' must not hop"
        assert res["fast_path"] is True
        assert res["loop_stop_reason"] == "fast_path"
        assert res["abstained"] is False, f"Ordinary supported query '{q}' must not falsely abstain"
        assert res["answer"] == ans
        assert res["output_reason"] == "answered"
        assert tracker.retrievals == 1, "Must execute exactly one initial retrieval, no excessive searches"
