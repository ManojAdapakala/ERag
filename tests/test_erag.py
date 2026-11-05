from unittest.mock import MagicMock
from src.llm.client import Tracker
from src.systems.erag import build_graph, conf_value, propose_queries, run_erag, targeted_retrieve


def test_propose_queries():
    llm = MagicMock()
    llm.chat.return_value = "Where was Marie Curie born?\nMarie Curie birthplace university\nWarsaw Poland history"
    tr = Tracker()

    queries = propose_queries(llm, "Marie Curie was born in Warsaw.", tr, n=3)
    assert len(queries) == 3
    assert queries[0] == "Where was Marie Curie born?"


def test_conf_value():
    sc = {"conf_majority": 0.8, "conf_draft_agreement": 0.6}
    assert conf_value(sc, {"erag": {"confidence_def": "majority"}}) == 0.8
    assert conf_value(sc, {"erag": {"confidence_def": "draft_agreement"}}) == 0.6


def test_targeted_retrieve():
    llm = MagicMock()
    llm.chat.return_value = "Query 1\nQuery 2"
    retriever = MagicMock()
    # Query 1 returns already seen doc; Query 2 returns novel doc
    retriever.search.side_effect = [
        [{"title": "Doc Seen", "sentences": ["Old."]}],
        [{"title": "Doc Novel", "sentences": ["New info."]}],
    ]

    claim = MagicMock()
    claim.text = "Sample claim text"
    claim.eff = 0.4
    pool = [{"title": "Doc Seen", "sentences": ["Old."]}]
    tr = Tracker()

    novel = targeted_retrieve(llm, retriever, claim, pool, k=3, ablate=set(), tracker=tr)
    assert len(novel) == 1
    assert novel[0]["title"] == "Doc Novel"
    assert tr.probe_searches == 2
    assert tr.retrievals == 1


def test_run_erag_observational_mode():
    item = {
        "_id": "test_obs_1",
        "question": "What is the capital of France?",
        "context": [["Paris", ["Paris is the capital of France."]]],
        "supporting_facts": [["Paris", 0]],
    }
    retriever = MagicMock()
    retriever.search.return_value = [{"title": "Paris", "sentences": ["Paris is the capital of France."], "text": "Paris. Paris is the capital of France."}]

    llm = MagicMock()
    llm.chat.side_effect = [
        "Paris",        # Hop 0 draft
        "Paris", "Paris", "Paris", "Paris", "Paris",  # 5 self-consistency samples
        '{"claims": [{"text": "Paris is the capital.", "answer": true, "depends_on": []}]}',  # Claim extraction
    ]

    nli = MagicMock()
    nli.verify.return_value = {"status": "supported", "conflict": False, "sup": [("Paris", 0.9)], "con": []}

    cfg = {
        "retrieval": {"k": 3},
        "nli": {"theta_s": 0.5, "theta_c": 0.5},
        "erag": {
            "mode": "observe",
            "max_hops": 2,
            "sc_samples": 5,
            "sc_temperature": 0.7,
            "theta_fast": 0.8,
            "theta_gap": 0.6,
            "theta_abs": 0.4,
            "max_claims": 5,
            "pool_cap": 12,
        },
    }
    tr = Tracker()

    out = run_erag(item, retriever, llm, nli, cfg, tr)

    assert out["answer"] == "Paris"
    assert out["draft"] == "Paris"
    assert out["hops"] == 0  # Observe mode stops at hop 0
    assert out["abstained"] is False
    assert out["answer_conf"] > 0.5

