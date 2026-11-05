import os
import tempfile
from unittest.mock import MagicMock
from src.llm.client import Tracker
from src.systems.standard import run_closed_book, run_standard
from scripts.run_experiment import summarize


def test_run_standard():
    item = {
        "_id": "q1",
        "question": "What is Paris?",
        "context": [["Paris", ["Paris is the capital of France."]]],
        "supporting_facts": [["Paris", 0]],
    }
    retriever = MagicMock()
    retriever.search.return_value = [{"title": "Paris", "sentences": ["Paris is the capital of France."], "text": "Paris. Paris is the capital of France."}]

    llm = MagicMock()
    llm.chat.return_value = "Capital of France"

    cfg = {"retrieval": {"k": 3}}
    tr = Tracker()

    out = run_standard(item, retriever, llm, cfg, tr)

    assert out["answer"] == "Capital of France"
    assert out["pool_titles"] == ["Paris"]
    assert out["hops"] == 0
    assert tr.retrievals == 1
    assert llm.chat.call_count == 1


def test_run_closed_book():
    item = {"_id": "q2", "question": "What is the speed of light?"}
    retriever = MagicMock()
    llm = MagicMock()
    llm.chat.return_value = "299,792,458 m/s"

    cfg = {"retrieval": {"k": 3}}
    tr = Tracker()

    out = run_closed_book(item, retriever, llm, cfg, tr)

    assert out["answer"] == "299,792,458 m/s"
    assert out["pool_titles"] == []
    assert out["hops"] == 0
    assert tr.retrievals == 0
    assert retriever.search.call_count == 0


def test_summarize_jsonl():
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
        f.write('{"id": "1", "em": 1.0, "f1": 1.0, "unknown": false, "recall": 1.0, "failed": false, "llm_calls": 1, "retrievals": 1, "in_tokens": 100, "out_tokens": 10, "wall_latency": 0.5}\n')
        f.write('{"id": "2", "em": 0.0, "f1": 0.5, "unknown": false, "recall": 0.5, "failed": false, "llm_calls": 1, "retrievals": 1, "in_tokens": 100, "out_tokens": 10, "wall_latency": 0.5}\n')
        f.write('{"id": "3", "em": 0.0, "f1": 0.0, "unknown": true, "recall": 0.0, "failed": false, "llm_calls": 1, "retrievals": 1, "in_tokens": 100, "out_tokens": 10, "wall_latency": 0.5}\n')
        tmp_name = f.name

    try:
        s = summarize(tmp_name)
        assert s["n"] == 3
        assert s["failed"] == 0
        assert abs(s["em"] - 1/3) < 1e-5
        assert abs(s["f1"] - 0.5) < 1e-5
        assert abs(s["coverage"] - 2/3) < 1e-5
        assert abs(s["selective_em"] - 0.5) < 1e-5
    finally:
        os.remove(tmp_name)

