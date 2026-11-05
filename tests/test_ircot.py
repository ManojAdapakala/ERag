from unittest.mock import MagicMock
from src.llm.client import Tracker
from src.systems.ircot import next_step, run_ircot


def test_next_step():
    llm = MagicMock()
    llm.chat.return_value = "Scott Derrickson was born in Denver, Colorado.\nExtra line."
    tr = Tracker()

    sentence = next_step(
        llm,
        question="Where was Scott Derrickson born?",
        pool=[{"title": "Doc 1", "sentences": ["Scott is a director."], "text": "Doc 1. Scott is a director."}],
        steps=[],
        tracker=tr,
    )
    assert sentence == "Scott Derrickson was born in Denver, Colorado."


def test_run_ircot_terminates_on_answer_condition():
    retriever = MagicMock()
    retriever.search.side_effect = [
        [{"title": "Doc A", "sentences": ["Paragraph A."], "text": "Doc A. Paragraph A."}],
        [{"title": "Doc B", "sentences": ["Paragraph B."], "text": "Doc B. Paragraph B."}],
    ]

    llm = MagicMock()
    # Step 1: intermediate reasoning
    # Step 2: "So the answer is: Paris." -> should trigger early break!
    # Step 3: Final reader answer
    llm.chat.side_effect = [
        "First step of reasoning.",
        "So the answer is: Paris.",
        "Paris",
    ]

    item = {
        "_id": "test_ircot_1",
        "question": "What is the capital?",
        "context": [["Doc A", ["Paragraph A."]], ["Doc B", ["Paragraph B."]]],
        "supporting_facts": [["Doc A", 0]],
    }
    cfg = {"retrieval": {"k": 3}, "ircot": {"max_steps": 4, "pool_cap": 12}}
    tr = Tracker()

    out = run_ircot(item, retriever, llm, cfg, tr)

    assert out["answer"] == "Paris"
    assert out["hops"] == 2  # Terminated at step 2
    assert len(out["steps"]) == 2
    assert "Doc A" in out["pool_titles"]
    assert "Doc B" in out["pool_titles"]

