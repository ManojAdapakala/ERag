from unittest.mock import MagicMock
from src.epistemic.claims import _parse, extract_claims
from src.epistemic.confidence import self_consistency
from src.llm.client import Tracker


def test_parse_valid_json_claims():
    raw_json = """
    Here are the claims:
    ```json
    {
      "claims": [
        {"text": "Marie Curie was born in Warsaw.", "answer": false, "depends_on": []},
        {"text": "Warsaw was part of the Russian Empire.", "answer": false, "depends_on": [0]},
        {"text": "Marie Curie won two Nobel Prizes.", "answer": true, "depends_on": [0]}
      ]
    }
    ```
    """
    claims = _parse(raw_json, max_claims=5)
    assert len(claims) == 3
    assert claims[0]["text"] == "Marie Curie was born in Warsaw."
    assert claims[0]["answer_bearing"] is False
    assert claims[0]["depends_on"] == []

    assert claims[1]["depends_on"] == [0]
    assert claims[2]["answer_bearing"] is True
    assert claims[2]["depends_on"] == [0]


def test_parse_fallback_when_invalid():
    raw_invalid = "This is not valid json at all."
    claims = _parse(raw_invalid, max_claims=5)
    assert claims == []


def test_extract_claims_fallback_mechanism():
    llm = MagicMock()
    llm.chat.return_value = "Non-json output"
    tr = Tracker()

    claims = extract_claims(
        llm,
        question="Where was Marie Curie born?",
        answer="Warsaw",
        paras=[{"title": "Curie", "sentences": ["Curie was born in Warsaw."]}],
        tracker=tr,
    )
    assert len(claims) == 1
    assert claims[0]["answer_bearing"] is True
    assert claims[0]["fallback"] is True
    assert "Warsaw" in claims[0]["text"]


def test_self_consistency_confidence():
    llm = MagicMock()
    # 3 samples say "Paris", 2 samples say "Lyon"
    llm.chat.side_effect = ["Paris", "Paris", "Paris", "Lyon", "Lyon"]
    tr = Tracker()

    sc = self_consistency(
        llm,
        question="What is the capital of France?",
        paras=[{"title": "France", "sentences": ["Paris is capital."]}],
        draft="Paris",
        tracker=tr,
        k=5,
        temperature=0.7,
    )
    assert sc["majority"] == "paris"
    assert sc["conf_majority"] == 3 / 5  # 0.6
    assert sc["conf_draft_agreement"] == 3 / 5  # 0.6
    assert len(sc["samples"]) == 5
