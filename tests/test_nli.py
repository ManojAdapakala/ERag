import pytest
from src.epistemic.nli import NLI

class DummyNLI(NLI):
    def __init__(self):
        # Override init to not load model weights during fast unit test
        self.memo = {}
        self.i_ent, self.i_con = 0, 2
        
    def best(self, claim: str, passage: dict):
        # Return high contradiction and zero entailment to test entity filter
        return 0.0, 0.99


def test_extract_subject_entities():
    c1 = "The director of the romantic comedy \"Big Stone Gap\" is Adriana Trigiani."
    ents1 = NLI._extract_subject_entities(c1)
    assert "Big Stone Gap" in ents1

    c2 = "The 2014 S/S album is the debut album of the South Korean boy group WINNER."
    ents2 = NLI._extract_subject_entities(c2)
    assert "2014 S/S" in ents2

    c3 = "Terry Richardson was born on August 14, 1965."
    ents3 = NLI._extract_subject_entities(c3)
    assert "Terry Richardson" in ents3

    c4 = "Both Ed Wood and Scott Derrickson share the same nationality."
    ents4 = NLI._extract_subject_entities(c4)
    assert "Ed Wood" in ents4
    assert "Scott Derrickson" in ents4


def test_entity_grounded_contradiction_filter():
    nli = DummyNLI()
    claim = "The director of the romantic comedy \"Big Stone Gap\" is Adriana Trigiani."
    
    # Topic-disjoint passage: mentions romantic and comedy and director, but neither entity
    disjoint_passage = {
        "title": "I Love NY (2015 film)",
        "sentences": ["I Love NY is a romantic comedy directed by Radhika Rao and Vinay Sapru."]
    }
    ev = nli.verify(claim, [disjoint_passage])
    assert ev["status"] == "unresolved"
    assert len(ev["con"]) == 0

    # Grounded passage: explicitly mentions the subject entity Big Stone Gap
    grounded_passage = {
        "title": "Big Stone Gap (film)",
        "sentences": ["Big Stone Gap is a film directed by someone else entirely."]
    }
    ev2 = nli.verify(claim, [grounded_passage])
    assert ev2["status"] == "contradicted"
    assert len(ev2["con"]) == 1
