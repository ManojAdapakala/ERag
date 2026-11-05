"""Unit tests for Verified-Belief Episodic Memory module."""

import numpy as np
import pytest
from src.systems.memory import BeliefMemory


class MockEmbedder:
    """Mock embedder returning deterministic vectors based on string length and hashing."""

    def encode(self, texts):
        out = []
        for t in texts:
            val = float(len(t) % 10)
            vec = np.array([val, 1.0, 0.0], dtype="float32")
            vec = vec / (np.linalg.norm(vec) + 1e-9)
            out.append(vec)
        return np.stack(out)


class MockNLI:
    """Mock NLI returning configurable verify outputs."""

    def __init__(self, override_on_text=None):
        self.override_on_text = override_on_text or []

    def verify(self, claim, passages, theta_s=0.5, theta_c=0.5):
        if any(bad in claim for bad in self.override_on_text):
            return {
                "status": "contradicted",
                "ent": [],
                "con": [("Doc A", 0.95)],
                "unres": [],
            }
        return {
            "status": "supported",
            "ent": [("Doc B", 0.90)],
            "con": [],
            "unres": [],
        }


class MockClaim:
    def __init__(self, text, status="supported", eff=0.85):
        self.text = text
        self.status = status
        self.eff = eff


class MockGraph:
    def __init__(self, claims):
        self.claims = claims


@pytest.fixture
def base_cfg():
    return {
        "memory": {
            "enabled": True,
            "capacity": 3,
            "w_mem": 0.5,
            "theta_mem": 0.5,
            "theta_write": 0.7,
            "theta_surprise": 0.6,
            "decay": 0.9,
        },
        "nli": {
            "theta_s": 0.5,
            "theta_c": 0.5,
        },
    }


def test_memory_insert_and_read(base_cfg):
    emb = MockEmbedder()
    mem = BeliefMemory(emb, base_cfg)
    
    # Plant belief
    mem.insert("Paris is the capital of France.", conf=0.95)
    assert len(mem.items) == 1
    
    nli = MockNLI()
    passages, events = mem.read("What is Paris?", [{"title": "Doc", "sentences": ["Paris is in France."]}], nli, base_cfg)
    
    assert len(passages) == 1
    assert passages[0]["title"] == "memory"
    assert passages[0]["is_memory"] is True
    assert abs(passages[0]["weight"] - 0.5 * 0.95) < 1e-4
    assert len(events) == 0


def test_memory_override_deletion(base_cfg):
    """Test wrong-belief override: contradictory fresh evidence purges memory belief (E5)."""
    emb = MockEmbedder()
    mem = BeliefMemory(emb, base_cfg)
    
    # Plant false belief
    mem.insert("The 1998 festival was held in Leeds.", conf=0.9)
    assert len(mem.items) == 1
    
    nli = MockNLI(override_on_text=["Leeds"])
    fresh_pool = [{"title": "Festival History", "sentences": ["The 1998 festival took place in Reading, not Leeds."]}]
    
    passages, events = mem.read("Where was the 1998 festival?", fresh_pool, nli, base_cfg)
    
    # Contradicted belief must be dropped immediately and logged
    assert len(passages) == 0
    assert len(events) == 1
    assert events[0]["event"] == "overridden"
    assert "Leeds" in events[0]["belief"]
    assert len(mem.items) == 0  # Deleted from storage


def test_memory_write_surprise_gating(base_cfg):
    emb = MockEmbedder()
    mem = BeliefMemory(emb, base_cfg)
    
    graph = MockGraph([MockClaim("Claim A", "supported", 0.85)])
    
    # High confidence (c0 = 0.8 >= 0.6) and no contradiction -> not surprising -> should NOT write
    mem.write(graph, c0=0.8, seen_contra=False, cfg=base_cfg)
    assert len(mem.items) == 0
    
    # Low confidence (c0 = 0.4 < 0.6) -> surprising -> SHOULD write
    mem.write(graph, c0=0.4, seen_contra=False, cfg=base_cfg)
    assert len(mem.items) == 1
    assert mem.items[0]["text"] == "Claim A"


def test_memory_decay_and_eviction(base_cfg):
    emb = MockEmbedder()
    mem = BeliefMemory(emb, base_cfg)
    
    # Add beliefs up to capacity (3)
    mem.insert("Belief 1", conf=0.8)
    mem.insert("Belief 2", conf=0.8)
    mem.insert("Belief 3", conf=0.8)
    
    # Advance time without using Belief 1
    mem.t = 1
    # Trigger write with empty graph to execute decay
    mem.write(MockGraph([]), c0=0.1, seen_contra=False, cfg=base_cfg)
    
    # Unused beliefs should decay by factor 0.9
    for it in mem.items:
        assert abs(it["conf"] - 0.8 * 0.9) < 1e-4
        
    # Test capacity eviction: inserting 4th item when capacity is 3
    mem.insert("Belief 4", conf=0.9)
    mem.write(MockGraph([]), c0=0.1, seen_contra=False, cfg=base_cfg)
    assert len(mem.items) <= base_cfg["memory"]["capacity"]

