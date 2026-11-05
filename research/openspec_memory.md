# OpenSpec Proposal: Verified-Belief Memory Module

**Component:** `src/systems/memory.py`  
**Author:** Senior AI/ML Engineer & RAG Research Scientist  
**Date:** 2026-11-05  
**Status:** PROPOSED (Awaiting Human Approval Before Implementation)  
**Primary Reference:** *E-RAG Implementation Guide: End to End* (§12, pp. 33–35)  

---

## 1. RESEARCH MOTIVATION & OBJECTIVE

In sequential multi-question settings (e.g. question streams where later questions reuse facts established in earlier questions), an agent should maintain a persistent memory of verified beliefs. However, standard RAG memories often preserve hallucinations or stale facts indefinitely.

The **Verified-Belief Memory** module implements an episodic belief store with:
1. **Strict verification gating:** Only beliefs supported by external NLI evidence ($c.status == \text{"supported"}$) with high effective confidence ($c.eff \ge \theta_{write}$) are written.
2. **Surprise-gated writing:** Memories are written only when the initial confidence was low ($c_0 < \theta_{surprise}$) or contradiction was resolved during retrieval, avoiding memory clutter from trivial parametric knowledge.
3. **Contradiction-driven deletion (override):** Stored beliefs contradicted by fresh evidence retrieved for a new question are automatically deleted and logged in `mem_events`.
4. **Memory decay & capacity bounds:** Unused beliefs decay over time ($decay = 0.98$), and least-recently-used beliefs are evicted when capacity ($200$) is reached.
5. **Down-weighting:** Memory passages enter NLI verification with a reduced weight ($w_{mem} = 0.5$) relative to fresh external text ($1.0$).

---

## 2. CONFIGURATION PARAMETERS

```yaml
memory:
  enabled: false
  capacity: 200
  w_mem: 0.5             # Weight applied to memory evidence during NLI verification
  theta_mem: 0.75        # Embedding similarity threshold to retrieve stored beliefs
  theta_write: 0.7       # Minimum confidence to write a verified belief to memory
  theta_surprise: 0.6    # Write trigger: initial confidence below this threshold
  decay: 0.98            # Multiplicative decay per question for unused beliefs
```

---

## 3. INTERFACE SPECIFICATION

### 3.1 Class Architecture
```python
class BeliefMemory:
    def __init__(self, embedder: Embedder, cfg: dict):
        ...
    def insert(self, text: str, conf: float = 0.9):
        # Used for planting synthetic wrong beliefs in Experiment E5
        ...
    def read(self, question: str, fresh_pool: list, nli: NLI, cfg: dict) -> Tuple[list, list]:
        # Returns (memory_passages, mem_events)
        # Contradicted beliefs are purged immediately
        ...
    def write(self, graph: BeliefGraph, c0: float, seen_contra: bool, cfg: dict):
        # Writes supported, surprising claims; decays unused beliefs; evicts LRU over capacity
        ...
```

---

## 4. EXPERIMENTAL VALIDATION (E5)

1. **Stream Evaluation:** Run 30 questions in 5 entity-sharing clusters (`data/manifests/stream.jsonl`) with memory disabled vs enabled.
2. **Wrong-Belief Planted Test:** Plant 5 false beliefs into memory (`data/manifests/planted.json`). Verify that when fresh evidence contradicts the false belief, the belief is deleted (`"event": "overridden"`) and does not corrupt subsequent answers.

