# OpenSpec Proposal: System C (E-RAG Adaptive Epistemic Loop)

**Component:** `src/systems/erag.py`  
**Author:** Senior AI/ML Engineer & RAG Research Scientist  
**Date:** 2026-11-05  
**Status:** PROPOSED (Awaiting Human Approval Before Implementation)  
**Primary Reference:** *E-RAG Implementation Guide: End to End* (§11, pp. 28–32)  

---

## 1. RESEARCH MOTIVATION & OBJECTIVE

Standard RAG (System A) performs single-pass retrieve-then-generate, failing on multi-hop questions when the initial top-$k$ context is incomplete. Iterative RAG (like IRCoT, System B) continues retrieving up to a fixed step cap without an explicit model of belief, contradiction, or confidence calibration.

**E-RAG (System C)** treats generation as an explicit sequence of epistemic belief updates. It maintains a structured belief DAG of atomic claims, estimates uncertainty via self-consistency sampling, verifies evidence support and contradiction with an NLI cross-encoder, propagates uncertainty through dependencies, identifies evidence gaps, skips redundant retrieval via a No-Gap Fast Path, targets follow-up queries using Expected Information Gain (EIG), and safely abstains when evidence remains insufficient or contradictory.

---

## 2. THE 12-STEP ADAPTIVE EPISTEMIC LOOP

```
Step 1:  Retrieve initial evidence pool P_0 (top-k=3) for question q
Step 2:  Generate initial draft answer Y_0 using P_0
Step 3:  Estimate draft confidence c0 via self-consistency sampling (k=5, T=0.7)
Step 4:  Extract atomic factual claims C = {c_1, ..., c_m} with DAG dependencies
Step 5:  NLI verification of each claim against sentence windows in P_0 (and memory)
Step 6:  Construct Belief Graph B_0 (nodes=claims, edges=dependencies & NLI status)
Step 7:  Compute logit-space belief update and weakest-link uncertainty propagation
Step 8:  Identify evidence gaps: claims where status != "supported" or eff_conf < theta_gap
Step 9:  Fast Path check:
            If mode == "observe" or no gaps exist and min_eff >= theta_fast:
                Terminate retrieval immediately (Fast Path)
Step 10: Decision-Theoretic Query Selection:
            Propose n=3 candidate queries for the weakest gap claim;
            Score via EIG heuristic: Score(qy) = (1.0 - claim.eff) * novelty(qy, pool);
            Select q* = argmax Score(qy)
Step 11: Execute adaptive retrieval with q*, expand evidence pool, re-verify claims, update graph.
            Repeat Steps 8-11 while hop < max_hops.
Step 12: Final Answer Generation or Calibrated Abstention:
            If active and (answer_claim.eff < theta_abs or answer_claim.status == "contradicted"):
                Return "unknown" (Abstain)
            Else:
                Return draft answer
```

---

## 3. THRESHOLDS & HYPERPARAMETERS

| Parameter | Default Value | Role / Operational Definition |
|:---|:---|:---|
| `max_hops` | `2` | Maximum follow-up retrieval iterations |
| `sc_samples` | `5` | Independent completions sampled for confidence estimation |
| `sc_temperature` | `0.7` | Sampling temperature for self-consistency |
| `theta_fast` | `0.8` | Stopping threshold: skip follow-up retrieval if all claims $\ge 0.8$ |
| `theta_gap` | `0.6` | Gap threshold: trigger adaptive retrieval if effective confidence $< 0.6$ |
| `theta_abs` | `0.4` | Abstention threshold: answer `"unknown"` if final answer claim $< 0.4$ |
| `theta_s` | `0.5` | NLI entailment threshold for support |
| `theta_c` | `0.5` | NLI contradiction threshold |
| `w_s` | `1.0` | Positive weight assigned to supporting evidence in logit space |
| `w_c` | `2.0` | Negative penalty assigned to contradictory evidence in logit space |
| `max_claims` | `5` | Maximum atomic propositions decomposed per answer |
| `pool_cap` | `12` | Maximum cumulative paragraphs retained in evidence pool |

---

## 4. SIX INDEPENDENT ABLATION SWITCHES

To isolate the scientific contribution of each module, System C implements six independent ablation flags via `ablate: [flag]` in configuration:

1. **`no_nli`:** Disables NLI verification. All claims are marked `"unresolved"`, so gap identification and belief updates depend exclusively on self-consistency confidence.
2. **`no_conf`:** Disables confidence-driven decisions. Retrieval continues for the answer claim up to `max_hops`, and the system never abstains.
3. **`no_prop`:** Disables weakest-link dependency propagation. Each claim's effective confidence equals its own local confidence ($c_{eff} = c_{conf}$).
4. **`no_adaptive`:** Disables follow-up retrieval (draft, verify, and abstain only).
5. **`no_fast`:** Disables the No-Gap Fast Path. Always retrieves for the weakest claim until `max_hops` is reached.
6. **`no_eig`:** Disables candidate query generation and EIG scoring. Uses the raw text of the weakest claim as the search query.

---

## 5. RETURN SCHEMA & OBSERVATIONAL MODE (GATE G1 & G2)

### 5.1 Observational Mode (Gate G1)
When `erag.mode = "observe"`:
- The system executes Hop 0 (draft, self-consistency, claims, NLI, belief graph).
- The system logs `would_abstain = (ans.eff < theta_abs or ans.status == "contradicted")`.
- The system **strictly returns the uninhibited initial draft** (`answer = draft`).
- **Gate G1 Assertion:** $\text{EM}_{\text{observe}} \equiv \text{EM}_{\text{Standard RAG}}$ ($1.000$ ratio).

### 5.2 Return Schema
```python
{
    "answer": answer,              # Predicted answer (or "unknown" if abstained)
    "draft": draft,                # Unmodified initial draft
    "abstained": bool,             # True if active mode triggered abstention
    "would_abstain": bool,         # True if abstention condition was met
    "answer_conf": float,          # Effective confidence of answer-bearing claim
    "c0": float,                   # Initial self-consistency confidence
    "pool_titles": list,           # Final evidence pool paragraph titles
    "hops": int,                   # Hops executed (0 if Fast Path, up to max_hops)
    "trace": list,                 # Hop-by-hop JSON serializations of belief graphs
    "mem_used": int,               # Number of memory passages attached
    "mem_events": list,            # Memory override/lookup event log
    "claim_fallbacks": int,        # Count of claims generated via fallback
}
```
