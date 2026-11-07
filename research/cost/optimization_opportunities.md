# Phase 1: Computational Optimization Opportunities & Research Hypotheses

**Audit Date:** 2026-11-07  
**Auditor:** Senior AI Research Engineer & RAG Architect  
**Objective:** Formulate controlled, testable hypotheses for Phase 2 cost reduction without sacrificing answer accuracy or epistemic calibration.

---

## 1. Primary Inefficiencies Identified in Phase 1 Audit

| Inefficiency Category | Empirical Cost | Cause / Mechanism | Proposed Fix for Phase 2 |
| :--- | :---: | :--- | :--- |
| **1. Dual Self-Consistency Resampling** | **$3.61$ calls/Q** ($25.4\%$ of calls) | When $hop > 0$, E-RAG runs self-consistency a second time with $k=5$ samples, even if the revised draft answer is identical to the initial draft. | **Hypothesis H1:** Skip post-hop self-consistency if normalized draft answer did not change, or reduce post-hop $k$ to $3$ or $2$. |
| **2. Self-Consistency Sample Redundancy** | **$4.00$ to $6.00$ calls/Q** ($35.0\%$ of calls) | Initial sampling uses $k=5$ uniformly across all questions, including straightforward queries where the initial draft has high confidence. | **Hypothesis H2:** Test $k \in \{1, 2, 3, 5\}$ on Dev-100. Evaluate if $k=3$ maintains identical ECE and AUROC while cutting $40\%$ of sampling calls. |
| **3. The False-Search Phenomenon** | **$0.48$ calls/Q** + $1.44$ probe searches | In $43.2\%$ of Fast Path questions ($48/111$), E-RAG generated queries and searched FAISS, but all retrieved passages were already in the pool. | **Hypothesis H3:** Fast-fail candidate evaluation: check title overlap immediately. If pool is saturated or no new passages exist, suppress subsequent queries and exit immediately. |
| **4. Redundant Full-Pass Claim Extraction** | **$1.07$ calls/Q** ($7.5\%$ of calls) | When revising drafts at hop 1 and hop 2, all claims are extracted from scratch, forcing full re-extraction and re-verification of already verified premises. | **Hypothesis H4:** Incremental claim extraction: preserve previously verified claim nodes and only extract and verify novel claims introduced by newly added context. |
| **5. EIG Query Generation LLM Calls** | **$1.46$ calls/Q** ($10.3\%$ of calls) | An LLM call is made to brainstorm 3 candidate queries for every gap. | **Hypothesis H5:** Direct claim text retrieval: evaluate whether searching directly with the atomic claim text matches the recall of LLM-generated search queries (as tested in the `no_eig` ablation where recall was $90.5\%$). |
| **6. Cross-Encoder CPU Verification Overhead** | **$1,605.9\text{ ms}$** ($59.7\%$ of latency) | DeBERTa-v3 evaluates sliding windows across all passages sequentially on CPU. | **Hypothesis H6:** Fast entity pre-filtering before NLI scoring: skip cross-encoder inference entirely if neither the passage title nor any subject entity appears in the passage text. |

---

## 2. Experimental Hypotheses for Phase 2

### Hypothesis 1 (Sampling Economy):
$$\mathcal{H}_1: \text{Reducing } k_{\text{sc}} \text{ from } 5 \text{ to } 3 \text{ reduces total LLM calls by } \ge 30\% \text{ while keeping Exact Match within } \pm 0.005 \text{ and ECE within } \pm 0.025.$$
- **Baseline:** $k=5$ (Mean calls: $14.22$, Cost: \$1.008/1K Q).
- **Target:** $k=3$ (Expected calls: $\sim 9.50$, Expected cost: $\sim \$0.650$/1K Q).
- **Validation Metric:** Dev-100 EM, F1, ECE, Brier score.

### Hypothesis 2 (Post-Hop Resampling Gating):
$$\mathcal{H}_2: \text{Executing post-hop self-consistency conditionally (only when } \text{draft}_{\text{new}} \ne \text{draft}_{\text{old}}\text{) eliminates } \approx 2.5 \text{ calls/Q without any change to final answers.}$$
- **Mechanism:** If newly retrieved evidence did not alter the draft answer, the prior consensus $c_0$ remains valid; re-sampling 5 identical completions is mathematically redundant.

### Hypothesis 3 (Fast Pre-NLI Filtering):
$$\mathcal{H}_3: \text{Enforcing the entity grounding filter prior to cross-encoder inference eliminates } \ge 50\% \text{ of CPU forward passes, cutting unmemoized latency from } 1.6\text{s} \text{ to } <0.6\text{s.}$$
- **Mechanism:** In the current code (`src/epistemic/nli.py`), cross-encoder probabilities $P$ are computed for all passage windows first, and entity grounding is applied post-hoc to filter contradictions. Moving entity matching *before* `self.best()` avoids running DeBERTa-v3 on topic-disjoint distractors.

---

## 3. Candidate Configurations to Benchmark in Phase 2

To explore the Pareto frontier between accuracy, reliability, and computational cost, Phase 2 will construct and benchmark three distinct configurations on Dev-100:

### Configuration A: E-RAG-Reference (Unchanged Frozen Baseline)
- `sc_samples`: $5$
- `post_hop_resample`: `always`
- `claim_extraction`: `full_reextract`
- `eig_query_gen`: `llm_3_queries`
- `nli_filter`: `post_hoc`
- **Expected Calls:** $\sim 14.2$ calls/Q
- **Expected Cost:** $\sim \$1.01$ / 1K Q

### Configuration B: E-RAG-Cost (Minimum Expenditure Profile)
- `sc_samples`: $2$
- `post_hop_resample`: `never` (reuse $c_0$ updated via NLI evidence update only)
- `claim_extraction`: `single_atomic_answer_claim`
- `eig_query_gen`: `direct_claim_text` (zero LLM calls)
- `nli_filter`: `pre_inference`
- **Target Calls:** $\le 4.5$ calls/Q ($\sim 70\%$ reduction)
- **Target Cost:** $\le \$0.35$ / 1K Q

### Configuration C: E-RAG-Balanced (Optimal Quality-Cost Pareto Point)
- `sc_samples`: $3$
- `post_hop_resample`: `conditional` (only if $\text{draft}_{\text{new}} \ne \text{draft}_{\text{old}}$)
- `claim_extraction`: `full_reextract` (preserves multi-hop reasoning DAG)
- `eig_query_gen`: `llm_3_queries` with false-search suppression
- `nli_filter`: `pre_inference`
- **Target Calls:** $\le 8.0$ calls/Q ($\sim 45\%$ reduction)
- **Target Cost:** $\le \$0.55$ / 1K Q
- **Quality Constraint:** Must achieve $\ge 98\%$ of Reference Exact Match and Gold Recall.

