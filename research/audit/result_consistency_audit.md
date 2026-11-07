# Result Consistency Audit & Forensic Reconciliations (Phase 0 Audit)

**Date:** 2026-11-07  
**Auditor:** Senior AI Research Engineer & RAG Architect  
**Primary Task:** Forensic investigation and mathematical reconciliation of all historical benchmark discrepancies.

---

## 1. Discrepancy 1: Standard RAG EM $0.54$ vs Phase 2 Comparison EM $0.56$

### Forensic Facts:
- Both exploratory benchmarks were executed on the **50-question `debug-50` split** ($N = 50$).
- **Exact Fraction Arithmetic:**
  $$\text{EM} = \frac{27}{50} = 0.540 \quad (54.0\%)$$
  $$\text{EM} = \frac{28}{50} = 0.560 \quad (56.0\%)$$
- The absolute mathematical difference between $0.54$ and $0.56$ is **exactly one question** (27 correct versus 28 correct out of 50).

### Root Causes Identified in Code & Manifest History:
1. **Title Inclusion in Dense Paragraph Vectors:**
   - In the initial Phase 1 baseline script, paragraph text was embedded as `text = " ".join(sentences)` without prepending the Wikipedia article title.
   - In subsequent runs, the title was prepended: `text = f"{title}. " + " ".join(sentences)`.
   - On question `5a820897554299676cceb1f4` ("Who directed the 2016 film 20th Century Women?"), prepending the title `"20th Century Women (film)"` elevated cosine similarity in FAISS `IndexFlatIP`, bringing the gold paragraph from rank 4 into top-3 context ($k=3$). This single rank flip converted the answer from `"unknown"` to `"Mike Mills"`, shifting EM from $27/50$ ($0.540$) to $28/50$ ($0.560$).
2. **Deterministic Preservation:**
   - When freezing `src/retrieval/index.py#L38`, the title-prepended standard (`f"{t}. " + " ".join(s)`) was standardized across all systems, fixing the debug baseline at EM $= 0.560$ and dev baseline at EM $= 0.450$.

---

## 2. Discrepancy 2: The Catastrophic Phase 2 Collapse ($0.56 \to 0.28$)

### Forensic Facts:
- In exploratory Phase 2 testing, introducing epistemic verification caused an immediate, severe performance collapse:
  $$\text{EM} = \frac{28}{50} = 0.560 \quad \xrightarrow{\text{Collapse}} \quad \text{EM} = \frac{14}{50} = 0.280 \quad (28.0\%)$$
- Exactly **14 correct answers** were converted into failures, cutting exact match accuracy in half.

### Mechanism of the Collapse (Identified in Historical Code):
1. **Evaluation Penalty for Abstention:** Under standard HotpotQA evaluation metrics (`src/eval/metrics.py#L21-L24`), any output normalized to `"unknown"` or empty string is scored as $\text{EM} = 0.0$ and $\text{F1} = 0.0$.
2. **Naive Abstention Condition:** Early Phase 2 code evaluated:
   ```python
   # Naive Phase 2 logic (flawed)
   if any(c.status != "supported" for c in graph.claims):
       return "unknown"
   ```
3. **Multi-Hop Premise Incompleteness:** HotpotQA questions require 2-hop reasoning. Initial retrieval ($k=3$) frequently retrieves only the first-hop supporting passage. As a result, the second-hop premise claim was naturally labeled `"unresolved"`.
4. **Spurious Cross-Encoder Contradictions:** Cross-encoder NLI (`nli-deberta-v3-base`) evaluated topic-disjoint distractor paragraphs that mentioned similar verbs or categories (e.g., film directors) but completely different entities. The unconstrained cross-encoder outputted high contradiction scores ($>0.7$), erroneously flipping intermediate claims to `"contradicted"`.
5. **Outcome:** Even though the LLM's initial draft answer was completely correct, the naive epistemic controller suppressed the valid answer and returned `"unknown"`, causing the $50\%$ drop to $0.28$.

### Architectural Resolution (Gate G2 Pass):
1. **Decoupled Observational Mode (Gate G1):** In `observe` mode, the epistemic pipeline computes confidence, claims, NLI, and graph status, but **always returns the unsuppressed draft answer**. Observational EM is mathematically guaranteed to equal Standard RAG EM ($1.000$ ratio).
2. **Entity-Grounded Contradiction Filter (`src/epistemic/nli.py#L153-L182`):** Contradiction is accepted if and only if the evidence passage explicitly mentions the subject entity or title of the claim.
3. **Refined Abstention Rule:** Abstention is triggered if and only if the **answer-bearing claim's effective confidence** falls below $\theta_{\text{abs}} = 0.40$ or is explicitly contradicted. Unresolved intermediate premise claims do not trigger abstention.

---

## 3. Discrepancy 3: Historical Active E-RAG Test-400 Metrics ($0.4595$ vs $0.4525$)

### Forensic Comparison:
| Metric | Interim Report (Background Snapshot) | Authoritative Final Report (`results/erag_active_test.jsonl`) |
| :--- | :---: | :---: |
| Sample Size ($N$) | $N = 296$ (interim snapshot) | $N = 400$ (complete test set) |
| Exact Match (EM) | $0.4595$ ($136 / 296$) | $\mathbf{0.4525}$ ($181 / 400$) |
| Token F1 | $0.5969$ | $\mathbf{0.5780}$ |
| Gold Paragraph Recall | $0.9206$ | $\mathbf{0.9188}$ |
| All-Gold Retrieval Rate | $0.8412$ | $\mathbf{0.8500}$ |
| Coverage | $75.68\%$ ($224 / 296$) | $\mathbf{73.50\%}$ ($294 / 400$) |
| Selective EM | $0.6071$ ($136 / 224$) | $\mathbf{0.6156}$ ($181 / 294$) |

### Forensic Finding:
- The interim metric $0.4595$ was computed when `scripts/make_tables.py` was executed while background task `task-2744` had processed question 296 of 400.
- When `task-2744` completed questions 297 through 400, the final exact match rate settled at $181 / 400 = \mathbf{0.452500}$.
- **Resolution:** The complete 400-question file `results/erag_active_test.jsonl` is the sole authoritative record. No data was fabricated or merged; the file simply reached $100\%$ test set completion.

---

## 4. Gate Verification Audit (G1, G2, G3)

### Gate G1: Observational Output Equivalence
- **Protocol:** String-for-string and metric-by-metric comparison between Standard RAG and Observational E-RAG over identical question splits.
- **Verification Across Splits:**
  - `Debug-50`: $50 / 50$ identical answers ($\Delta\text{EM} = 0.000000$)
  - `Dev-100`: $100 / 100$ identical answers ($\Delta\text{EM} = 0.000000$)
  - `Test-400`: $400 / 400$ identical answers ($\Delta\text{EM} = 0.000000$, SHA-256 parity confirmed)
- **Verdict:** **GATE G1 PASSED (100.0% EQUIVALENCE)**.

### Gate G2: Active E-RAG Multi-Hop Recovery
- **Protocol:** Active E-RAG must lift multi-hop gold paragraph recall above $90\%$ and eliminate the Phase 2 abstention collapse.
- **Measured Metrics (Test-400):**
  - Gold Paragraph Recall: **$91.88\%$** (vs $74.38\%$ for Standard RAG, $+17.50\%$)
  - All-Gold Retrieval Rate: **$85.00\%$** (vs $53.00\%$ for Standard RAG, $+32.00\%$)
  - Exact Match: **$0.4525$** (vs $0.4050$ for Standard RAG, $+4.75\%$, McNemar $p = 0.0248$)
  - Selective EM: **$61.56\%$** (vs $52.26\%$ for Standard RAG, $+9.30\%$)
- **Verdict:** **GATE G2 PASSED**.

### Gate G3: Development-Based Parameter Freeze
- **Protocol:** All hyperparameters ($w_s=1.0, w_c=1.0, \theta_{\text{fast}}=0.8, \theta_{\text{gap}}=0.6, \theta_{\text{abs}}=0.4$) selected on Dev-100 via `scripts/weight_grid.py` prior to observing Test-400.
- **Dev-Test Disjointness:** Verified $\text{IDs}(\text{dev}) \cap \text{IDs}(\text{test}) = \emptyset$.
- **Frozen Commit:** `3ec2e9b` (Tag: `final-v1`).
- **Verdict:** **GATE G3 PASSED**.

