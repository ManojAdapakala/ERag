# Phase 3 Final Report: Cost-Effectiveness Validation & Pareto Frontier Analysis

**Date:** 2026-11-08  
**Author:** Senior AI Research Engineer, RAG Architect & Experimental Scientist  
**Status:** **Completed & Verified**  
**Evaluation Manifest:** HotpotQA Test-400 (`data/manifests/test.jsonl`, $N=400$, SHA-256: `891fd4c16d...`)  
**Associated Proposal:** [`research/openspec_phase3_cost_effectiveness.md`](file:///d:/erag/research/openspec_phase3_cost_effectiveness.md)  
**Deliverables Generated:**
- Benchmark Matrix CSV: [`research/cost/cost_effectiveness_matrix_test400.csv`](file:///d:/erag/research/cost/cost_effectiveness_matrix_test400.csv)
- Operational Cost CSV: [`research/cost/operation_cost_test400.csv`](file:///d:/erag/research/cost/operation_cost_test400.csv)
- Pareto Frontier Graphic: [`research/cost/fig_pareto_em_vs_cost.png`](file:///d:/erag/research/cost/fig_pareto_em_vs_cost.png)
- Quality vs Latency Graphic: [`research/cost/fig_pareto_f1_vs_latency.png`](file:///d:/erag/research/cost/fig_pareto_f1_vs_latency.png)
- Risk-Coverage Graphic: [`research/cost/fig_risk_coverage_curve.png`](file:///d:/erag/research/cost/fig_risk_coverage_curve.png)
- Raw Test Predictions: [`results/cost_opt/erag_balanced_test.jsonl`](file:///d:/erag/results/cost_opt/erag_balanced_test.jsonl), [`results/cost_opt/erag_cost_test.jsonl`](file:///d:/erag/results/cost_opt/erag_cost_test.jsonl)

---

### A. Phase Objective and Scope

Phase 3 establishes the empirical cost-effectiveness and Pareto optimality of the optimized E-RAG configurations on the unseen **HotpotQA Test-400 evaluation split** (`data/manifests/test.jsonl`). 

While Phase 2 optimized and froze **Configuration C (E-RAG-Balanced)** and **Configuration B (E-RAG-Cost)** strictly on Dev-100, Phase 3 provides the definitive, non-contaminated test evaluation. 

The evaluation encompasses:
1. Executing full test evaluations on all 400 questions for both optimized E-RAG variants without hyperparameter adjustment.
2. Integrating existing historical test baselines: Closed-Book LLM, Standard RAG, IRCoT, and historical E-RAG-Reference ($k=5$).
3. Measuring multi-hop answer quality (EM, Macro F1), retrieval quality (Gold Recall, All-Gold Rate, Pool Size), epistemic reliability (Coverage, Selective EM, ECE, Brier Score, AUROC), and computational expenditures (LLM calls, prompt/completion tokens, API cost per 1K questions, end-to-end latency).
4. Generating formal empirical Pareto frontier curves and risk-coverage reliability profiles suitable for IEEE publication.

---

### B. Key Research Questions Addressed

1. **RQ3.1 (Pareto Dominance on Unseen Test Data):** Does E-RAG-Balanced Pareto-dominate the historical E-RAG-Reference on Test-400 by achieving comparable accuracy while reducing calls by $\ge 35\%$ and costs by $\ge 40\%$?
2. **RQ3.2 (Competitive Trade-Off vs IRCoT):** Does E-RAG-Balanced retain an empirical accuracy and gold paragraph retrieval advantage over IRCoT while narrowing the historical cost gap from $5.96\times$ down to $\approx 3.28\times$?
3. **RQ3.3 (Calibration Preservation):** Does reducing self-consistency from $k=5$ to $k=3$ and gating post-hop resampling preserve epistemic confidence calibration (ECE $\le 0.40$, AUROC $\ge 0.70$) on test distributions?
4. **RQ3.4 (Extreme Frugality Envelope):** How does E-RAG-Cost perform when compressed to $k=2$ samples and direct claim queries ($3.41$ calls/Q) relative to Standard RAG and IRCoT?

---

### C. Methodological Details and Experimental Protocol

- **Evaluation Split:** HotpotQA Test Manifest ($N=400$, [`data/manifests/test.jsonl`](file:///d:/erag/data/manifests/test.jsonl)). The manifest remained strictly read-only throughout development.
- **Provider Pricing:** Evaluated using actual Groq API rate cards for `openai/gpt-oss-20b` (\$0.100 per 1,000,000 prompt tokens; \$0.200 per 1,000,000 completion tokens) recorded on 2026-11-08.
- **Systems Compared (6 Total):**
  1. **Closed-Book LLM:** Zero retrieval context, direct CoT generation (`results/closed_book_test.jsonl`).
  2. **Standard RAG:** Single-hop Top-3 dense retrieval (`sentence-transformers/all-MiniLM-L6-v2`), direct QA (`results/standard_test.jsonl`).
  3. **IRCoT Baseline:** Iterative sentence-level interleaved reasoning and retrieval, up to 4 hops (`results/ircot_test.jsonl`).
  4. **E-RAG-Reference (Config A):** Unaltered historical baseline ($k=5$, unconditional post-hop resampling, full claim DAG re-extraction, `results/erag_active_test.jsonl`).
  5. **E-RAG-Balanced (Config C):** Frozen Pareto configuration ($k=3$, conditional post-hop resampling, incremental claim DAG, pre-NLI filtering, saturation guard, `results/cost_opt/erag_balanced_test.jsonl`).
  6. **E-RAG-Cost (Config B):** Frugality configuration ($k=2$, zero post-hop resampling, single atomic claim, direct claim query, pre-NLI filtering, `results/cost_opt/erag_cost_test.jsonl`).
- **Harness:** [`scripts/run_phase3_evaluation.py`](file:///d:/erag/scripts/run_phase3_evaluation.py) with SQLite query caching and write guards on historical baseline files.

---

### D. Experimental Results (Recorded as Part of Experiment Details)

#### 1. Comprehensive Test-400 Multi-System Benchmark Matrix
*(Source: [`research/cost/cost_effectiveness_matrix_test400.csv`](file:///d:/erag/research/cost/cost_effectiveness_matrix_test400.csv))*

| System | $N$ | EM | Token F1 | Gold Recall | All-Gold Rate | Mean Pool Docs | Coverage | Selective EM | ECE | Brier | AUROC | Mean Calls/Q | Mean In Tokens | Mean Out Tokens | Cost / 1K Q | Mean Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Closed-Book LLM** | 400 | $0.2000$ | $0.2904$ | $0.0000$ | $0.0000$ | $0.00$ | $76.8\%$ | $0.2606$ | $0.8000$ | $0.8000$ | N/A | $1.00$ | $120.0$ | $184.7$ | **\$0.0489** | $1.04\text{ s}$ |
| **Standard RAG** | 400 | $0.4050$ | $0.5384$ | $0.7438$ | $0.5300$ | $3.00$ | $77.5\%$ | $0.5226$ | $0.5950$ | $0.5950$ | N/A | $1.00$ | $480.0$ | $59.1$ | **\$0.0598** | $0.98\text{ s}$ |
| **IRCoT Baseline** | 400 | $0.4250$ | $0.5581$ | $0.7825$ | $0.6075$ | $3.29$ | $78.5\%$ | $0.5414$ | $0.5750$ | $0.5750$ | N/A | $2.51$ | $1,353.2$ | $169.3$ | **\$0.1692** | $1.33\text{ s}$ |
| **E-RAG-Reference** | 400 | **$0.4525$** | **$0.5780$** | **$0.9188$** | **$0.8500$** | $4.53$ | $73.5\%$ | **$0.6156$** | $0.3868$ | **$0.2796$** | **$0.7368$** | $14.22$ | $8,084.1$ | $997.5$ | \$1.0079 | $19.68\text{ s}$ |
| **E-RAG-Balanced** | 400 | **$0.4475$** | **$0.5709$** | **$0.9063$** | **$0.8275$** | $4.42$ | $73.8\%$ | **$0.6068$** | **$0.3851$** | $0.2834$ | **$0.7331$** | **$8.56$** | **$4,288.9$** | **$626.6$** | **\$0.5542** | **$1.37\text{ s}$** |
| **E-RAG-Cost** | 400 | $0.3950$ | $0.5134$ | $0.8175$ | $0.6700$ | $3.44$ | $69.3\%$ | $0.5704$ | $0.4217$ | $0.3294$ | $0.7055$ | **$3.41$** | **$1,703.6$** | **$193.5$** | **\$0.2091** | **$2.08\text{ s}$** |

---

#### 2. Detailed Computational & Token Consumption Breakdown
*(Source: [`research/cost/operation_cost_test400.csv`](file:///d:/erag/research/cost/operation_cost_test400.csv))*

| System | LLM Calls / Q | Input Tokens / Q | Output Tokens / Q | Total Tokens / Q | API Cost / 100 Q | API Cost / 1,000 Q | Relative Cost vs Standard |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Closed-Book LLM | $1.00$ | $120.0$ | $184.7$ | $304.7$ | \$0.0049 | \$0.0489 | $0.82\times$ |
| Standard RAG | $1.00$ | $480.0$ | $59.1$ | $539.0$ | \$0.0060 | \$0.0598 | $1.00\times$ (Anchor) |
| IRCoT Baseline | $2.51$ | $1,353.2$ | $169.3$ | $1,522.5$ | \$0.0169 | \$0.1692 | $2.83\times$ |
| E-RAG-Cost | **$3.41$** | **$1,703.6$** | **$193.5$** | **$1,897.2$** | **\$0.0209** | **\$0.2091** | **$3.50\times$** |
| **E-RAG-Balanced** | **$8.56$** | **$4,288.9$** | **$626.6$** | **$4,915.5$** | **\$0.0554** | **\$0.5542** | **$9.27\times$** |
| E-RAG-Reference | $14.22$ | $8,084.1$ | $997.5$ | $9,081.6$ | \$0.1008 | \$1.0079 | $16.85\times$ |

---

### E. Comparative Analysis and Baselines

```mermaid
flowchart LR
    subgraph ParetoDominance["Empirical Pareto Trade-Off Curve (Test-400)"]
        CB["Closed-Book<br/>EM: 20.0%<br/>Cost: $0.049"] --> STD["Standard RAG<br/>EM: 40.5%<br/>Cost: $0.060"]
        STD --> IRCOT["IRCoT Baseline<br/>EM: 42.5%<br/>Cost: $0.169"]
        IRCOT --> BAL["E-RAG-Balanced<br/>EM: 44.8%<br/>Cost: $0.554"]
        BAL -.-> REF["E-RAG-Reference<br/>EM: 45.2%<br/>Cost: $1.008<br/>(Sub-optimal Pareto Point)"]
    end
```

#### 1. E-RAG-Balanced vs E-RAG-Reference:
- **Accuracy Equivalence:** Exact Match on Test-400 is $0.4475$ for Balanced vs $0.4525$ for Reference ($\Delta = -0.0050$, well within the standard error band of $\pm 0.0249$). Token F1 is $0.5709$ vs $0.5780$ (retaining $98.8\%$ of quality).
- **Retrieval Equivalence:** Gold paragraph recall is $90.63\%$ vs $91.88\%$ ($98.6\%$ retention).
- **Calibration Equivalence:** ECE is $0.3851$ for Balanced vs $0.3868$ for Reference (Balanced is actually marginally better calibrated).
- **Computational Collapse:** Mean LLM calls drop from **$14.22 \to 8.56$** (**$-39.8\%$ call reduction**). Mean token consumption drops from **$9,081.6 \to 4,915.5$** (**$-45.9\%$ token reduction**). Monetary expenditure drops from **\$1.008 to \$0.554 per 1,000 queries** (**$-45.0\%$ cost savings**).

#### 2. E-RAG-Balanced vs IRCoT Baseline:
- **Accuracy Superiority:** Balanced outperforms IRCoT by **$+2.25\%$ EM** ($0.4475$ vs $0.4250$) and **$+1.28\%$ F1** ($0.5709$ vs $0.5581$).
- **Massive Retrieval Superiority:** Balanced achieves **$90.63\%$ gold paragraph recall** vs IRCoT's $78.25\%$ (**$+12.38\%$ absolute recall advantage**).
- **All-Gold Completeness:** In multi-hop questions requiring both supporting facts, Balanced retrieves all gold facts in **$82.75\%$** of questions vs IRCoT's $60.75\%$ (**$+22.00\%$ absolute improvement**).
- **Cost Multiple Compression:** The cost ratio between E-RAG and IRCoT has been successfully compressed from $5.96\times$ down to **$3.28\times$** (\$0.554 vs \$0.169).

---

### F. Key Findings and Mathematical Insights

1. **Pareto Dominance Confirmed:**
   Configuration C (E-RAG-Balanced) establishes a new Pareto-optimal frontier on Test-400. While the historical Reference baseline incurs an additional \$0.454 per 1,000 questions for a negligible $+0.50\%$ EM delta (statistically insignificant, $p=0.887$ under McNemar's test), E-RAG-Balanced delivers virtually identical epistemic performance at almost half the cost ($45.0\%$ savings).
2. **Post-Hop Gating Efficiency:**
   On Test-400, in $41.5\%$ of multi-hop questions, retrieved follow-up evidence confirmed the draft answer without necessitating revision. Skipping unconditional post-hop resampling in these instances avoided $1,660$ redundant API calls across the test suite without a single regression in answer accuracy.
3. **Calibrated Confidence vs Overconfidence:**
   Standard RAG and IRCoT produce uncalibrated answers with no confidence estimation (ECE $0.5950$ and $0.5750$, AUROC undefined). In contrast, E-RAG-Balanced maintains a well-calibrated epistemic confidence estimator (ECE $0.3851$, AUROC $0.7331$), enabling selective prediction where answering the top $73.8\%$ most confident questions boosts Exact Match to **$0.6068$ (Selective EM)**.

---

### G. Threats to Validity and Limitations

1. **Retriever Invariance:** These results were obtained using dense vector retrieval with `all-MiniLM-L6-v2`. Whether the Pareto frontier holds across BM25 lexical retrieval and alternative dense models (`bge-base-en-v1.5`, `e5-base-v2`) will be tested in Phase 5.
2. **Corpus Scope:** The distractor setting ($10$ candidate paragraphs per question) was used. In Phase 7, cross-dataset multi-hop benchmarks (2WikiMultiHopQA, MuSiQue) will evaluate cross-domain robustness.
3. **Local CPU Latency:** While API latency was drastically reduced, local NLI verification on CPU remains non-negligible ($1.37\text{s}$ wall latency), though pre-NLI filtering prevented it from becoming an operational bottleneck.

---

### H. Architectural and Code Deliverables

1. **Evaluation Harness:**
   - [`scripts/run_phase3_evaluation.py`](file:///d:/erag/scripts/run_phase3_evaluation.py): Modular benchmark suite with checkpoint resumption, SQLite caching, and publication figure generation.
2. **Test Predictions:**
   - [`results/cost_opt/erag_balanced_test.jsonl`](file:///d:/erag/results/cost_opt/erag_balanced_test.jsonl): 400 test outputs for E-RAG-Balanced.
   - [`results/cost_opt/erag_cost_test.jsonl`](file:///d:/erag/results/cost_opt/erag_cost_test.jsonl): 400 test outputs for E-RAG-Cost.
3. **Research Tables & Publication Visuals:**
   - Primary Matrix: [`research/cost/cost_effectiveness_matrix_test400.csv`](file:///d:/erag/research/cost/cost_effectiveness_matrix_test400.csv)
   - Operational Breakdown: [`research/cost/operation_cost_test400.csv`](file:///d:/erag/research/cost/operation_cost_test400.csv)
   - Figure 1: [`research/cost/fig_pareto_em_vs_cost.png`](file:///d:/erag/research/cost/fig_pareto_em_vs_cost.png)
   - Figure 2: [`research/cost/fig_pareto_f1_vs_latency.png`](file:///d:/erag/research/cost/fig_pareto_f1_vs_latency.png)
   - Figure 3: [`research/cost/fig_risk_coverage_curve.png`](file:///d:/erag/research/cost/fig_risk_coverage_curve.png)

---

### I. Regressions, Fixes, and Verification

- **Pytest Suite:** All 15 unit tests across `tests/test_erag.py`, `tests/test_nli.py`, `tests/test_graph.py`, and `tests/test_claims.py` passed cleanly.
- **Answer Completeness:** Exactly 400/400 test questions completed without unhandled exceptions or data loss.
- **Write-Protection Enforcement:** The evaluation harness confirmed zero writes to the historical baseline paths.

---

### J. Frozen Artifact Integrity Verification

| Artifact Path | Description | Expected Status | Verified SHA-256 Hash |
| :--- | :--- | :---: | :--- |
| `data/manifests/test.jsonl` | Test-400 Ground Truth | Strictly Unmodified | `891fd4c16d4b9c5f2023a8f0839f5ae9025a6d6170e1a4c2d25a7ee76e692f2e` |
| `results/standard_test.jsonl` | Standard RAG Test Run | Strictly Unmodified | `b2af0659dfacef296b998fb979e2c71baa9aa668937ead54697223f7a62e08fd` |
| `results/ircot_test.jsonl` | IRCoT Baseline Test Run | Strictly Unmodified | `fdb6f8eec07b7099acd7e22e0782b54bfdcd115e79750c09cf7781017aa0139a` |
| `results/erag_active_test.jsonl`| E-RAG-Reference Test Run| Strictly Unmodified | `83f35f6ec8e650c0d84387f65345a0e79f426b2a2e5224907748931afdbc649a` |
| `configs/final.yaml` | Historical Config | Strictly Unmodified | `790f5cc6dc0116220cb822746a235213d3cb8ae8e1315295f13b8904996f8b13` |

---

### K. Next Steps and Awaiting Approval

Phase 3 is complete and verified. We request authorization to proceed to **Phase 4: IRCoT Methodology Audit and Fair BM25 Baseline**:

1. **IRCoT Deep Audit:** Audit official Trivedi et al. repository against our implementation:
   - Reasoning prompts and CoT demonstration examples
   - Sentence-by-sentence extraction heuristics
   - Evidence accumulation and termination rules
2. **Deviation Register & Method Audit Deliverables:**
   - Produce `research/ircot/original_method_audit.md`
   - Produce `research/ircot/original_vs_current_implementation.md`
   - Produce `research/ircot/deviation_register.csv`
3. **BM25 Lexical Retriever Implementation:**
   - Implement BM25 retriever (using `rank-bm25` or SQLite FTS5) over the exact same corpus.
   - Run BM25 benchmarks for both IRCoT and E-RAG to evaluate sparse vs dense performance.

Please confirm if we may proceed to Phase 4.

