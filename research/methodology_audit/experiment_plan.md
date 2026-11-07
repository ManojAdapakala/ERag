# Controlled Dev-100 Experimental Plan & Sensitivity Study Matrix

**Target Dataset:** HotpotQA Dev-100 (`data/manifests/dev.jsonl`, $N=100$, frozen).  
**Protocol Constraint:** All parameter sweeps, sensitivity evaluations, and calibration studies must be performed **strictly on Dev-100**. The Test-400 benchmark (`data/manifests/test.jsonl`) remains frozen.

---

## 1. Experimental Architecture & Isolation Strategy

To ensure zero cross-contamination and guarantee complete audit traceability:
1. **Isolated Namespaces:** Every experimental sweep will write to a designated, clean cache namespace in `results/cache.sqlite` (e.g. `dev_sens_ircot`, `dev_sens_epistemic`, `dev_sens_graph`, `dev_sens_memory`).
2. **Deterministic Seed Control:** All sampling mechanisms (Self-Consistency, tie-breaking, order) will use `seed=13`.
3. **Offline Fast-Grid for Post-Inference Parameters:**  
   Parameters that govern post-generation logic (such as belief weights $w_s, w_c$, propagation $\text{propagate}$, gap threshold $\theta_{\text{gap}}$, fast-path threshold $\theta_{\text{fast}}$, and abstention threshold $\theta_{\text{abs}}$) can be evaluated across large combinatorial grids using saved intermediate claim-NLI states (`trace` objects) without consuming external Groq API calls or incurring network latency.
4. **Isolated Instrumentation Layer:** Sub-component latency profiling (retrieval, NLI, graph, memory, LLM) will be executed via a dedicated non-destructive profiling harness (`scripts/profile_efficiency.py`) on Dev-100 with separate cold-start and warm-inference timing passes.

---

## 2. Experimental Sensitivity Matrices (Dev-100)

### Matrix C: IRCoT Parameter Sensitivity
Investigating the interaction between step budget, retrieval capacity, and lexical vs semantic queries:

| Experiment ID | Varied Parameter | Baseline | Candidate Values | Fixed Parameters | Primary Metrics |
|---|---|:---:|---|---|---|
| **C1** | `retrieval.k` (Initial & Step $k$) | $3$ | $1, 3, 5, 7, 10$ | `max_steps=4`, `pool_cap=12`, $T=0.0$ | Recall, EM, Token F1, Retrievals/Q |
| **C2** | `ircot.max_steps` ($M$) | $4$ | $1, 2, 3, 4, 5, 6$ | $k=3$, `pool_cap=12`, $T=0.0$ | Hop distribution, Stopping trigger, EM, Latency |
| **C3** | `ircot.pool_cap` ($P$) | $12$ | $6, 9, 12, 15, 20$ | $k=3$, `max_steps=4`, $T=0.0$ | Pool saturation, Recall, Context window tokens |
| **C4** | Reasoning Temperature ($T$) | $0.0$ | $0.0, 0.2, 0.5, 0.7$ | $k=3$, `max_steps=4`, `pool_cap=12` | Reasoning step diversity, EM, F1 variance |

---

### Matrix D: Epistemic Confidence, Claims & NLI Verification
Evaluating uncertainty estimation, factual granularity, and cross-encoder thresholds:

| Experiment ID | Varied Parameter | Baseline | Candidate Values | Fixed Parameters | Primary Metrics |
|---|---|:---:|---|---|---|
| **D1.1** | `sc_samples` ($k_{\text{sc}}$) | $5$ | $1, 3, 5, 7, 10$ | $T=0.7$, `max_claims=5` | ECE, Brier score, LLM cost, Latency |
| **D1.2** | `sc_temperature` ($T_{\text{sc}}$) | $0.7$ | $0.0, 0.2, 0.5, 0.7, 1.0$ | $k_{\text{sc}}=5$, `max_claims=5` | Agreement entropy, Calibration error |
| **D1.3** | `confidence_def` | `majority` | `majority`, `draft_agreement` | $k_{\text{sc}}=5$, $T=0.7$ | Discrimination, AUROC, Brier |
| **D2.1** | `max_claims` ($n_{\text{max}}$) | $5$ | $1, 3, 5, 7, 10$ | $w_s=1.0, w_c=1.0$ | Claim count, Fallback %, Graph depth |
| **D3.1** | NLI Threshold $\theta_s$ | $0.5$ | $0.3, 0.4, 0.5, 0.6, 0.7$ | $\theta_c=0.5$, `window=2` | Support rate, False support %, Selective EM |
| **D3.2** | NLI Threshold $\theta_c$ | $0.5$ | $0.3, 0.4, 0.5, 0.6, 0.7$ | $\theta_s=0.5$, with entity grounding | Contradiction rate, Spurious abstention % |

---

### Matrix E: Belief Graph Construction & Propagation
Evaluating Bayesian evidence updating and weakest-link uncertainty propagation:

| Experiment ID | Varied Parameter | Baseline | Candidate Values | Fixed Parameters | Primary Metrics |
|---|---|:---:|---|---|---|
| **E1.1** | Support Weight $w_s$ | $1.0$ | $0.25, 0.5, 1.0, 1.5, 2.0$ | $w_c=1.0$, `propagate=True` | Posterior confidence shifts, AUROC |
| **E1.2** | Contradiction Weight $w_c$ | $1.0$ | $0.25, 0.5, 1.0, 1.5, 2.0$ | $w_s=1.0$, `propagate=True` | Penalty impact, Abstention rate, EM |
| **E1.3** | Weakest-Link Propagation | `True` | `True` (Weakest-Link), `False` (Independent) | $w_s=1.0, w_c=1.0$ | AUROC, Selective EM, Error calibration |
| **E2.1** | Fast-Path Threshold $\theta_{\text{fast}}$| $0.8$ | $0.65, 0.75, 0.80, 0.85, 0.90$ | $\theta_{\text{gap}}=0.6, \theta_{\text{abs}}=0.4$ | Fast-path exit rate, Retrieval hops, Recall |
| **E2.2** | Evidence-Gap Threshold $\theta_{\text{gap}}$| $0.6$ | $0.4, 0.5, 0.6, 0.7$ | $\theta_{\text{fast}}=0.8, \theta_{\text{abs}}=0.4$ | Gap trigger frequency, Targeted retrievals |
| **E2.3** | Abstention Threshold $\theta_{\text{abs}}$ | $0.4$ | $0.2, 0.3, 0.4, 0.5, 0.6$ | $\theta_{\text{fast}}=0.8, \theta_{\text{gap}}=0.6$ | Coverage, Selective EM, Risk-coverage trade-off |

---

### Matrix F: Adaptive Retrieval & EIG Query Selection
Evaluating epistemic-directed evidence acquisition:

| Experiment ID | Varied Parameter | Baseline | Candidate Values | Fixed Parameters | Primary Metrics |
|---|---|:---:|---|---|---|
| **F1** | Max Retrieval Hops | $2$ | $0, 1, 2, 3, 4$ | $k=3$, `pool_cap=12` | Paragraph recall, All-gold rate, Latency |
| **F2** | Retrieval Pool Cap | $12$ | $6, 9, 12, 15, 20$ | $\text{max\_hops}=2$ | Pool saturation, Recall gains vs context overhead |
| **F3** | Query Selection Strategy | EIG Novelty | EIG Novelty vs No-EIG (Raw Claim) | $\text{max\_hops}=2, k=3$ | Probe searches, Unique titles added, Recall |
| **F4** | Retrieval Setting | Distractor | Distractor (10) vs Pooled (1,000) | Full E-RAG Active | Discrimination under expanded distractor corpus |

---

### Matrix G: Verified-Belief Episodic Memory
Evaluating dynamic memory writing, surprise triggers, contradiction deletion, and retention:

| Experiment ID | Varied Parameter | Baseline | Candidate Values | Evaluation Setting | Primary Metrics |
|---|---|:---:|---|---|---|
| **G1** | Memory Write Threshold $\theta_{\text{write}}$ | $0.7$ | $0.6, 0.7, 0.8, 0.9$ | Dev Stream-30 | Memory insertions, Belief precision |
| **G2** | Surprise Threshold $\theta_{\text{surprise}}$ | $0.6$ | $0.2, 0.3, 0.4, 0.5, 0.6$ | Dev Stream-30 | Gated write frequency, Redundant store suppression |
| **G3** | Memory Decay Rate | $0.98$ | $0.90, 0.95, 0.98, 0.99, 1.0$ | Extended sequence (30 items) | Retention half-life, Eviction rate |
| **G4** | Memory Capacity | $200$ | $50, 100, 200, 500$ | Extended sequence | LRU eviction count, Memory hit rate |
| **G5** | Contradiction Purge (Override) | Enabled | Enabled vs Disabled | Planted false-belief test | Erroneous belief purge rate, Repeat accuracy |

---

### Matrix H: Micro-Benchmarking & Latency Profiling (Dev-100)
Executing isolated profiling harness with `time.perf_counter()` to collect Section H metrics across all 5 systems:
1. `t_retrieval_init`: Time to encode question and search FAISS for initial top-$k$.
2. `t_retrieval_hop`: Time to execute follow-up hop retrievals (IRCoT / E-RAG).
3. `t_llm_draft`: Time for initial draft completion.
4. `t_sc`: Time for self-consistency sampling (5 parallel/sequential requests).
5. `t_claims`: Time for claim decomposition prompt and JSON parsing.
6. `t_nli`: Time for DeBERTa-v3 batch scoring across sliding passage windows.
7. `t_graph`: Time for logit evidence updates, topological sort, and weakest-link propagation.
8. `t_memory`: Time for embedding similarity lookup and contradiction screening.
9. `t_total`: Total end-to-end wall-clock latency per question.
10. `cost_api`: Calculated Groq token costs based on exact input and output token tallies.

---

## 3. Execution Schedule & Milestones

Following the user's explicit Stage workflow:
- **Milestone 1 (Stage 1):** Complete Read-Only Audit & Data Manifests (COMPLETED).
- **Milestone 2 (Stage 2):** Authoritative Visualizations & 1-vs-1 IRCoT vs E-RAG Active Analysis (from existing validated $N=400$ test data).
- **Milestone 3 (Stage 3):** Isolated Efficiency Profiler Implementation and Dev-100 Micro-Benchmark Execution.
- **Milestone 4 (Stage 4):** Execution of Dev-100 Sensitivity Studies (Matrices C through G).
- **Milestone 5 (Stage 5):** Generation of all 17 sensitivity plots and Pareto frontiers.
- **Milestone 6 (Stage 6):** Synthesis of Springer-format research audit report (`final_methodology_audit.md`).

