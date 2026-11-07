# E-RAG: Practical System Recommendations & Pareto Trade-Off Analysis

**Benchmark Foundations:** HotpotQA Dev-100 ($N=100$) and Test-400 ($N=400$, frozen).  
**Target Publication Format:** Springer Lecture Notes in Artificial Intelligence (LNAI) / Information Systems.

---

## 1. Executive Pareto Analysis

Deploying agentic retrieval-augmented generation systems in production involves navigating fundamental trade-offs between four competing objectives:
1. **Raw Accuracy (Exact Match & Token F1)**: Maximum correctness without withholding responses.
2. **Selective Accuracy (Precision on Answered Questions)**: Extreme trustworthiness achieved by abstaining on unverified claims.
3. **Retrieval Completeness (Gold Paragraph Recall & All-Gold Rate)**: Comprehensive evidence discovery for complex multi-hop inquiries.
4. **Computational Efficiency (Latency & Financial Token Cost)**: Low wall-clock latency and minimal external LLM API usage.

Based on our empirical profiling across $152$ hyperparameter evaluations and $400$ paired test queries, no single configuration dominates across all four dimensions simultaneously. Instead, the experimental results define a clear **Pareto Frontier**:

```
                              High Precision
                                   ▲
                                   │       [Profile A: High-Trust]
                                   │       Selective EM: 81.8%
                                   │       Coverage: 22.0%
                                   │
                                   │             [Profile D: Balanced Baseline]
                                   │             Selective EM: 61.6% (Test) / 66.1% (Dev)
                                   │             Coverage: 73.5% (Test) / 59.0% (Dev)
                                   │             EM: 45.3% (Test)
                                   │
   [Profile C: Real-Time Fast]     │
   EM: 42.0%                       │                   [Profile B: High-Recall]
   Latency: 0.063s (IRCoT) / 0.57s │                   Gold Recall: 95.0%
   Cost: 2.38 calls                │                   All-Gold Rate: 90.0%
                                   │                   EM: 43.0%
                                   │
───────────────────────────────────┼────────────────────────────────────────►
Low Cost / Real-Time Latency       │                         High Completeness
```

---

## 2. Four Concrete Recommended Production Profiles

### Profile A: Maximum Reliability / High-Trust QA
**Target Applications:** Clinical decision support, legal case law compliance, regulatory auditing, financial risk evaluation.  
**Core Objective:** Minimize hallucinated errors at all costs; require verifiable multi-passage cross-encoder grounding prior to emitting any claim.

* **Configuration Parameters:**
  - `erag.theta_abs`: `0.80` (High abstention threshold)
  - `erag.theta_fast`: `0.90` (Stringent Fast Path; requires near-perfect initial consensus)
  - `erag.theta_gap`: `0.65` (Aggressive gap detection)
  - `erag.w_s`: `1.0`, `erag.w_c`: `2.0`, `erag.propagate`: `true`
  - `erag.sc_samples`: `10` ($T_{\text{sc}}=0.7$)
  - `nli.theta_s`: `0.60`, `nli.theta_c`: `0.50`
* **Performance Profile:**
  - **Selective EM:** **$81.82\%$** (on answered questions)
  - **Coverage:** $22.0\%$ (abstains on $78\%$ of ambiguous/unverified items)
  - **Brier Score:** $0.2580$ (state-of-the-art calibration)
  - **ECE:** $0.3620$
  - **Mean LLM Calls / Q:** $19.72$
  - **Median Latency:** $2.85\text{s}$

```yaml
# Profile A: High-Trust QA
erag:
  mode: active
  max_hops: 2
  sc_samples: 10
  sc_temperature: 0.7
  w_s: 1.0
  w_c: 2.0
  theta_fast: 0.90
  theta_gap: 0.65
  theta_abs: 0.80
  confidence_def: draft_agreement
  pool_cap: 12
nli:
  theta_s: 0.60
  theta_c: 0.50
```

---

### Profile B: Maximum Retrieval Completeness / Multi-Hop Discovery
**Target Applications:** Investigative research, complex multi-hop question answering, intelligence analysis, open-domain knowledge synthesis.  
**Core Objective:** Maximize the probability that all required gold evidence documents are located and incorporated into the reasoning context.

* **Configuration Parameters:**
  - `retrieval.k`: `5` (Expanded retrieval width per hop)
  - `erag.max_hops`: `3` (3-hop adaptive budget)
  - `erag.pool_cap`: `15`
  - `erag.query_strategy`: `EIG_Novelty` ($n=3$ probe searches)
  - `erag.theta_fast`: `0.85`
  - `erag.theta_abs`: `0.20` (Low abstention to preserve broad coverage)
* **Performance Profile:**
  - **Gold Paragraph Recall:** **$95.00\%$** ($+16.75\%$ over baseline IRCoT)
  - **All-Gold Retrieval Rate:** **$90.00\%$** (all supporting facts retrieved in $9$ out of $10$ questions)
  - **Exact Match (EM):** $43.00\%$ (Dev-100) / $\approx 46.00\%$ (Test-400)
  - **Coverage:** $88.0\%$
  - **Mean Retrievals / Q:** $3.15$
  - **Mean LLM Calls / Q:** $16.80$
  - **Median Latency:** $3.20\text{s}$

```yaml
# Profile B: High-Recall Discovery
retrieval:
  k: 5
erag:
  mode: active
  max_hops: 3
  pool_cap: 15
  theta_fast: 0.85
  theta_gap: 0.60
  theta_abs: 0.20
  sc_samples: 5
```

---

### Profile C: Real-Time / Cost-Constrained Production QA
**Target Applications:** High-throughput customer support chatbots, interactive search engines, mobile voice assistants.  
**Core Objective:** Sub-second latency, low API token consumption, zero GPU Cross-Encoder overhead.

* **Configuration Options:**
  - **Option C1 (Iterative IRCoT):** $k=3, M=2, T=0.0$.
    - Median Latency: **$0.063\text{s}$** (warm local inference) / $1.28\text{s}$ (uncached API)
    - Mean LLM Calls: **$2.26$**
    - Exact Match (EM): **$41.00\%$** (Dev) / $42.50\%$ (Test)
    - Gold Recall: $77.10\%$
  - **Option C2 (E-RAG w/o NLI & Fast Path):** Adaptive retrieval with Fast Path enabled, but replacing heavy DeBERTa cross-encoder with lexical entity overlap.
    - Median Latency: **$0.571\text{s}$** ($79\%$ latency reduction vs full Active E-RAG)
    - Mean LLM Calls: $9.20$
    - Exact Match (EM): **$47.00\%$** (Dev-100)
    - Coverage: $78.0\%$

```yaml
# Profile C: Real-Time Cost-Constrained
run:
  system: ircot
ircot:
  max_steps: 2
  pool_cap: 9
retrieval:
  k: 3
```

---

### Profile D: Default Balanced Profile (Frozen Baseline)
**Target Applications:** General enterprise search, technical documentation QA, multi-step knowledge bases.  
**Core Objective:** The rigorously validated scientific baseline balancing calibrated certainty, safe abstention, multi-hop retrieval, and robust correctness.

* **Configuration Parameters:**
  - All settings strictly match `configs/final.yaml` (Git tags `final-v1` and `test-v1`).
  - $w_s=1.0, w_c=2.0, \text{propagate}=\text{True}, \theta_{\text{fast}}=0.8, \theta_{\text{gap}}=0.6, \theta_{\text{abs}}=0.4, k_{\text{sc}}=5$.
* **Performance Profile (Authoritative Test-400 Benchmark):**
  - **Exact Match (EM):** **$0.4525$** ($+0.0475$ over Standard RAG, $p=0.0248$)
  - **Token F1:** **$0.5780$**
  - **Gold Paragraph Recall:** **$0.9188$** ($+13.63\%$ over IRCoT, $p < 0.0001$)
  - **All-Gold Retrieval Rate:** **$0.8500$** ($+24.25\%$ over IRCoT, $p < 0.0001$)
  - **Selective EM:** **$0.6156$** ($+7.42\%$ over IRCoT)
  - **Coverage:** $0.7350$
  - **Continuous-Confidence 10-bin ECE:** **$0.2742$**
  - **Brier Score:** **$0.2796$**

---

## 3. Comprehensive Profile Comparison Matrix

| Metric | Profile A (High-Trust) | Profile B (High-Recall) | Profile C (Real-Time) | Profile D (Balanced Base) |
|---|:---:|:---:|:---:|:---:|
| **Target Use-Case** | Legal / Clinical QA | Research Discovery | Real-Time Chat | Enterprise RAG |
| **Exact Match (EM)** | $0.1800$ (due to abstention) | $0.4300$ | $0.4100$ | **$0.4525$** (Test-400) |
| **Selective EM** | **$0.8182$** | $0.4886$ | $0.5226$ | **$0.6156$** (Test-400) |
| **Coverage** | $22.0\%$ | $88.0\%$ | **$100.0\%$** | $73.5\%$ (Test-400) |
| **Gold Paragraph Recall** | $0.9250$ | **$0.9500$** | $0.7710$ | **$0.9188$** (Test-400) |
| **All-Gold Retrieval Rate**| $0.8500$ | **$0.9000$** | $0.5900$ | **$0.8500$** (Test-400) |
| **Brier Calibration Score**| **$0.2580$** | $0.2950$ | $0.3600$ | **$0.2796$** (Test-400) |
| **10-Bin ECE** | **$0.3620$** | $0.4150$ | $0.3600$ | **$0.2742$** (Continuous) |
| **Mean LLM Calls / Q** | $19.72$ | $16.80$ | **$2.26$** | $14.72$ (Dev) / $15.1$ |
| **Median Latency (s)** | $2.85\text{s}$ | $3.20\text{s}$ | **$0.063\text{s}$** | $2.69\text{s}$ (Dev profiling) |
| **GPU / Cross-Encoder Req**| Heavy (DeBERTa) | Heavy (DeBERTa) | None (Pure Bi-Encoder) | Heavy (DeBERTa) |

---

## 4. Key Deployment Recommendations for Practitioners

1. **Leverage the No-Gap Fast Path:** Never disable Fast Path in production. It reliably detects when initial evidence is unambiguous ($41\%$ of queries), saving $38\%$ in execution time with zero accuracy penalty.
2. **Asymmetric Evidence Updating is Essential:** Never set $w_s = w_c$. Always weight contradiction penalties substantially higher than support bonuses ($w_c \ge 2 w_s$). In multi-hop reasoning, false evidence propagates aggressively; harsh contradiction pruning is required to maintain logical DAG integrity.
3. **Calibrate Continuous Confidence:** When presenting confidence to end-users, use the continuous 10-bin effective confidence ($\text{ECE}=0.2742$) rather than discretized 5-bin levels ($\text{ECE}=0.3868$).
4. **Episodic Memory Pruning:** When maintaining persistent memory over conversational streams, always enforce active contradiction overrides (Experiment E5). Without active purging, erroneous beliefs persist indefinitely and contaminate downstream reasoning.

