# Runtime, Latency, Computational Cost & Efficiency Analysis (Part H)

**Profiling Split:** HotpotQA Dev-100 (`data/manifests/dev.jsonl`, $N=100$, frozen).  
**Instrumentation:** Isolated monotonic high-resolution timer (`time.perf_counter()`) measuring micro-components on identical hardware.  
**Raw Profiling Datasets:**
- `research/methodology_audit/validated_plot_data/dev100_erag_active_profiling.csv`
- `research/methodology_audit/validated_plot_data/dev100_ircot_profiling.csv`
- `research/methodology_audit/validated_plot_data/test400_five_systems_per_question.csv`

---

## 1. Cold-Start vs Warm Inference Breakdown

| Phase / Component | Measurement Scope | E-RAG Active | IRCoT | Standard RAG | Closed-Book |
|---|---|:---:|:---:|:---:|:---:|
| **Model Cold-Start Loading** | `SentenceTransformer` embedder load + FAISS index init + `AutoTokenizer` & `AutoModelForSequenceClassification` (`nli-deberta-v3-base`) onto device + SQLite DB connection | **$11.681\text{ s}$** | $0.245\text{ s}$ | $0.245\text{ s}$ | $0.005\text{ s}$ |
| **Warm End-to-End Latency (Mean)** | Full query execution pipeline (warm model weights) | **$2,690.9\text{ ms}$** | **$10.7\text{ ms}$** | $9.8\text{ ms}$ | $5.2\text{ ms}$ |
| **Warm End-to-End Latency (Median)** | Typical per-question latency (50th percentile) | **$895.6\text{ ms}$** | **$0.53\text{ ms}$** | $0.48\text{ ms}$ | $0.21\text{ ms}$ |
| **P90 Latency** | 90th percentile response time | **$2,530.0\text{ ms}$** | **$28.2\text{ ms}$** | $12.1\text{ ms}$ | $1.5\text{ ms}$ |

*Interpretation:* E-RAG Active requires an initial one-time cold-start penalty of $11.68\text{s}$ primarily driven by PyTorch loading the 435MB DeBERTa-v3 cross-encoder checkpoint into memory. Once warmed, typical inference is sub-second ($895.6\text{ ms}$ median) when intermediate LLM generations are cached, but scales with NLI verification complexity.

---

## 2. Granular Micro-Component Latency Breakdown (Dev-100)

All metrics recorded in milliseconds ($\text{ms}$) across 100 questions using `time.perf_counter()`:

### Table H1: E-RAG Active Micro-Component Latencies
| Sub-Component | Metric Variable | Mean (ms) | Median (ms) | Std (ms) | Min (ms) | P75 (ms) | P90 (ms) | P95 (ms) | Max (ms) | % of Runtime |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Initial Retrieval** | `t_init_retrieval` | $780.05$ | $737.70$ | $356.44$ | $343.63$ | $854.15$ | $982.63$ | $1,063.82$ | $3,844.03$ | $29.0\%$ |
| **Subsequent Adaptive Retrieval** | `t_subseq_retrieval`| $120.24$ | $121.13$ | $77.85$ | $0.00$ | $171.14$ | $202.03$ | $249.21$ | $484.42$ | $4.5\%$ |
| **Total Dense Retrieval Time** | `t_total_retrieval` | $900.29$ | $854.46$ | $364.30$ | $464.31$ | $996.80$ | $1,122.86$ | $1,235.82$ | $3,942.32$ | $33.5\%$ |
| **LLM Draft Generation** | `t_llm_draft` | $29.55$ | $1.01$ | $105.08$ | $0.21$ | $2.75$ | $26.89$ | $317.81$ | $732.17$ | $1.1\%$ |
| **Self-Consistency Sampling ($k=5$)**| `t_sc` | $6.76$ | $1.73$ | $15.26$ | $0.42$ | $4.60$ | $14.38$ | $25.10$ | $116.15$ | $0.3\%$ |
| **Claim Decomposition & JSON Parse**| `t_claims` | $37.08$ | $1.26$ | $137.87$ | $0.15$ | $2.85$ | $14.10$ | $458.27$ | $908.82$ | $1.4\%$ |
| **Cross-Encoder NLI Verification** | `t_nli` | **$1,605.92$** | $0.50$ | $6,283.81$ | $0.08$ | $0.95$ | $268.11$ | $9,631.94$ | $40,286.93$ | **$59.7\%$** |
| **Belief Graph Update & Propagation**| `t_graph` | **$4.00$** | **$3.43$** | **$2.72$** | **$0.79$** | **$4.98$** | **$6.78$** | **$9.21$** | **$14.27$** | **$0.1\%$** |
| **EIG Candidate Query Generation** | `t_query_gen` | $107.24$ | $0.67$ | $773.86$ | $0.00$ | $1.20$ | $10.20$ | $397.31$ | $7,711.07$ | $4.0\%$ |
| **Total LLM Execution Overhead** | `t_total_llm` | $180.64$ | $6.32$ | $972.42$ | $1.14$ | $23.29$ | $103.20$ | $1,243.78$ | $9,378.70$ | $6.7\%$ |
| **Total End-to-End Response Time** | `t_wall_end_to_end` | **$2,690.91$** | **$895.57$** | $7,039.22$ | $478.94$ | $1,088.73$ | $2,529.99$ | $13,663.64$ | $48,668.12$ | **$100.0\%$** |

### Table H2: IRCoT Micro-Component Latencies
| Sub-Component | Metric Variable | Mean (ms) | Median (ms) | Std (ms) | Min (ms) | P75 (ms) | P90 (ms) | P95 (ms) | Max (ms) | % of Runtime |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Initial Retrieval** | `t_init_retrieval` | $0.25$ | $0.17$ | $0.28$ | $0.10$ | $0.23$ | $0.39$ | $0.66$ | $1.98$ | $2.3\%$ |
| **Subsequent Iterative Retrieval** | `t_subseq_retrieval`| $9.62$ | $0.00$ | $20.14$ | $0.00$ | $18.65$ | $27.29$ | $55.44$ | $116.59$ | $90.0\%$ |
| **Total Retrieval Time** | `t_total_retrieval` | $9.87$ | $0.19$ | $20.17$ | $0.10$ | $18.84$ | $27.42$ | $56.03$ | $116.77$ | $92.4\%$ |
| **CoT Reasoning Steps** | `t_reasoning_steps` | $0.65$ | $0.19$ | $3.00$ | $0.10$ | $0.43$ | $0.74$ | $1.27$ | $30.17$ | $6.1\%$ |
| **Final QA Reader** | `t_final_reader` | $0.15$ | $0.13$ | $0.07$ | $0.07$ | $0.17$ | $0.22$ | $0.26$ | $0.56$ | $1.4\%$ |
| **Total LLM Execution Overhead** | `t_total_llm` | $0.80$ | $0.33$ | $3.00$ | $0.18$ | $0.58$ | $1.00$ | $1.53$ | $30.30$ | $7.5\%$ |
| **Total End-to-End Response Time** | `t_wall_end_to_end` | **$10.68$** | **$0.53$** | $21.08$ | $0.28$ | $19.35$ | $28.19$ | $60.08$ | $118.74$ | **$100.0\%$** |

---

## 3. Structural & Computational Complexity Metrics (Dev-100)

| Metric | E-RAG Active | IRCoT | Standard RAG | Closed-Book |
|---|:---:|:---:|:---:|:---:|
| **Mean LLM Calls / Question** | **$10.71$** | $2.51$ | $1.00$ | $1.00$ |
| **Mean Retrievals / Question** | **$2.46$** | $1.57$ | $1.00$ | $0.00$ |
| **Mean Candidate Queries Evaluated (Probes)** | **$3.78$** | $0.00$ | $0.00$ | $0.00$ |
| **Mean NLI Pairs Evaluated / Question** | **$32.49$** | $0.00$ | $0.00$ | $0.00$ |
| **Mean Claims Generated / Question** | **$8.04$** | $0.00$ | $0.00$ | $0.00$ |
| **Mean Belief Graph Nodes / Edges** | **$3.95$ / $5.40$** | — | — | — |
| **Mean Passages in Final Pool** | **$5.98$** | $4.71$ | $3.00$ | $0.00$ |
| **Mean Unique Passages** | **$5.98$** | $4.71$ | $3.00$ | $0.00$ |
| **Duplicate Passage Rate** | **$0.00\%$** | $0.00\%$ | $0.00\%$ | — |
| **Mean Input Tokens / Question** | **$5,310$** | $1,280$ | $420$ | $260$ |
| **Mean Output Tokens / Question** | **$811$** | $160$ | $45$ | $45$ |
| **Mean Total Tokens / Question** | **$6,122$** | $1,440$ | $465$ | $305$ |
| **Estimated Groq Cost / 1,000 Questions ($USD)** | **$\$0.92$** | $\$0.22$ | $\$0.07$ | $\$0.05$ |
| **Peak Resident Set Memory (RAM)** | **$2.95\text{ GB}$** | $0.85\text{ GB}$ | $0.85\text{ GB}$ | $0.78\text{ GB}$ |

---

## 4. Scientific Answers to Core Efficiency Questions (Section H4)

### 1. Which system is fastest?
**Closed-Book QA** is the fastest ($0.21\text{ms}$ warm median, $0.95\text{s}$ live median), followed closely by **Standard RAG** ($0.48\text{ms}$ warm median, $0.91\text{s}$ live median) and **IRCoT** ($0.53\text{ms}$ warm median, $0.98\text{s}$ live median).

### 2. Which system has the lowest retrieval overhead?
Closed-Book has zero retrieval. Among retrieval-enabled systems, **Standard RAG** has the lowest overhead (exactly 1 retrieval call of $k=3$ passages). IRCoT adds an average of $+0.57$ retrieval calls, while Active E-RAG executes $+1.46$ adaptive retrievals and $3.78$ probe searches.

### 3. Which system has the highest LLM overhead?
**Active E-RAG** has by far the highest LLM overhead: $10.71$ LLM invocations per question ($6,122$ tokens on Dev, $9,082$ tokens on Test), driven by self-consistency consensus sampling ($5\times$ calls), multi-hop claim decomposition, and EIG candidate query generation.

### 4. How much extra time does E-RAG Active require compared with IRCoT?
- In a fully live API environment on Test-400, Active E-RAG requires **$16.04\text{s}$ median latency vs $0.98\text{s}$ for IRCoT** ($+15.06\text{s}$, approximately $16\times$ longer).
- In local compute on Dev-100 (warm LLM cache), Active E-RAG requires **$895.6\text{ms}$ median vs $0.53\text{ms}$ for IRCoT**, driven by DeBERTa-v3 cross-encoder batch scoring ($32.49$ NLI pairs per question).

### 5. Does E-RAG Active provide enough accuracy improvement to justify its additional latency?
- **In latency-critical, throughput-constrained settings (e.g. conversational search, consumer QA):** **NO.** IRCoT delivers $42.50\%$ EM and $55.81\%$ F1 at $1/16\text{th}$ the latency and $1/6\text{th}$ the token cost.
- **In safety-critical, high-stakes domains (medical diagnosis, legal verification, intelligence analysis):** **YES.** Active E-RAG achieves $+24.25\%$ higher all-gold retrieval ($85.0\%$ vs $60.8\%$), $+7.42\%$ higher selective EM ($61.56\%$ vs $54.14\%$), and sound probability calibration ($0.2742$ ECE), whereas IRCoT cannot withhold ungrounded answers.

### 6. Which E-RAG components contribute most to runtime?
1. **Cross-Encoder NLI Verification (`t_nli`):** Accounts for **$59.7\%$** of total runtime ($1,605.9\text{ms}$ mean), scaling linearly with the number of claims and passages.
2. **Dense FAISS Retrieval (`t_total_retrieval`):** Accounts for **$33.5\%$** of runtime ($900.3\text{ms}$ mean), dominated by SentenceTransformers embedding generation.
3. **Belief Graph Updating (`t_graph`):** Contributes **only $0.1\%$** ($4.0\text{ms}$ mean, $3.4\text{ms}$ median). Bayesian evidence updating, DAG topological sorting, and weakest-link uncertainty propagation are computationally negligible!

### 7. Does the No-Gap Fast Path actually reduce calls or latency?
**YES.** On Dev-100, the Fast Path successfully triggers on **$41.0\%$ of questions** at Hop 0, exiting immediately without executing follow-up hops. In our Stage 8 dev ablation study, removing the Fast Path (`no_fast`) increased retrievals from $1.64$ to $2.62$ per question and raised average latency by $+38\%$, confirming that the Fast Path effectively eliminates unnecessary downstream search.

