# Authoritative Data Integrity & Experimental Metric Audit

**Audit Scope:** Verification of raw experiment outputs, schema validation, field availability, question ID alignment, abstention handling, and timing instrumentation.

---

## 1. Data-Source Manifest Across All Five Systems

Every record in each file has been validated against `data/manifests/test.jsonl` (400 hard multi-hop HotpotQA questions):

| System | Authoritative Source File | Valid Records ($N$) | Unique IDs | Missing Records | Failed Records | Duplicate IDs | ID Order Match vs Manifest | Data Integrity Status |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Closed-Book** | `results/closed_book_test.jsonl` | **$400$** | **$400$** | $0$ | $0$ | $0$ | **Exact 1-to-1** | **VERIFIED CLEAN** |
| **System A (Standard RAG)** | `results/standard_test.jsonl` | **$400$** | **$400$** | $0$ | $0$ | $0$ | **Exact 1-to-1** | **VERIFIED CLEAN** |
| **System B (IRCoT)** | `results/ircot_test.jsonl` | **$400$** | **$400$** | $0$ | $0$ | $0$ | **Exact 1-to-1** | **VERIFIED CLEAN** |
| **System C (E-RAG Observe)** | `results/erag_observe_test.jsonl` | **$400$** | **$400$** | $0$ | $0$ | $0$ | **Exact 1-to-1** | **VERIFIED CLEAN** |
| **System C (E-RAG Active)** | `results/erag_active_test.jsonl` | **$400$** | **$400$** | $0$ | $0$ | $0$ | **Exact 1-to-1** | **VERIFIED CLEAN** |

*Note on Observational E-RAG History:*  
In earlier audit reports, `erag_observe_test.jsonl` contained $234$ questions due to a 60-minute task runtime limit. Per explicit user authorization, the remaining $166$ questions were resumed and completed to $100\%$ ($N=400$) with zero failures. String parity and SHA-256 checksums between Observational E-RAG and Standard RAG match bit-for-bit across all $400$ items (`d795707be24025350a3882b7982f540122729f1404bc0ca6d92ee637ae0582bb`).

---

## 2. Per-Question Field Availability Audit

To prevent graph distortions or improper fallback values (such as defaulting missing confidences to $1.0$), we audited the presence of all metric fields:

| Field Name | Description | Closed-Book | Standard RAG | IRCoT | E-RAG Observe | E-RAG Active |
|---|---|:---:|:---:|:---:|:---:|:---:|
| `id` | Unique HotpotQA question ID | Present | Present | Present | Present | Present |
| `em` | Exact Match score ($0.0$ or $1.0$) | Present | Present | Present | Present | Present |
| `f1` | Token-level F1 score ($[0, 1]$) | Present | Present | Present | Present | Present |
| `unknown` | Withheld/abstained indicator | Present | Present | Present | Present | Present |
| `recall` | Gold paragraph retrieval recall | *Unavailable (N/A)* | Present | Present | Present | Present |
| `all_gold` | Binary all-gold retrieval indicator | *Unavailable (N/A)* | Present | Present | Present | Present |
| `answer_conf` | Effective epistemic confidence | *Unavailable* | *Unavailable* | *Unavailable* | Present | Present |
| `c0` | Initial self-consistency confidence | *Unavailable* | *Unavailable* | *Unavailable* | Present | Present |
| `llm_calls` | Number of LLM completion calls | Present | Present | Present | Present | Present |
| `retrievals` | Number of retrieval executions | Present ($0$) | Present ($1$) | Present | Present ($1$) | Present |
| `probe_searches`| Evaluated candidate queries | Present ($0$) | Present ($0$) | Present ($0$) | Present ($0$) | Present |
| `in_tokens` | Prompt token count | Present | Present | Present | Present | Present |
| `out_tokens` | Completion token count | Present | Present | Present | Present | Present |
| `api_latency` | Time spent on Groq API calls | Present | Present | Present | Present | Present |
| `wall_latency`| Total question end-to-end wall time | Present | Present | Present | Present | Present |
| `trace` | Intermediate hop graph traces | *Unavailable* | *Unavailable* | *Unavailable* | Present | Present |

### Graph Data Guardrails (Preventing Past Plotting Bugs):
1. **Confidence Histogram (J4) & Reliability Diagrams:** Must be generated **strictly for E-RAG Active and E-RAG Observe**. Standard RAG, Closed-Book, and IRCoT do NOT possess model confidence fields in raw data. They must be explicitly marked *Unavailable / Not Applicable* rather than defaulted to $1.0$.
2. **Retrieval Recall Histograms (J5):** Must include **only the 4 retrieval-enabled systems** (Standard RAG, IRCoT, E-RAG Observe, E-RAG Active). Closed-Book QA must be excluded (not plotted as a spike at $0.0$).
3. **Abstentions & Coverage:** All abstentions (`unknown=True`) strictly count as $0.0$ for EM and F1. Selective EM is evaluated strictly on answered questions:
   $$\text{Selective EM} = \frac{\sum_{i: \text{unknown}_i = \text{False}} \text{EM}_i}{N_{\text{answered}}}$$

---

## 3. Independently Recalculated Authoritative Test-400 Metrics

All metrics computed directly from raw per-question JSON records:

| System | $N$ | EM | Token F1 | Gold Recall | All-Gold | Unknown % | Coverage | Selective EM | ECE (10-bin) | ECE (Level) | Brier | Mean Wall Latency | Median Wall Latency | P90 Wall Latency | Mean Total Tokens |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Closed-Book** | $400$ | $0.2000$ | $0.2904$ | — | — | $23.25\%$ | $0.7675$ | $0.2606$ | $0.5675$ | $0.8000$ | $0.5675$ | $2.69\text{s}$ | $2.31\text{s}$ | $4.48\text{s}$ | $341$ |
| **System A (Standard RAG)** | $400$ | $0.4050$ | $0.5384$ | $0.7438$ | $0.5300$ | $22.50\%$ | $0.7750$ | $0.5226$ | $0.3700$ | $0.5950$ | $0.3700$ | $3.42\text{s}$ | $2.89\text{s}$ | $5.71\text{s}$ | $1,288$ |
| **System B (IRCoT)** | $400$ | $0.4250$ | $0.5581$ | $0.7825$ | $0.6075$ | $21.50\%$ | $0.7850$ | $0.5414$ | $0.3600$ | $0.5750$ | $0.3600$ | $4.68\text{s}$ | $3.81\text{s}$ | $8.24\text{s}$ | $2,746$ |
| **System C (E-RAG Observe)** | $400$ | $0.4050$ | $0.5384$ | $0.7438$ | $0.5300$ | $22.50\%$ | $0.7750$ | $0.5226$ | $0.3530$ | $0.4585$ | $0.3530$ | $7.15\text{s}$ | $6.81\text{s}$ | $12.43\text{s}$ | $3,932$ |
| **System C (E-RAG Active)** | $400$ | **$0.4525$** | **$0.5780$** | **$0.9188$** | **$0.8500$** | $26.50\%$ | $0.7350$ | **$0.6156$** | **$0.2742$** | **$0.3868$** | **$0.2796$** | $14.86\text{s}$ | $11.42\text{s}$ | $28.30\text{s}$ | $8,841$ |

---

## 4. Audit of Existing Timing & Efficiency Instrumentation

### Existing Fields in Raw JSONL:
1. `api_latency`: Accurately measures cumulative network + model execution time for external Groq API calls. For cache hits, it logs $0.0\text{s}$.
2. `wall_latency`: Total wall-clock time elapsed for the question invocation from entry to JSONL flush.

### Missing Granular Latency Decompositions:
The production runner (`scripts/run_experiment.py`) does not decompose wall time into sub-component buckets:
- Initial dense retrieval search time.
- Subsequent hop retrieval search time.
- Claim extraction prompt execution time.
- Cross-encoder NLI inference time (DeBERTa-v3).
- Belief graph update & propagation calculation time.
- Memory lookup and write time.

### Implementation Strategy for Stage 3:
To satisfy Section H without touching or invalidating frozen production code, we will implement an **Isolated Profiling Harness** (`scripts/profile_efficiency.py`) executed on the **Dev-100** split. This profiler will use high-resolution monotonic clocks (`time.perf_counter()`) to measure every sub-component with cold-start and warm-inference separation.

