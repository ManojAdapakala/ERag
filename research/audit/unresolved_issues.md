# Unresolved Research Questions, Technical Deficits & Open Challenges (Phase 0 Audit)

**Date:** 2026-11-07  
**Auditor:** Senior AI Research Engineer & RAG Architect  
**Purpose:** Identify technical debt, efficiency bottlenecks, methodological gaps, and open research challenges prior to Phase 1.

---

## 1. Critical Computational Expenditure in E-RAG Active

### Issue Description:
- E-RAG Active consumes an average of **$14.22$ LLM calls per question** on Test-400 (vs $2.51$ for IRCoT and $1.00$ for Standard RAG).
- **Primary Cost Drivers:**
  1. *Self-Consistency Sampling:* Generating $k=5$ samples at $T=0.7$ for initial confidence ($c_0$) consumes 5 API completions per question. When adaptive retrieval triggers a hop ($hop > 0$), self-consistency is executed **a second time**, adding another 5 completions. Together, self-consistency accounts for **$70\%$ of all LLM calls** ($10 / 14.22$).
  2. *Duplicate Verification Steps:* Re-extracting claims and re-evaluating NLI for already-verified claims after each retrieval hop adds redundant API calls and NLI computation.
  3. *Unconstrained Query Generation:* Generating 3 candidate search queries via LLM prompts for every gap adds 1 completion per hop.

---

## 2. Cross-Encoder CPU Latency Bottleneck

### Issue Description:
- On CPU without CUDA acceleration (Intel i5-1035G1), unmemoized NLI inference (`cross-encoder/nli-deberta-v3-base`) accounts for **$59.7\%$ of total execution latency** ($1,605.92\text{ ms}$ mean per question on Dev-100).
- While warm evaluation benefits from `results/nli_memo.pkl`, cold evaluation or new query streams encounter substantial latency when verifying multiple 2-sentence windows across 3 to 12 passages.

---

## 3. Low Episodic Memory Reuse Rate (6.7%)

### Issue Description:
- In the 30-question evaluation stream (`data/manifests/stream.jsonl`), memory reuse occurred on only **$2 / 30$ questions** ($6.7\%$).
- **Identified Causes:**
  1. *High Similarity Threshold:* The retrieval gating threshold $\theta_{\text{mem}} = 0.75$ is strict for dense SentenceTransformer embeddings between distinct sentence forms.
  2. *Surprise Gating Restriction:* New beliefs are written to memory only if initial confidence $c_0 < 0.60$ or a contradiction was detected. Confident multi-hop facts ($c_0 \ge 0.60$) are excluded, preventing the memory store from accumulating valuable entity facts.
  3. *Stream Design:* While 30 questions cluster around specific entities (e.g., German cinema, film directors), multi-hop questions require distinct bridge attributes that may not overlap sufficiently.

---

## 4. Multi-Retriever Generalization Deficit

### Issue Description:
- To date, all benchmarks have operated on a single dense retriever: `sentence-transformers/all-MiniLM-L6-v2` + FAISS `IndexFlatIP`.
- It remains unverified whether E-RAG's epistemic advantage holds when using:
  1. Sparse lexical retrieval (**BM25** via Elasticsearch or rank-bm25).
  2. State-of-the-art dense embeddings (**E5-base-v2** and **BGE-base-en-v1.5**).
- Without a multi-retriever benchmark matrix, reviewers may question whether E-RAG's multi-hop recovery is merely compensating for MiniLM's retrieval limitations.

---

## 5. Cross-Dataset Generalization Deficit

### Issue Description:
- The authoritative benchmark is currently confined to **HotpotQA**.
- Secondary multi-hop benchmarks (**2WikiMultiHopQA** and **MuSiQue**) have not yet been evaluated end-to-end with frozen manifests and comparative tables.
- Evaluator scripts and paragraph construction logic must be validated for 2WikiMultiHopQA and MuSiQue to assess generalizability across disparate multi-hop reasoning topologies.

---

## 6. Statistical Precision Distinction (EM vs Token F1)

### Issue Description:
- On Test-400, Active E-RAG demonstrates a statistically significant improvement over Standard RAG in **Exact Match** ($+4.75\%$, $p = 0.0248 < 0.05$, $95\%$ CI: $[+0.010, +0.088]$).
- However, for **Token F1**, the improvement ($+3.95\%$) does **not** achieve statistical significance at the $95\%$ level because the bootstrap confidence interval crosses zero ($[-0.002, +0.082]$).
- **Scientific Framing Requirement:** The final research paper must strictly distinguish between the significant Exact Match gain and the non-significant Token F1 point estimate, avoiding over-claiming.

