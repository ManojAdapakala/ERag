# System Comparison & Methodology Differences

**Date:** 2026-11-05  
**Document:** Methodology & Implementation Variations (Stage 6 Acceptance Artifact)  
**Primary Specification:** *E-RAG Implementation Guide: End to End* (§10, p. 27–28)  

---

## 1. System B (IRCoT) Differences from Original Trivedi et al. (2022)

The table below documents the architectural and implementation differences between the original IRCoT publication and this research evaluation environment:

| Aspect | Original IRCoT (Trivedi et al., ACL 2023) | This Implementation (`src/systems/ircot.py`) | Methodological Justification |
| :--- | :--- | :--- | :--- |
| **Retriever** | BM25 sparse lexical retriever over a large static Wikipedia dump | Dense embedding retrieval (`sentence-transformers/all-MiniLM-L6-v2` + FAISS `IndexFlatIP`) | Ensures identical retriever architecture and representation across System A, B, and C. |
| **Retrieval Corpus** | Full Wikipedia dump ($>5\text{M}$ documents) | Distractor setting (10 candidate paragraphs per HotpotQA item) & Pooled setting ($N$-item pooled corpus) | Strictly controls distractor competition and eliminates external indexing discrepancies. |
| **Prompting** | Few-shot in-context demonstrations with fixed exemplars | Zero-shot chain-of-thought instruction prompt (`STEP_SYSTEM`) | Eliminates demonstration selection bias across diverse multi-hop question types. |
| **Base LLM** | Flan-T5, GPT-3 (text-davinci-002, 175B) | `openai/gpt-oss-20b` via Groq Chat Completions API | Modern open-weight instruction-tuned LLM at temperature 0.0 with deterministic caching. |
| **Reader Model** | Separate specialized reader model prompt | Shared canonical `answer_messages` context reader prompt | Holds reader prompt strictly constant across System A, System B, and System C. |
| **Stopping Rule** | Heuristic sentence termination or fixed step ceiling | Explicit `"so the answer is"` termination token, empty output detector, or `max_steps = 4` | Prevents infinite retrieval loops while preserving natural reasoning completion. |
| **Paragraph Budget** | Variable passage pool size | Step cap = 4, pool cap = 12 paragraphs (`pool_cap: 12`) | Guarantees bounded context length within 512–2048 token LLM window limits. |

---

## 2. Experimental Fair Comparison Protocol

To ensure rigorous scientific validity:
1. **Retrieval Budget Alignment:** System B with step cap $s$ executes at most $1 + s$ retrievals; System C with hop limit $h$ executes at most $1 + h$ retrievals.
2. **Probe Search Accounting:** All auxiliary probe searches used by E-RAG to compute Expected Information Gain (EIG) are tracked explicitly via `tracker.probe_searches` so computational overhead is transparently accounted for.
3. **Identical Evaluator:** All three systems are scored with the identical HotpotQA evaluation implementation (`src/eval/metrics.py`).

