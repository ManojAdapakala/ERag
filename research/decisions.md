# E-RAG Architectural and Experimental Decisions

## D-001: Model and Inference Provider Selection
- **Date:** 2026-11-05
- **Status:** APPROVED
- **Context:** The implementation guide specifies Groq as the fast inference provider and `openai/gpt-oss-20b` (or `llama-3.3-70b-versatile` if required) as the primary generation model.
- **Decision:** Use Groq API with `openai/gpt-oss-20b` as specified in `configs/base.yaml`.
- **Reasoning:** High inference throughput (tokens/sec) on Groq makes multi-hop iterative RAG and self-consistency sampling ($k=5$) computationally tractable within practical research budgets.

## D-002: Confidence Estimation Mechanism
- **Date:** 2026-11-05
- **Status:** APPROVED
- **Context:** Hosted APIs frequently lack token log-probabilities or return uncalibrated log-probs across different architectures.
- **Decision:** Implement self-consistency sampling ($k=5$, temperature $0.7$) as the empirical confidence estimator $c_0$, evaluating both `majority` agreement and `draft_agreement` on dev.
- **Reasoning:** Empirical sampling confidence produces discrete, verifiable probability estimates ($[0.2, 0.4, 0.6, 0.8, 1.0]$) that integrate directly into calibration metrics (ECE, AUROC).

## D-003: Dense Embedding and Similarity Search Architecture
- **Date:** 2026-11-05
- **Status:** APPROVED
- **Context:** Baseline RAG and E-RAG require efficient dense retrieval over both distractor contexts ($10$ paragraphs/query) and pooled corpora ($\approx 5,000$ paragraphs).
- **Decision:** Use `sentence-transformers/all-MiniLM-L6-v2` with FAISS `IndexFlatIP` on $L_2$-normalized passage embeddings.
- **Reasoning:** Normalizing vectors before inner-product calculation yields exact cosine similarity with minimal CPU overhead, guaranteeing deterministic ranking across runs.

## D-004: Decoupled Observational Mode (Gate G1) and Abstention Safeguard
- **Date:** 2026-11-05
- **Status:** APPROVED
- **Context:** Naive epistemic abstention in earlier phases dropped Exact Match from $0.56$ to $0.28$ because intermediate multi-hop claims were falsely treated as grounds for answering `"unknown"`.
- **Decision:** 
  1. Enforce Observational Mode (`erag.mode=observe`), where the system performs claim extraction, NLI verification, and graph updates, records `would_abstain`, but strictly returns the uninhibited baseline draft.
  2. In Active Mode (`erag.mode=active`), abstain only if the final answer-bearing claim's effective confidence $< \theta_{abs}$ ($0.4$) or status is `"contradicted"`. Intermediate unresolved claims trigger adaptive search, not abstention.
- **Reasoning:** Prevents premature abstention penalties on multi-hop questions while preserving auditable selective accuracy.

## D-005: Cache Key Collision Prevention
- **Date:** 2026-11-05
- **Status:** APPROVED
- **Context:** Generating multiple self-consistency samples for the same prompt requires distinct cache entries.
- **Decision:** Compute SHA-256 cache keys over `{namespace, model, messages, temperature, max_tokens, reasoning_effort, sample_idx}` in `src/llm/client.py`.
- **Reasoning:** Prevents identical cached completions across the $k=5$ self-consistency samples while allowing zero-cost instant replay for deterministic evaluation runs.

