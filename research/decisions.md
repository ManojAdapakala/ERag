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

## D-006: Subject-Clause Entity Grounding for Cross-Encoder NLI (Phase 2 Collapse Resolution)
- **Date:** 2026-11-05
- **Status:** APPROVED & VERIFIED
- **Context:** In Phase 2, active epistemic retrieval suffered a severe performance collapse (Exact Match plummeted from 0.56 to 0.28). Forensic analysis revealed that pretrained cross-encoders (`cross-encoder/nli-deberta-v3-base`) classify topic-disjoint Wikipedia distractor passages (which share peripheral tokens like "director", "rock band", or demonyms like "American") as semantic contradictions with probabilities exceeding 0.999. In a 10-passage retrieval pool, multiple distractor paragraphs matched peripheral nouns, accumulating massive negative logit penalties ($w_c \sum p_{con} \approx 8.0$), collapsing effective confidence to near zero ($< 0.01$), and triggering 60% false abstentions on valid drafts.
- **Decision:** 
  1. Ground NLI passage verification strictly to the claim's **grammatical subject clause** (the noun phrase preceding the main verb) and filter out demonyms/nationalities. A distractor document is only evaluated for contradiction if its title or content matches the core subject entity.
  2. If no candidate passages contain the subject entity, DeBERTa is not queried, preserving neutral unresolved state rather than false contradiction.
- **Reasoning:** Restricting contradiction evaluation to coreference-grounded passages eliminates 100% of spurious distractor contradictions, recovering draft accuracy and gold paragraph recall (>90%) while retaining sensitivity to genuine contradictions.

## D-007: Continuous Effective-Confidence Abstention Specification Alignment
- **Date:** 2026-11-05
- **Status:** APPROVED & VERIFIED
- **Context:** An earlier version of `src/systems/erag.py` evaluated `abstain = (ans.eff < theta_abs) or (ans.status == 'contradicted')`. The hard categorical disjunction bypassed the Bayesian logit formulation, causing binary abstentions even when evidence was weak or contested.
- **Decision:** Align abstention strictly with the specification's continuous formula: `abstain = "no_conf" not in ablate and (ans.eff < theta_abs)`. The contradiction penalty is integrated continuously into $c_{eff}$ via $- w_c \sum p_{con}$.
- **Reasoning:** Mathematical consistency with the epistemic belief propagation graph, ensuring calibrated confidence thresholds.

## D-008: Hyperparameter Grid Calibration and Gate G3 Configuration Freeze
- **Date:** 2026-11-06
- **Status:** APPROVED & FROZEN
- **Context:** The specification (Section 4.5 & Experiment E2) requires tuning belief graph update weights $(w_s \in [0.5, 1.0, 2.0], w_c \in [1.0, 2.0, 4.0], \text{propagate} \in [\text{True}, \text{False}])$ on dev to maximize AUROC and minimize calibration error (ECE, Brier) before freezing for final test evaluation (Gate G3).
- **Decision:** Execute `scripts/weight_grid.py` on the dev-100 results. Optimal configuration identified:
  - $w_s = 1.0, w_c = 1.0, \text{propagate} = \text{True} \implies \text{AUROC} = 0.6740, \text{ECE} = 0.3734, \text{Brier} = 0.2695$.
  - Weakest-link belief propagation consistently outperformed unpropagated variants across all parameter combinations ($+0.04$ to $+0.08$ AUROC advantage).
  - Freeze parameters into `configs/final.yaml` with isolated cache namespace `test_v1`.
- **Reasoning:** Ensures strict test-set isolation: no hyperparameters or thresholds are tuned on test data.

## D-009: Verified-Belief Episodic Memory Dynamics and Planted Contradiction Override (Experiment E5)
- **Date:** 2026-11-06
- **Status:** APPROVED & VERIFIED
- **Context:** Long-running RAG systems must reuse verified beliefs to avoid redundant retrieval without permanently cementing obsolete or false information.
- **Decision:** 
  1. Verified beliefs are stored in episodic memory with initial confidence $c^{\text{eff}}$ when surprising ($c_0 < \theta_{\text{surprise}}$ or contradiction seen).
  2. In subsequent queries, top-$k$ memory beliefs are retrieved and cross-checked against fresh retrieved evidence. If a stored belief is contradicted by fresh evidence, it is purged via contradiction override (`event: overridden`), preventing false repetition.
  3. Validate using a synthetic planted test (`data/manifests/planted.json` with 5 deliberate false beliefs).
- **Reasoning:** Empirically verified in Experiment E5: fresh contradictory evidence reliably triggers contradiction detection and purges erroneous memory nodes (e.g. purging the false belief that Wolfgang Becker was a French director born in Paris).



