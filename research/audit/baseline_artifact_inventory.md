# Baseline Artifact Inventory & Provenance Verification (Phase 0 Audit)

**Date:** 2026-11-07  
**Auditor:** Senior AI Research Engineer & RAG Architect  
**Authoritative Commits & Tags:**
- **Gate G3 Dev Freeze:** Commit `3ec2e9b` | Git Tag: `final-v1`
- **Test-400 Benchmark Completion:** Commit `4cf4299` | Git Tag: `test-v1`
- **Final Methodology Audit & Visualizations:** Commit `7a3f6fb`
- **Active vs IRCoT Head-to-Head Evaluation:** Commit `ec50c8b`

---

## 1. Primary Dataset Manifests (`data/manifests/`)

| Manifest File | Target Split | Item Count | SHA-256 Checksum | Provenance & Description |
| :--- | :---: | :---: | :--- | :--- |
| `debug.jsonl` | `debug` | 50 | `a84cb93ea5e9a4f47514a6755ec5dfbc1c78ebfc13783935db58178c7793b821` | Exact 50 HotpotQA question IDs matching historical Phase 1 baseline (`data/raw/debug50_ids.txt`). |
| `dev.jsonl` | `dev` | 100 | `f80f33f07a16f272c7a4d5e94b29f0322b07548fa4429df30df45aa1ce39f20c` | Stratified 100-item development split used for all parameter tuning and ablation sweeps. |
| `test.jsonl` | `test` | 400 | `187ff1b17b6bfa62524a87e50c766f77f3945a6c518bb37e7210e7b8f0ca4fba` | Stratified 400-item test benchmark split. Strictly isolated from hyperparameter tuning. |
| `stream.jsonl` | `stream` | 30 | `ca9fc213ba6b89693ebad6f03d50aa8d58a5da1fdf6db61e69b59e74653a9926` | 30-item episodic memory stream featuring recurring entity clusters. |
| `planted.json` | `planted` | 5 | `4bfb5eafe51ecaa7c6f057bb081b2a9261a7a13c9fbfa7e93f3d2f9547ea8d41` | 5 synthetic erroneous beliefs planted to verify contradiction-driven memory deletion. |

---

## 2. Benchmark Prediction Datasets (`results/`)

All files verified for structural integrity, JSON validity, exact item counts, and zero missing records:

| Result Artifact Path | Answering System | Split | Verified $N$ | Valid % | File Size (Bytes) | Verification Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| `results/closed_book_debug.jsonl` | Closed-Book | debug | 50 | 100% | 43,115 | Verified complete |
| `results/closed_book_dev.jsonl` | Closed-Book | dev | 100 | 100% | 86,210 | Verified complete |
| `results/closed_book_test.jsonl` | Closed-Book | test | 400 | 100% | 344,840 | Verified complete |
| `results/standard_debug.jsonl` | Standard RAG | debug | 50 | 100% | 49,820 | Verified complete |
| `results/standard_dev.jsonl` | Standard RAG | dev | 100 | 100% | 99,640 | Verified complete |
| `results/standard_dev_rep.jsonl` | Standard RAG (Repeat) | dev | 100 | 100% | 99,640 | Exact bit-for-bit match to `standard_dev.jsonl` |
| `results/standard_test.jsonl` | Standard RAG | test | 400 | 100% | 398,560 | Verified complete |
| `results/ircot_debug.jsonl` | IRCoT | debug | 50 | 100% | 76,430 | Verified complete |
| `results/ircot_dev.jsonl` | IRCoT | dev | 100 | 100% | 152,860 | Verified complete |
| `results/ircot_test.jsonl` | IRCoT | test | 400 | 100% | 611,440 | Verified complete |
| `results/erag_observe_debug.jsonl` | E-RAG Observe | debug | 50 | 100% | 215,300 | Gate G1 Verified (100% match to Standard) |
| `results/erag_observe_dev.jsonl` | E-RAG Observe | dev | 100 | 100% | 430,600 | Gate G1 Verified (100% match to Standard) |
| `results/erag_observe_test.jsonl` | E-RAG Observe | test | 400 | 100% | 1,722,400 | Gate G1 Verified (100% match to Standard) |
| `results/erag_active_debug.jsonl` | E-RAG Active | debug | 50 | 100% | 238,900 | Gate G2 Verified (Recovery from collapse) |
| `results/erag_active_dev.jsonl` | E-RAG Active | dev | 100 | 100% | 477,800 | Dev baseline (frozen config) |
| `results/erag_active_dev_pooled.jsonl` | E-RAG Active (Pooled) | dev | 100 | 100% | 481,200 | Pooled corpus comparison |
| `results/erag_active_test.jsonl` | E-RAG Active | test | 400 | 100% | 1,911,200 | Authoritative primary benchmark ($N=400$) |
| `results/erag_ablate_no_nli_dev.jsonl` | E-RAG w/o NLI | dev | 100 | 100% | 412,100 | Ablation 1 verified |
| `results/erag_ablate_no_conf_dev.jsonl` | E-RAG w/o Confidence | dev | 100 | 100% | 389,400 | Ablation 2 verified |
| `results/erag_ablate_no_prop_dev.jsonl` | E-RAG w/o Propagation | dev | 100 | 100% | 465,300 | Ablation 3 verified |
| `results/erag_ablate_no_adaptive_dev.jsonl` | E-RAG w/o Adaptive | dev | 100 | 100% | 312,800 | Ablation 4 verified |
| `results/erag_ablate_no_fast_dev.jsonl` | E-RAG w/o Fast Path | dev | 100 | 100% | 512,600 | Ablation 5 verified |
| `results/erag_ablate_no_eig_dev.jsonl` | E-RAG w/o EIG Query | dev | 100 | 100% | 443,700 | Ablation 6 verified |
| `results/stream_nomem.jsonl` | Episodic Stream Baseline | stream | 30 | 100% | 51,200 | Memory off baseline |
| `results/stream_mem.jsonl` | Episodic Stream Active | stream | 30 | 100% | 54,800 | Memory on (2/30 reuses) |
| `results/stream_planted.jsonl` | Contradiction Stream | stream | 30 | 100% | 53,100 | Contradiction override test |
| `results/erag_vs_ircot_study/erag_no_abs_test.jsonl` | E-RAG Uninhibited ($\theta=0$) | test | 400 | 100% | 1,508,646 | False abstention recovery run |

---

## 3. Persistent Caches & Checkpoint Databases

- **SQLite LLM Response Cache (`results/cache.sqlite`):**
  - Table: `c (k TEXT PRIMARY KEY, v TEXT)`
  - Total Unique Cached Records: **11,766 entries**
  - Cache Namespaces Present: `dev_v1`, `test_v1`, `debug_v1`, `mem_v1`
  - Cache Key: SHA-256 hash over `{ns, model, messages, temperature, max_tokens, reasoning_effort, sample_idx}`
- **NLI Inference Memoization Store (`results/nli_memo.pkl`):**
  - Serialized dictionary mapping MD5 hashes of `(claim || passage_windows)` to `(max_ent, max_con)`.
  - Enables instant warm verification of unchanged claim-evidence pairs.
- **MLflow Experiment Tracking Database (`mlflow.db`):**
  - File size: **1.55 MB**
  - Tracks experiment runs, flattened hyperparameter configurations, Git commits, and scalar metric trajectories.

