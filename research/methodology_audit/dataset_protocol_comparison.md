# Dataset and Experimental Protocol Audit: Original IRCoT vs E-RAG Test Bed

**Audit Scope:** Data provenance, corpus scale, sampling methodology, and evaluation protocols across Trivedi et al. (ACL 2023) and the E-RAG benchmark.

---

## 1. Property-by-Property Comparison Matrix

| Property | Original IRCoT (Trivedi et al. 2023) | E-RAG Project Test Bed | Same? | Consequence & Impact on Scientific Interpretation |
|---|---|---|---|---|
| **1. Dataset Family** | HotpotQA, 2WikiMultiHopQA, MuSiQue, IIRC. | HotpotQA (Yang et al., EMNLP 2018). | **Same** (for HotpotQA) | Both focus on multi-hop question answering requiring reasoning over 2+ supporting facts. |
| **2. Source Version & Setting** | HotpotQA Fullwiki (Open-Domain) setting. | HotpotQA Distractor setting (`hotpot_dev_distractor_v1.json`). | **Different** | Original paper searches full Wikipedia; our project searches candidate pools containing 2 gold + 8 adversarial distractors. |
| **3. Retrieval Corpus** | Oct 2017 Wikipedia dump: ~5,233,329 passages indexed in Elasticsearch. | Per-question candidate distractor context (10 passages per question) or Pooled setting (~1,000 passages on Dev, ~4,000 on Test). | **Different** | Absolute retrieval recall is substantially higher in our distractor setting ($74\%$ to $92\%$) than in 5M open-domain ($40\%$ to $65\%$). Cross-paper scores cannot be compared numerically. |
| **4. Raw Source Files** | HotpotQA official dev set (`hotpot_dev_fullwiki_v1.json`, 7,405 questions). | HotpotQA official dev set (`hotpot_dev_distractor_v1.json`, 7,405 questions, SHA-256: `e3da074df24...`). | **Different file / Same split** | Both sample questions from the official HotpotQA development split. However, the files provide different context metadata (distractor vs fullwiki). |
| **5. Original Sampling Strategy** | Random sample of 100 dev questions for hyperparameter tuning; random sample of 500 dev questions for evaluation. | Stratified random sample partitioned into three mutually disjoint splits: Debug-50 ($N=50$), Dev-100 ($N=100$), and Test-400 ($N=400$). | **Different** | Original used 100/500 random splits; ours uses 50/100/400 stratified splits partitioned across question type (`bridge` vs `comparison`) and difficulty (`hard`). |
| **6. Question ID Overlap** | Random seed in `StonyBrookNLP/ircot` sampled 500 questions out of 7,405. | Deterministic seed (`seed=13`) in `scripts/make_manifests.py` generated frozen manifests in `data/manifests/`. | **Partial overlap** | Because both independently sampled from the 7,405 dev questions, there is an expected statistical intersection of $\approx 500 \times \frac{400}{7405} \approx 27$ common questions, but they are not the same evaluation subset. |
| **7. Source Split Provenance** | Development split of HotpotQA. | Development split of HotpotQA (`hotpot_dev_distractor_v1.json`). | **Same** | All questions originate from the canonical HotpotQA dev split, meaning ground-truth answers and gold supporting facts share identical annotation quality. |
| **8. Supporting Paragraphs** | Gold supporting facts annotated as `(title, sentence_index)` pairs in HotpotQA. | Gold supporting facts annotated as `(title, sentence_index)` pairs in HotpotQA. | **Same** | Ground-truth gold titles and sentences are identical between the datasets. |
| **9. Distractor Context** | Discarded in open-domain setting; all non-gold context is retrieved dynamically from 5.23M passages. | Preserved as 8 adversarial distractor paragraphs selected by HotpotQA authors based on TF-IDF similarity to gold facts. | **Different** | Our setting forces models to discriminate against topically deceptive distractors that share subject nouns and entities. |
| **10. Corpus Document Counts** | 5,233,329 Wikipedia passages. | 10 paragraphs per question in distractor mode; 1,000 paragraphs in Dev pooled mode; 4,000 paragraphs in Test pooled mode. | **Different** | Smaller, denser search space in our project, allowing exhaustive evaluation of epistemic calibration and graph propagation. |
| **11. Document Representation** | Title prepended to text: `"{title}\n{text}"`. | Title prepended to text: `"{title}. " + " ".join(sentences)`. | **Nearly Identical** | Both format passages with explicit title headers to allow entity grounding and citation. |
| **12. Gold Evidence Format** | Set of unique gold titles: `{title for title, sent_idx in item["supporting_facts"]}`. | Set of unique gold titles: `{title for title, sent_idx in item["supporting_facts"]}`. | **Same** | Identical set representation in `src/eval/metrics.py:retrieval_recall`. |
| **13. Retrieval Top-$k$ & Evaluation** | Evaluated top-$k$ at $k \in \{2, 5, 10, 15\}$ paragraphs across iterative hops. | Fixed initial $k=3$, adaptive hop $k=3$, capped at `pool_cap=12`. | **Partial Match** | Comparable pool sizes ($12$ vs $15$), but step retrieval sizes differ slightly ($k=3$ vs $k=2$ or $k=6$). |
| **14. Answer Normalization** | Official HotpotQA script: lowercase, remove punctuation, remove articles (`a`, `an`, `the`), whitespace normalize. | `src/eval/metrics.py:normalize_answer`: lowercase, remove punctuation, remove articles, whitespace normalize. | **Same** | 100% bit-for-bit mathematical and algorithmic match. |
| **15. QA Evaluation Metrics** | Exact Match (EM) and token-level macro F1 score. | Exact Match (EM) and token-level macro F1 score, supplemented by Selective EM, Coverage, ECE, and Brier score. | **Superset** | Standard EM and F1 are identical; E-RAG adds epistemic calibration and selective risk metrics. |

---

## 2. Direct Answers to Core Audit Questions

### 1. Same dataset family?
**YES.** Both projects evaluate on **HotpotQA** (Yang et al., EMNLP 2018), a multi-hop question answering benchmark requiring information synthesis across multiple Wikipedia documents.

### 2. Same source split?
**YES.** Both projects draw their evaluation samples from the official **HotpotQA Development Split** (7,405 instances). Neither project evaluates on the hidden test set via the private evaluation server.

### 3. Same question instances?
**NO.** The original IRCoT paper evaluated on an unstratified random sample of 500 dev questions. Our project evaluates on a frozen, stratified sample of 400 dev questions (`data/manifests/test.jsonl`, seed 13, balanced across `bridge` and `comparison` question types, hard difficulty).

### 4. Same retrieval corpus?
**NO.** The original paper used the full **open-domain Wikipedia dump** (5,233,329 passages) indexed in Elasticsearch. Our project uses the official **HotpotQA distractor corpus** (10 candidate paragraphs per question) and an aggregated pooled corpus (1,000 paragraphs on Dev, 4,000 on Test).

### 5. Same retrieval setting?
**NO.** Original IRCoT evaluates in the **Open-Domain QA (ODQA) setting**, where retrieval must isolate relevant evidence from millions of irrelevant web articles. Our project evaluates in the **Distractor Setting**, where retrieval must isolate relevant evidence from a localized pool containing adversarial distractors designed specifically to fool multi-hop models.

### 6. Same evaluation protocol?
**YES.** For QA accuracy (Exact Match and Token F1) and Gold Paragraph Recall, both projects use the identical official HotpotQA evaluation formulas, text normalization rules, and set-intersection logic.

---

## 3. Scientific Implications and Cross-Paper Comparability

> [!WARNING]
> **Cross-Paper Numerical Comparison is Scientifically Invalid:**
> Due to the difference in retrieval corpora ($5.23\text{M}$ open-domain passages vs $10$ distractor passages per question), absolute scores from Trivedi et al. (e.g., IRCoT Codex EM $\approx 0.32$ in fullwiki) cannot be directly compared against our reported IRCoT EM of $0.4250$.
>
> In an open-domain setting, the retriever faces a massive needle-in-a-haystack challenge. In our distractor setting, the challenge is semantic discrimination against adversarial confounders that share names, demonyms, and keywords.
>
> **Within-Test-Bed Controlled Validity:**
> Within our test bed, **all five systems (Closed-Book, Standard RAG, IRCoT, Observational E-RAG, Active E-RAG) are evaluated under identical conditions**:
> 1. Same 400 test questions in identical order.
> 2. Same candidate passage pools.
> 3. Same embedding backbone (`all-MiniLM-L6-v2`) and FAISS index.
> 4. Same LLM backbone (`openai/gpt-oss-20b` via Groq).
> 5. Same evaluation script and normalization rules.
>
> Therefore, all relative comparisons ($\Delta\text{EM}$, $\Delta\text{F1}$, $\Delta\text{Recall}$, McNemar exact tests, and paired bootstrap CIs) are methodologically sound, fair, and statistically valid.

