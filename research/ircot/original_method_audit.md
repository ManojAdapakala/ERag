# Original IRCoT Method & Official Implementation Audit

**Document ID:** `IRCOT-AUDIT-2026-001`  
**Target Literature:** *"Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step Questions"*, Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, Ashish Sabharwal (ACL 2023).  
**Official Repository:** `https://github.com/StonyBrookNLP/ircot`  
**Audit Date:** 2026-11-08  
**Auditor:** Senior AI Research Engineer, RAG Architect & Experimental Scientist  

---

## 1. Executive Summary of Trivedi et al. (ACL 2023)

IRCoT addresses the fundamental limitation of standard single-step Retrieval-Augmented Generation (RAG) when applied to multi-hop questions: namely, that a retriever conditioned only on the initial raw question $Q$ frequently fails to retrieve the "bridge" or second-hop documents because the entities connecting the hops are not mentioned in $Q$.

IRCoT proposes an interleaved architecture where:
1. Retrieval guides Chain-of-Thought (CoT) reasoning by providing retrieved passages to ground intermediate reasoning steps.
2. CoT reasoning guides retrieval by treating the newly generated reasoning sentence as a search query to fetch the next set of relevant documents.
3. This process alternates cyclically until a termination criterion is met (the model generates a final answer or reaches a step budget).

---

## 2. In-Depth Component Analysis: Official Paper vs Official Repository

### 2.1 Retrieval Engine & Algorithm
- **Paper Specification:** Trivedi et al. evaluate two retrieval backbones:
  1. **Sparse Lexical Retrieval:** BM25 via Elasticsearch server (primary setting in main results).
  2. **Dense Semantic Retrieval:** Contriever (Izacard et al., 2022) studied in the Appendix.
- **Repository Implementation (`retriever.py` in `StonyBrookNLP/ircot`):**
  - Connects to an external Elasticsearch 6.x server over HTTP (`elasticsearch` Python library).
  - Uses `BM25Similarity` with standard parameters ($k_1 = 1.2$, $b = 0.75$).
  - Queries are formed by stripping prefixes and executing multi-match queries over document `title` and `text`.

### 2.2 Retrieval Corpus & Document Representation
- **Paper Specification:** Open-domain Wikipedia corpus derived from the October 2017 Wikipedia dump (the standard HotpotQA fullwiki corpus containing $\approx 5.23$ million passages).
- **Document Chunking:** Each Wikipedia article is segmented into introductory paragraphs; each document entry consists of a title and paragraph text (`title \n body`).
- **Indexing:** Documents are assigned a unique doc ID; duplicate documents retrieved in subsequent hops are filtered out by ID.

### 2.3 Reasoning Prompts & Demonstration Exemplars
- **Paper Specification:** In-context few-shot prompting using hand-crafted multi-hop demonstration exemplars.
- **Exemplar Count:**
  - For HotpotQA: 6 to 8 static multi-hop demonstrations formatted with interleaved `Question`, `Paragraphs`, and `Thought` reasoning sentences.
  - Exemplars demonstrate the step-by-step resolution of multi-hop facts (e.g., identifying the spouse of an actor, then querying for the spouse's birth city).
- **Template Structure in Official Repository (`prompts/hotpotqa/`):**
  ```
  Question: {question}
  Paragraphs:
  {accumulated_paragraphs}
  Thought: {reasoning_so_far}
  ```

### 2.4 Generator Models & Decoding Configurations
- **Paper Specification:** Evaluated primarily on:
  1. `code-davinci-002` (OpenAI Codex, 175B parameters)
  2. `text-davinci-002` (OpenAI InstructGPT, 175B parameters)
  3. `flan-t5-xxl` (11B open-weights encoder-decoder)
- **Decoding Hyperparameters:** Greedy decoding ($\text{temperature} = 0.0$), `max_tokens` set per step (typically $32$ to $64$ tokens per reasoning sentence), with a stop token set to newline (`stop=["\n"]`).

### 2.5 Iterative Accumulation & Termination Mechanism
- **Step Generation:** The LLM generates the next single reasoning sentence.
- **Termination Conditions:**
  1. Generation of an answer sentence beginning with `"So the answer is:"`.
  2. Generation of an empty string or immediate newline.
  3. Reaching the maximum step budget ($M \in [4, 7]$ steps, default $M=4$ for HotpotQA).
- **Evidence Accumulation:** If the sentence is not terminal, it is passed directly as a search query to retrieve top-$k$ paragraphs ($k \in [2, 3]$). Newly discovered paragraphs are appended to the context pool up to a cumulative paragraph cap ($P \in [12, 15]$).

### 2.6 Downstream QA Modes
- **Mode 1 (`ircot`):** Direct extraction from the final CoT reasoning sentence (e.g., parsing the text after `"So the answer is:"`).
- **Mode 2 (`ircot_qa`):** A separate reader call that passes the question $Q$ and all accumulated paragraphs to an independent QA prompt to output a concise answer string. Trivedi et al. report that `ircot_qa` generally yields higher Exact Match scores.

---

## 3. Methodological Comparison with the E-RAG Test Bed

The central question for research reproducibility is: **How does our implementation relate to the original Trivedi et al. study?**

| Scientific Attribute | Trivedi et al. (ACL 2023) | Our Research Implementation (`src/systems/ircot.py`) | Methodological Parity Status |
| :--- | :--- | :--- | :---: |
| **Interleaving Logic** | Cyclic: CoT sentence $\to$ Query $\to$ Retrieval $\to$ Append | Cyclic: CoT sentence $\to$ Query $\to$ Retrieval $\to$ Append | **Exact Match (Tier 1 Core)** |
| **Stopping Condition** | `"So the answer is"` or max steps | `"so the answer is" in s.lower() or not s` | **Exact Match (Tier 1 Core)** |
| **Evidence Accumulation** | Persistent ordered pool with deduplication | Persistent ordered pool with title deduplication | **Exact Match (Tier 1 Core)** |
| **Context Budget** | $P \approx 12$ paragraphs | $pool\_cap = 12$ paragraphs | **Exact Match (Tier 1 Core)** |
| **Final Reader Call** | Separate reader conditioned on context (`ircot_qa`) | Separate reader conditioned on context (`answer_messages`) | **Exact Match (Tier 1 Core)** |
| **Retriever Algorithm** | BM25 (Elasticsearch) | Dense `all-MiniLM-L6-v2` + FAISS (and Phase 4 BM25) | **Controlled Adaptation** |
| **Corpus Scale** | Open-domain Wikipedia ($5.23\text{M}$ passages) | Standardized 10-paragraph distractor pool per question | **Controlled Adaptation** |
| **Prompt Engineering** | 6-shot to 8-shot hand-crafted exemplars | Zero-shot structured directive prompt | **Controlled Adaptation** |
| **Generator Backbone** | OpenAI Codex (`code-davinci-002`) | Groq API (`openai/gpt-oss-20b`) | **Controlled Adaptation** |

---

## 4. Key Takeaways for Publication

1. **Core Algorithmic Fidelity:** Our implementation in `src/systems/ircot.py` faithfully replicates the exact algorithmic mechanics of IRCoT (interleaving, reasoning-query generation, evidence deduplication, paragraph capping, and `ircot_qa` final reading).
2. **Experimental Fair Control:** Rather than testing IRCoT with few-shot Codex while testing E-RAG with zero-shot GPT-OSS-20b (which would conflate model capabilities with algorithmic merits), our testbed holds the **generator backbone, prompting style, corpus scale, and context budgets strictly identical**.
3. **Naming Requirement:** We formally designate this baseline as **"IRCoT-Controlled"** to distinguish it from open-domain Codex benchmarks while affirming its algorithmic fidelity.

