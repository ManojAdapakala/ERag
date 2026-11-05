# OpenSpec Proposal: System B (IRCoT-Inspired Retrieval Comparator)

**Component:** `src/systems/ircot.py`  
**Author:** Senior AI/ML Engineer & RAG Research Scientist  
**Date:** 2026-11-05  
**Status:** PROPOSED (Awaiting Human Approval Before Implementation)  
**Primary Reference:** *E-RAG Implementation Guide: End to End* (§10, pp. 26–28)  

---

## 1. RESEARCH MOTIVATION & OBJECTIVE

To scientifically evaluate the proposed **E-RAG** epistemic retrieval system (System C), we require an established multi-hop iterative retrieval comparator. We adopt an **IRCoT-inspired** (Interleaving Retrieval with Chain-of-Thought Reasoning, Trivedi et al., 2022) iterative baseline (System B).

The objective of System B is to test whether maintaining an explicit claim-level epistemic state (belief graph, NLI contradiction verification, EIG query utility) outperforms standard step-by-step reasoning-guided retrieval without epistemic modeling.

---

## 2. ARCHITECTURAL CONTROL FLOW

```
Question q
   │
   ▼
Initial Dense Retrieval (k=3) ──► Evidence Pool (up to pool_cap=12)
   │
   ▼
Iterative Reasoning Loop (Step 0 to max_steps - 1, default max_steps=4):
   │
   ├──► Generate Next Reasoning Sentence using current (q, Evidence Pool, Reasoning History)
   │
   ├──► Check Termination Condition:
   │       If sentence contains "so the answer is" (case-insensitive) or sentence is empty:
   │           BREAK LOOP
   │
   ├──► Use the generated reasoning sentence directly as the next search query:
   │       Retrieve top-k paragraphs
   │
   └──► Deduplicate against existing pool and expand evidence pool (capped at pool_cap)
   │
   ▼
Final Reader LLM Generation:
   Apply shared ANSWER_SYSTEM prompt over the accumulated Evidence Pool to produce the final answer.
```

---

## 3. PROMPT & INTERFACE SPECIFICATION

### 3.1 Step Reasoning Prompt
```python
STEP_SYSTEM = (
    "You are solving a multi-hop question step by step. You get the question, "
    "retrieved paragraphs and the reasoning so far. Write ONLY the next single reasoning sentence. "
    "If you can now answer, write a sentence that begins exactly with 'So the answer is:'."
)
```

User message template:
```
Question: {question}

Paragraphs:
{context_block(pool)}

Reasoning so far: {' '.join(steps) if steps else '(none)'}

Next sentence:
```

### 3.2 Reader Prompt
Identical to System A and System C (`src.llm.prompts.answer_messages`), ensuring the generator model and answer prompt are held strictly constant across all systems.

### 3.3 Function Signature & Return Schema
```python
def run_ircot(
    item: Dict[str, Any],
    retriever: Any,
    llm: Any,
    cfg: Dict[str, Any],
    tracker: Tracker,
) -> Dict[str, Any]:
    ...
    return {
        "answer": answer,              # Extracted final short answer
        "pool_titles": [p["title"] for p in pool],  # Final evidence pool titles
        "hops": len(steps),             # Number of reasoning/retrieval hops performed
        "steps": steps,                 # Full trace of intermediate reasoning sentences
    }
```

---

## 4. DOCUMENTED DIFFERENCES FROM ORIGINAL IRCOT

To ensure academic rigor, all differences between this implementation and the original Trivedi et al. (2022) study must be documented and reported in the paper:

| Aspect | Original IRCoT (Trivedi et al., 2022) | This Implementation (System B) | Justification / Control |
|:---|:---|:---|:---|
| **Retriever** | BM25 sparse lexical search over Wikipedia | Dense retrieval (`all-MiniLM-L6-v2` + FAISS `IndexFlatIP`) | Kept identical to System A and System C to isolate controller performance from retriever type |
| **Corpus Setting** | Open-domain Wikipedia dump | Distractor (10 paras/q) and Pooled corpus (~5,000 paras) | Directly tests multi-hop distraction resistance under controlled conditions |
| **Model** | OpenAI text-davinci-002 / GPT-3 / Flan-T5 | Groq `openai/gpt-oss-20b` | Held identical across all compared systems |
| **Prompting** | 8-shot in-context chain-of-thought demonstrations | Zero-shot structured step instruction | Eliminates prompt-engineering bias from few-shot exemplar selection |
| **Reader** | Dedicated reader prompt with CoT prefix | Shared zero-shot short-answer reader prompt | Guarantees exact comparability with Standard RAG and E-RAG outputs |
| **Limits** | Model-specific token limit and custom stopping heuristics | Explicit `max_steps=4` and `pool_cap=12` | Controlled budget matches E-RAG retrieval constraints |

---

## 5. EXPERIMENTAL FAIRNESS & VALIDATION CRITERIA

1. **Retrieval Budget Matching:**
   - System B with `max_steps = s` performs at most $1 + s$ retrieval queries.
   - System C with `max_hops = h` performs at most $1 + h$ retrieval queries.
   - Evaluated across sweeps: $s \in \{1, 2, 3, 4\}$ vs $h \in \{1, 2, 3\}$.
2. **Acceptance Tests:**
   - Executes successfully on `debug-50` with 0 failures.
   - Average hops $> 1.0$ (proves iterative expansion triggers on multi-hop questions).
   - Intermediate reasoning sentences (`steps`) and final evidence pools (`pool_titles`) are recorded in raw predictions.

