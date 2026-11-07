# E-RAG Component Inventory & Mathematical Specification

**Audit Target:** Full inventory of epistemic components, mathematical formulations, configuration constraints, and frozen baseline parameters.

---

## 1. Mathematical Formulations Implemented in Source Code

### Component D1: Self-Consistency Confidence Estimation (`src/epistemic/confidence.py`)
Samples $k$ independent answers at temperature $T > 0$:
$$\mathcal{S} = \{ \hat{y}^{(1)}, \hat{y}^{(2)}, \dots, \hat{y}^{(k)} \}$$
Normalized completions: $\bar{y}^{(i)} = \text{normalize}(\hat{y}^{(i)})$.
The system implements two distinct confidence definitions:
1. **Majority Consensus (`majority`):**
   $$c_0 = \frac{\max_{y} \sum_{i=1}^k \mathbb{I}(\bar{y}^{(i)} = y)}{k}$$
2. **Draft Agreement (`draft_agreement`):**
   $$c_0 = \frac{\sum_{i=1}^k \mathbb{I}(\bar{y}^{(i)} = \text{normalize}(\hat{y}_{\text{draft}}))}{k}$$
*Frozen Configuration:* `sc_samples = 5`, `sc_temperature = 0.7`, `confidence_def = majority`.

---

### Component D2: Claim Extraction & Dependency Parsing (`src/epistemic/claims.py`)
Prompts the LLM with structured JSON schema (`CLAIM_SYSTEM`) to decompose draft answer $\hat{y}$ into atomic factual propositions:
$$\mathcal{C} = \{ c_1, c_2, \dots, c_m \}, \quad m \le n_{\text{max}}$$
Each claim $c_j$ includes:
- `text`: Self-contained natural language sentence naming core entities.
- `answer_bearing`: Boolean indicator $\mathbb{I}(c_j \text{ states final answer})$.
- `depends_on`: List of indices $\{ d \in \mathbb{N} \mid 0 \le d < j \}$. Strict backward indexing guarantees that the resulting directed dependency graph is acyclic (DAG).
- `fallback`: Activated if JSON parsing fails or regex extraction finds no claims, yielding the single proposition *"The answer to [question] is [answer]"*.

*Frozen Configuration:* `max_claims = 5`.

---

### Component D3: Natural Language Inference Verification (`src/epistemic/nli.py`)
Uses `cross-encoder/nli-deberta-v3-base`. Evaluates each claim $c_j$ against candidate passages $P \in \mathcal{P}$ using sliding sentence windows of size $w=2$ with prepended title:
$$W_P = \{ \text{prefix} + \text{sent}_{t:t+w} \}$$
Probabilities:
$$P(\text{entailment}), P(\text{contradiction}) = \text{softmax}(\text{logits})$$
Maximum probabilities per passage:
$$e(c_j, P) = \max_{win \in W_P} P(\text{entailment} \mid win, c_j)$$
$$c(c_j, P) = \max_{win \in W_P} P(\text{contradiction} \mid win, c_j)$$

#### Subject-Clause Entity Grounding:
To eliminate spurious distractor contradictions (which caused the Phase 2 performance drop), a passage $P$ is eligible for contradiction verification **only if** its title or text mentions an extracted subject entity:
$$\text{eligible}(P, c_j) = \mathbb{I}(\exists \text{ entity } e \in \text{Entities}(c_j) \text{ s.t. } e \in P)$$
Classification rules:
$$\text{sup}(c_j) = \{ (P, e) \mid e(c_j, P) \ge \theta_s \}$$
$$\text{con}(c_j) = \{ (P, c) \mid c(c_j, P) \ge \theta_c \land \text{eligible}(P, c_j) \}$$
Status:
$$\text{status}(c_j) = \begin{cases} 
\text{"supported"} & \text{if } \sup \ne \emptyset \land (\con = \emptyset \lor \max e \ge \max c - 0.05) \\
\text{"contradicted"} & \text{if } \con \ne \emptyset \land (\sup = \emptyset \lor \max c > \max e + 0.05) \\
\text{"unresolved"} & \text{otherwise}
\end{cases}$$

*Frozen Configuration:* $\theta_s = 0.5$, $\theta_c = 0.5$, $\text{window} = 2$.

---

### Component E: Belief Graph Construction & Weakest-Link Propagation (`src/epistemic/belief_graph.py`)

#### 1. Evidence Update in Logit Space:
Bounded logit transformation:
$$\text{logit}(p, \epsilon=0.05) = \ln \left( \frac{\text{clip}(p, \epsilon, 1-\epsilon)}{1 - \text{clip}(p, \epsilon, 1-\epsilon)} \right)$$
Evidence accumulation for claim $c_j$:
$$z_j = \text{logit}(c_0) + w_s \sum_{(P, e) \in \text{sup}(c_j)} e - w_c \sum_{(P, c) \in \text{con}(c_j)} c$$
Clipping: $z_j^{\text{clipped}} = \text{clip}(z_j, -6.0, +6.0)$.
Posterior belief:
$$\text{conf}(c_j) = \sigma(z_j^{\text{clipped}}) = \frac{1}{1 + e^{-z_j^{\text{clipped}}}}$$

#### 2. Weakest-Link Uncertainty Propagation:
Evaluated in topological order (earlier claims first):
$$\text{eff}(c_j) = \begin{cases} 
\min \left( \text{conf}(c_j), \min_{d \in \text{depends\_on}(c_j)} \text{eff}(c_d) \right) & \text{if propagate is True and } \text{depends\_on}(c_j) \ne \emptyset \\
\text{conf}(c_j) & \text{otherwise}
\end{cases}$$

#### 3. Evidence-Gap Detection:
$$\text{gaps} = \{ c_j \in \mathcal{C} \mid (\text{use\_status} \land \text{status}(c_j) \ne \text{"supported"}) \lor \text{eff}(c_j) < \theta_{\text{gap}} \}$$

#### 4. Fast-Path Termination:
If $\text{gaps} = \emptyset$ and $\min_{c \in \mathcal{C}} \text{eff}(c) \ge \theta_{\text{fast}}$, the system exits without additional retrieval hops.

#### 5. Safe Abstention Rule:
Let $c_{\text{ans}}$ be the answer-bearing claim node. If $\text{eff}(c_{\text{ans}}) < \theta_{\text{abs}}$, the system abstains:
$$\text{final\_answer} = \begin{cases} \text{"unknown"} & \text{if Active mode and } \text{eff}(c_{\text{ans}}) < \theta_{\text{abs}} \\ \hat{y}_{\text{draft}} & \text{otherwise} \end{cases}$$

*Frozen Configuration (`configs/final.yaml`):*
- $w_s = 1.0$
- $w_c = 1.0$ (calibrated from $2.0$ via Dev grid search `scripts/weight_grid.py`)
- $\text{propagate} = \text{True}$
- $\theta_{\text{fast}} = 0.8$
- $\theta_{\text{gap}} = 0.6$
- $\theta_{\text{abs}} = 0.4$

---

### Component F: Retrieval & EIG-Inspired Query Selection (`src/systems/erag.py`)
Target selection:
$$c^* = \arg\min_{c \in \text{gaps}} \text{eff}(c)$$
Query proposal: Prompts LLM to generate $n=3$ candidate search queries targeting $c^*$.
Novelty scoring against current evidence pool $\mathcal{P}_{\text{curr}}$:
$$\text{novelty}(q) = \frac{\sum_{P \in \text{retrieve}(q, k)} \mathbb{I}(P \notin \mathcal{P}_{\text{curr}})}{\max(|\text{retrieve}(q, k)|, 1)}$$
Expected Information Gain objective:
$$\text{Score}(q) = (1.0 - \text{eff}(c^*)) \times \text{novelty}(q)$$
Best query $q^* = \arg\max_{q} \text{Score}(q)$ is executed, and non-duplicate passages are appended to the pool:
$$\mathcal{P}_{\text{new}} = (\mathcal{P}_{\text{curr}} \cup \text{retrieve}(q^*, k))_{[:\text{pool\_cap}]}$$

*Frozen Configuration:* $k = 3$, $\text{max\_hops} = 2$, $\text{pool\_cap} = 12$, candidate queries $= 3$.

---

### Component G: Verified-Belief Episodic Memory (`src/systems/memory.py`)
1. **Verification-Gated Writing:** Only beliefs with $\text{status}(c) = \text{"supported"}$ and $\text{eff}(c) \ge \theta_{\text{write}} = 0.7$ can be stored.
2. **Surprise Gating:** Memory writes trigger only when initial draft was uncertain ($c_0 < \theta_{\text{surprise}} = 0.6$) or contradicted evidence was resolved.
3. **Contradiction-Driven Deletion:** When memory beliefs are queried for fresh questions, NLI verifies them against the new retrieval pool. If $\text{status} = \text{"contradicted"}$, the memory belief is immediately deleted.
4. **Confidence Decay:** Multiplicative decay of $0.98$ per question step applied to unaccessed memory beliefs. Beliefs below $0.20$ confidence are purged.
5. **Fixed Capacity:** LRU eviction capped at `capacity = 200`.

*Frozen Configuration:* $\theta_{\text{write}} = 0.7$, $\theta_{\text{surprise}} = 0.6$, $\text{decay} = 0.98$, $\text{capacity} = 200$, $\theta_{\text{mem}} = 0.75$, $w_{\text{mem}} = 0.5$.

---

## 2. Complete Parameter Inventory

| Subsystem | Parameter | Type | Default (`base.yaml`) | Frozen (`final.yaml`) | Permissible Dev Sweep Range |
|---|---|---|---|---|---|
| **Model** | `name` | string | `openai/gpt-oss-20b` | `openai/gpt-oss-20b` | Fixed |
| **Model** | `provider` | string | `groq` | `groq` | Fixed |
| **Model** | `max_tokens` | int | 1024 | 1024 | Fixed |
| **Retrieval** | `embedder` | string | `all-MiniLM-L6-v2` | `all-MiniLM-L6-v2` | Fixed |
| **Retrieval** | `k` | int | 3 | 3 | $\{1, 3, 5, 7, 10\}$ |
| **Retrieval** | `setting` | string | `distractor` | `distractor` | `distractor` vs `pooled` |
| **IRCoT** | `max_steps` | int | 4 | 4 | $\{1, 2, 3, 4, 5, 6\}$ |
| **IRCoT** | `pool_cap` | int | 12 | 12 | $\{6, 9, 12, 15, 20\}$ |
| **Epistemic SC** | `sc_samples` | int | 5 | 5 | $\{1, 3, 5, 7, 10\}$ |
| **Epistemic SC** | `sc_temperature`| float | 0.7 | 0.7 | $\{0.0, 0.2, 0.5, 0.7, 1.0\}$ |
| **Epistemic SC** | `confidence_def`| string | `majority` | `majority` | `majority` vs `draft_agreement` |
| **Claims** | `max_claims` | int | 5 | 5 | $\{1, 3, 5, 7, 10\}$ |
| **NLI** | `theta_s` | float | 0.5 | 0.5 | $\{0.3, 0.4, 0.5, 0.6, 0.7\}$ |
| **NLI** | `theta_c` | float | 0.5 | 0.5 | $\{0.3, 0.4, 0.5, 0.6, 0.7\}$ |
| **NLI** | `window` | int | 2 | 2 | $\{1, 2, 3\}$ |
| **Belief Graph**| `w_s` | float | 1.0 | 1.0 | $\{0.25, 0.5, 1.0, 1.5, 2.0\}$ |
| **Belief Graph**| `w_c` | float | 2.0 | 1.0 | $\{0.25, 0.5, 1.0, 1.5, 2.0\}$ |
| **Belief Graph**| `propagate` | bool | True | True | True vs False |
| **E-RAG Loop** | `theta_fast` | float | 0.8 | 0.8 | $\{0.65, 0.75, 0.8, 0.85, 0.9\}$ |
| **E-RAG Loop** | `theta_gap` | float | 0.6 | 0.6 | $\{0.4, 0.5, 0.6, 0.7\}$ |
| **E-RAG Loop** | `theta_abs` | float | 0.4 | 0.4 | $\{0.2, 0.3, 0.4, 0.5, 0.6\}$ |
| **E-RAG Loop** | `max_hops` | int | 2 | 2 | $\{0, 1, 2, 3, 4\}$ |
| **E-RAG Loop** | `pool_cap` | int | 12 | 12 | $\{6, 9, 12, 15, 20\}$ |
| **Memory** | `enabled` | bool | False | False | True vs False |
| **Memory** | `theta_write` | float | 0.7 | 0.7 | $\{0.6, 0.7, 0.8, 0.9\}$ |
| **Memory** | `theta_surprise`| float | 0.6 | 0.6 | $\{0.2, 0.3, 0.4, 0.5, 0.6\}$ |
| **Memory** | `decay` | float | 0.98 | 0.98 | $\{0.90, 0.95, 0.98, 0.99, 1.0\}$ |
| **Memory** | `capacity` | int | 200 | 200 | $\{50, 100, 200, 500\}$ |

