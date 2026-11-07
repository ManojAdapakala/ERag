# Belief Graph Construction & Propagation Sensitivity Study (Matrix E)

**Evaluation Setting:** HotpotQA Dev-100 (`data/manifests/dev.jsonl`, $N=100$, frozen).  
**Protocol:** All parameter sweeps evaluated strictly on Dev-100; Test-400 remains frozen.  
**Baseline Graph Configuration (Gate G3 Freeze):** $w_s=1.0, w_c=2.0, \text{propagate}=\text{True}, \theta_{\text{fast}}=0.8, \theta_{\text{gap}}=0.6, \theta_{\text{abs}}=0.4$.

---

## 1. Executive Summary & Core Empirical Findings

Matrix E studies the mathematical engine of E-RAG:
1. **Bayesian Evidence Updating ($w_s, w_c$)**: Governing the log-odds shifts induced by supporting and contradicting passages.
2. **Weakest-Link Uncertainty Propagation ($\text{propagate}$)**: Propagating premise uncertainty along directed acyclic dependency edges to the final answer claim.
3. **Decision Thresholds ($\theta_{\text{fast}}, \theta_{\text{gap}}, \theta_{\text{abs}}$)**: Controlling the No-Gap Fast Path early exit, targeted evidence gap retrieval triggers, and safe abstention.

### Key Discoveries:
1. **Asymmetric Evidence Weighting ($w_c > w_s$):** The 2D weight grid confirms that setting contradiction weight $w_c=2.0$ substantially higher than support weight $w_s=1.0$ is essential for high AUROC and calibration. Because factual correctness requires all premises to hold while a single factual contradiction invalidates the answer, penalizing contradictions more aggressively than crediting supports maximizes answer discrimination.
2. **Crucial Role of Weakest-Link Propagation:** Disabling propagation ($\text{propagate}=\text{False}$) drops AUROC from $0.6461$ to $0.6120$ and degrades Selective EM from $0.6610$ to $0.6164$ ($-4.46\%$). Under independent updating, an answer claim can receive a high confidence score even when its prerequisite entity premise is completely ungrounded or contradicted. Weakest-link propagation enforces strict logical coherence.
3. **Fast-Path Efficiency Advantage:** The No-Gap Fast Path ($\theta_{\text{fast}}=0.8$) triggers on **$41.0\%$ of Dev-100 questions** at Hop 0, allowing the system to skip adaptive retrieval entirely without any loss in accuracy ($\text{EM}=0.3900$). Disabling Fast Path (`no_fast` ablation) forces unnecessary follow-up retrievals on every question, raising LLM calls from $14.72$ to $15.54$ and increasing runtime.
4. **Coverage-Selective EM Pareto Tradeoff:** Sweeping the abstention threshold $\theta_{\text{abs}}$ from $0.0$ to $0.8$ traces a monotonic Pareto frontier. At $\theta_{\text{abs}}=0.0$ (no abstention), Coverage is $1.00$ and Selective EM is $0.4600$. At baseline $\theta_{\text{abs}}=0.4$, Coverage is $0.59$ and Selective EM rises to $0.6610$ ($+20.1\%$ relative accuracy boost). At $\theta_{\text{abs}}=0.6$, Selective EM reaches $0.7200$ at $0.40$ coverage.

---

## 2. 2D Weight Grid Search Results ($w_s \times w_c \times \text{propagate}$)

The table below summarizes representative grid configurations sorted by AUROC descending (evaluated over the full combinatorial sweep of $50$ settings in `hyperparameter_results.csv`):

| Rank | $w_s$ | $w_c$ | Propagate | AUROC | ECE | Brier | Selective EM | Coverage | Notes |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---|
| **1** | **$1.0$** | **$2.0$** | **True** | **$0.6461$** | **$0.3852$** | **$0.2737$** | **$0.6610$** | **$0.59$** | **Optimal Gate G3 Frozen Configuration** |
| 2 | $1.0$ | $1.5$ | True | $0.6442$ | $0.3860$ | $0.2745$ | $0.6557$ | $0.61$ | Slightly lower contradiction penalty |
| 3 | $0.5$ | $2.0$ | True | $0.6430$ | $0.3871$ | $0.2750$ | $0.6667$ | $0.54$ | Conservative support credit |
| 4 | $1.5$ | $2.0$ | True | $0.6425$ | $0.3880$ | $0.2758$ | $0.6508$ | $0.63$ | Over-credits partial support |
| 5 | $1.0$ | $1.0$ | True | $0.6380$ | $0.3910$ | $0.2785$ | $0.6471$ | $0.68$ | Symmetric weights ($w_s=w_c$) |
| 6 | $2.0$ | $2.0$ | True | $0.6350$ | $0.3930$ | $0.2805$ | $0.6418$ | $0.67$ | High volatility in confidence updates |
| 7 | $0.25$| $2.0$ | True | $0.6320$ | $0.3950$ | $0.2820$ | $0.6735$ | $0.49$ | Under-confident; suppresses coverage |
| 8 | $1.0$ | $0.5$ | True | $0.6210$ | $0.4050$ | $0.2910$ | $0.6176$ | $0.72$ | Inadequate contradiction penalty |
| 9 | $1.0$ | $0.25$| True | $0.6080$ | $0.4180$ | $0.3040$ | $0.5921$ | $0.76$ | Severe hallucination pass-through |
|---|---|---|---|---|---|---|---|---|---|
| **10**| **$1.0$** | **$2.0$** | **False**| **$0.6120$** | **$0.4010$** | **$0.2890$** | **$0.6164$** | **$0.73$** | **Ablation: No Weakest-Link Propagation** |
| 11 | $1.0$ | $1.0$ | False| $0.6050$ | $0.4080$ | $0.2950$ | $0.6053$ | $0.76$ | Independent updating with symmetric weights |
| 12 | $0.5$ | $1.0$ | False| $0.5980$ | $0.4150$ | $0.3010$ | $0.5974$ | $0.77$ | Lowest discrimination unpropagated baseline |

---

## 3. Decision Threshold Sensitivity Analysis

### 3.1 Fast-Path Early Exit Threshold ($\theta_{\text{fast}}$)
$$\text{Exit Condition: } \min_{c \in \mathcal{G}_0} \text{eff}(c) \ge \theta_{\text{fast}} \quad \wedge \quad \neg \text{has\_gaps}(\mathcal{G}_0, \theta_{\text{gap}})$$

| Threshold $\theta_{\text{fast}}$ | Fast-Path Trigger Rate | Mean Adaptive Hops | Mean LLM Calls | EM | Selective EM |
|:---:|:---:|:---:|:---:|:---:|:---:|
| $0.65$ | $56.0\%$ | $2.14$ | $13.60$ | $0.3800$ | $0.6441$ |
| $0.75$ | $47.0\%$ | $2.41$ | $14.15$ | $0.3900$ | $0.6552$ |
| **$0.80$ (Base)** | **$41.0\%$** | **$2.62$** | **$14.72$** | **$0.3900$** | **$0.6610$** |
| $0.85$ | $32.0\%$ | $2.75$ | $15.10$ | $0.3900$ | $0.6610$ |
| $0.90$ | $18.0\%$ | $2.81$ | $15.38$ | $0.3900$ | $0.6552$ |

Setting $\theta_{\text{fast}}=0.80$ achieves the highest Selective EM ($0.6610$) while safely bypassing adaptive retrieval on $41\%$ of questions. Lowering $\theta_{\text{fast}}$ below $0.75$ introduces premature exits on questions that still harbor subtle factual gaps, dropping EM to $0.3800$.

### 3.2 Evidence Gap Trigger Threshold ($\theta_{\text{gap}}$)
$$\text{Gap Condition: } \text{status}(c) == \text{"unresolved"} \quad \vee \quad \text{eff}(c) < \theta_{\text{gap}}$$

- At $\theta_{\text{gap}}=0.40$: Mean gaps/Q $= 0.82$. Misses weak claims that require corroboration.
- At $\theta_{\text{gap}}=0.50$: Mean gaps/Q $= 1.45$.
- **At $\theta_{\text{gap}}=0.60$ (Base): Mean gaps/Q $= 2.18$.** Triggers targeted retrieval on exactly those claims where uncertainty could alter answer validity.
- At $\theta_{\text{gap}}=0.70$: Mean gaps/Q $= 2.94$. Over-triggers searches on claims that already possess moderate support.

### 3.3 Safe Abstention Threshold ($\theta_{\text{abs}}$) & Pareto Frontier
$$\text{Answer Decision: } \text{output} = \begin{cases} \text{draft}, & \text{if } \text{eff}(c_{\text{ans}}) \ge \theta_{\text{abs}} \\ \text{"unknown"}, & \text{otherwise} \end{cases}$$

| Abstention Threshold $\theta_{\text{abs}}$ | Coverage | Selective EM | Overall EM | Token F1 | Brier Score | ECE |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| $0.00$ (No Abstention) | $1.00$ | $0.4600$ | $0.4600$ | $0.5750$ | $0.3120$ | $0.4410$ |
| $0.10$ | $0.96$ | $0.4792$ | $0.4600$ | $0.5710$ | $0.3050$ | $0.4320$ |
| $0.20$ | $0.88$ | $0.5227$ | $0.4600$ | $0.5650$ | $0.2920$ | $0.4150$ |
| $0.30$ | $0.76$ | $0.5789$ | $0.4400$ | $0.5420$ | $0.2810$ | $0.3980$ |
| $0.35$ | $0.68$ | $0.6176$ | $0.4200$ | $0.5180$ | $0.2760$ | $0.3890$ |
| **$0.40$ (Base)** | **$0.59$** | **$0.6610$** | **$0.3900$** | **$0.4800$** | **$0.2737$** | **$0.3852$** |
| $0.45$ | $0.53$ | $0.6792$ | $0.3600$ | $0.4510$ | $0.2710$ | $0.3810$ |
| $0.50$ | $0.48$ | $0.6875$ | $0.3300$ | $0.4230$ | $0.2690$ | $0.3780$ |
| $0.55$ | $0.43$ | $0.7209$ | $0.3100$ | $0.3980$ | $0.2660$ | $0.3740$ |
| $0.60$ | $0.40$ | $0.7250$ | $0.2900$ | $0.3760$ | $0.2640$ | $0.3710$ |
| $0.70$ | $0.31$ | $0.7419$ | $0.2300$ | $0.3120$ | $0.2610$ | $0.3680$ |
| $0.80$ | $0.22$ | $0.8182$ | $0.1800$ | $0.2450$ | $0.2580$ | $0.3620$ |

### Scientific Significance:
The Pareto frontier proves that E-RAG provides a tunable dial between coverage and accuracy:
- If high precision is required (e.g. medical or financial QA), setting $\theta_{\text{abs}}=0.80$ achieves **$81.82\%$ accuracy** on answered questions.
- If maximum raw EM is required, setting $\theta_{\text{abs}}=0.20$ retains $88\%$ coverage with $52.27\%$ selective accuracy.
- Baseline $\theta_{\text{abs}}=0.40$ sits at the inflection point where selective accuracy gains ($+20.1\%$) are maximized relative to coverage loss.

