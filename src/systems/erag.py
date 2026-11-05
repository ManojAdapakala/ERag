"""System C: E-RAG adaptive epistemic retrieval system with belief graphs, EIG query selection, Fast Path, and safe abstention."""

from typing import Any, Dict, List, Optional
from src.epistemic.belief_graph import BeliefGraph
from src.epistemic.claims import extract_claims
from src.epistemic.confidence import self_consistency
from src.llm.client import Tracker
from src.llm.prompts import answer_messages

QUERY_SYSTEM = (
    "Write {n} short, different search queries that would find evidence confirming or "
    "refuting the claim. One query per line, no numbering."
)


def propose_queries(llm: Any, claim_text: str, tracker: Tracker, n: int = 3) -> List[str]:
    """Generate candidate follow-up search queries targeting a specific claim."""
    msgs = [
        {"role": "system", "content": QUERY_SYSTEM.format(n=n)},
        {"role": "user", "content": f"Claim: {claim_text}"},
    ]
    raw = llm.chat(msgs, tracker, max_tokens=200, temperature=0.0)
    lines = [
        l.strip(" -*0123456789.").strip()
        for l in raw.splitlines()
        if l.strip(" -*0123456789.").strip()
    ]
    return lines[:n]


def conf_value(sc: Dict[str, Any], cfg: Dict[str, Any]) -> float:
    """Extract selected confidence definition from self-consistency results."""
    conf_def = cfg.get("erag", {}).get("confidence_def", "majority")
    if conf_def == "majority":
        return float(sc.get("conf_majority", 1.0))
    return float(sc.get("conf_draft_agreement", 1.0))


def build_graph(
    claims: List[Dict[str, Any]],
    passages: List[Dict[str, Any]],
    nli: Any,
    c0: float,
    cfg: Dict[str, Any],
    ablate: set,
) -> BeliefGraph:
    """Build and update the epistemic belief graph from extracted claims and evidence passages."""
    e, n = cfg.get("erag", {}), cfg.get("nli", {})
    w_s = e.get("w_s", 1.0)
    w_c = e.get("w_c", 2.0)
    propagate = "no_prop" not in ablate

    g = BeliefGraph(w_s=w_s, w_c=w_c, propagate=propagate)
    for c in claims:
        if "no_nli" in ablate or nli is None:
            ev = {"status": "unresolved", "conflict": False, "sup": [], "con": []}
        else:
            theta_s = n.get("theta_s", 0.5)
            theta_c = n.get("theta_c", 0.5)
            ev = nli.verify(c["text"], passages, theta_s=theta_s, theta_c=theta_c)
        g.add(c["text"], c["answer_bearing"], c["depends_on"], c0, ev)
    return g.update()


def targeted_retrieve(
    llm: Any,
    retriever: Any,
    claim: Any,
    pool: List[Dict[str, Any]],
    k: int,
    ablate: set,
    tracker: Tracker,
) -> List[Dict[str, Any]]:
    """Select next evidence using EIG-inspired uncertainty reduction x pool novelty."""
    seen = {p["title"] for p in pool}
    if "no_eig" in ablate:
        queries = [claim.text]
    else:
        proposed = propose_queries(llm, claim.text, tracker, n=3)
        queries = proposed if proposed else [claim.text]

    best, best_score = [], -1.0
    for qy in queries:
        res = retriever.search(qy, k=k)
        tracker.probe_searches += 1
        novelty = sum(1 for r in res if r["title"] not in seen) / max(len(res), 1)
        score = (1.0 - claim.eff) * novelty
        if score > best_score:
            best, best_score = res, score

    tracker.retrievals += 1
    return [r for r in best if r["title"] not in seen]


def run_erag(
    item: Dict[str, Any],
    retriever: Any,
    llm: Any,
    nli: Any,
    cfg: Dict[str, Any],
    tracker: Tracker,
    memory: Any = None,
) -> Dict[str, Any]:
    """Execute complete 12-step E-RAG adaptive epistemic retrieval loop."""
    e = cfg.get("erag", {})
    k = cfg["retrieval"].get("k", 3)
    q = item["question"]
    ablate = set(e.get("ablate", []) or cfg.get("ablate", []))
    active = e.get("mode", "active") == "active"
    use_status = "no_nli" not in ablate

    # Step 1: Initial retrieval
    pool = retriever.search(q, k=k)
    tracker.retrievals += 1

    mem_passages, mem_events = [], []
    if memory is not None:
        mem_passages, mem_events = memory.read(q, pool, nli, cfg)

    hop, c0, trace, seen_contra = 0, None, [], False
    while True:
        # Step 2: Generation of draft answer
        draft = llm.chat(answer_messages(q, pool), tracker, temperature=0.0)

        # Step 3: Self-consistency confidence estimation
        if c0 is None:
            if "no_conf" in ablate:
                c0 = 1.0
            else:
                sc = self_consistency(
                    llm,
                    q,
                    pool,
                    draft,
                    tracker,
                    k=e.get("sc_samples", 5),
                    temperature=e.get("sc_temperature", 0.7),
                )
                c0 = conf_value(sc, cfg)

        # Step 4, 5, 6, 7: Claim decomposition, NLI verification, belief graph update & propagation
        claims = extract_claims(llm, q, draft, pool, tracker, max_claims=e.get("max_claims", 5))
        graph = build_graph(claims, pool + mem_passages, nli, c0, cfg, ablate)
        gaps = graph.gaps(e.get("theta_gap", 0.6), use_status=use_status)
        seen_contra |= any(c.status == "contradicted" for c in graph.claims)

        trace.append({
            "hop": hop,
            "answer": draft,
            "pool": [p["title"] for p in pool],
            "graph": graph.to_json(),
        })

        # Check stopping criteria: observational mode, no_adaptive ablation, or max_hops exhausted
        if not active or "no_adaptive" in ablate or hop >= e.get("max_hops", 2):
            break

        # Step 8 & 9: Gap selection and No-Gap Fast Path
        if "no_conf" in ablate:
            target = graph.answer_claim()
        elif gaps:
            target = graph.weakest(gaps)
        elif "no_fast" in ablate or graph.min_eff() < e.get("theta_fast", 0.8):
            target = graph.weakest(graph.claims)
        else:
            break  # Fast Path: all claims sufficiently supported

        # Step 10 & 11: EIG-inspired query selection and adaptive retrieval
        new = targeted_retrieve(llm, retriever, target, pool, k, ablate, tracker)
        if not new:
            break
        pool = (pool + new)[: e.get("pool_cap", 12)]
        hop += 1

    final_conf = c0
    if hop > 0:
        if "no_conf" in ablate:
            final_conf = 1.0
        else:
            sc = self_consistency(
                llm,
                q,
                pool,
                draft,
                tracker,
                k=e.get("sc_samples", 5),
                temperature=e.get("sc_temperature", 0.7),
            )
            final_conf = conf_value(sc, cfg)
        graph = build_graph(claims, pool + mem_passages, nli, final_conf, cfg, ablate)

    ans = graph.answer_claim()

    # Step 12: Safe abstention rule: withheld only if answer claim effective confidence is below threshold
    abstain = "no_conf" not in ablate and (ans.eff < e.get("theta_abs", 0.4))
    answer = "unknown" if (active and abstain) else draft

    if memory is not None and active:
        memory.write(graph, c0, seen_contra, cfg)

    return {
        "answer": answer,
        "draft": draft,
        "abstained": bool(active and abstain),
        "would_abstain": bool(abstain),
        "answer_conf": float(ans.eff),
        "c0": float(c0),
        "pool_titles": [p["title"] for p in pool],
        "hops": hop,
        "trace": trace,
        "mem_used": len(mem_passages),
        "mem_events": mem_events,
        "claim_fallbacks": sum(1 for c in claims if c.get("fallback", False)),
    }

