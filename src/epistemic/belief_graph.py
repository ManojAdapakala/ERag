"""Belief graph representation, evidence update, and weakest-link uncertainty propagation."""

from dataclasses import dataclass, field
import math
from typing import Any, Dict, List, Optional


def logit(p: float, eps: float = 0.05) -> float:
    """Bounded logit transformation: log(p / (1 - p)) clipped to [eps, 1 - eps]."""
    p_clipped = min(max(float(p), eps), 1.0 - eps)
    return math.log(p_clipped / (1.0 - p_clipped))


def sigmoid(z: float) -> float:
    """Numerically stable sigmoid function."""
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    else:
        exp_z = math.exp(z)
        return exp_z / (1.0 + exp_z)


@dataclass
class Claim:
    """Node in the epistemic belief graph representing an atomic factual claim."""
    idx: int
    text: str
    answer_bearing: bool
    depends_on: List[int]
    c0: float
    status: str
    sup: List[Any] = field(default_factory=list)
    con: List[Any] = field(default_factory=list)
    conflict: bool = False
    conf: float = 0.0  # Confidence after logit-space evidence update
    eff: float = 0.0   # Effective confidence after weakest-link propagation


class BeliefGraph:
    """Directed acyclic graph tracking atomic claims, dependencies, and evidence status."""

    def __init__(
        self,
        w_s: float = 1.0,
        w_c: float = 2.0,
        clip: float = 6.0,
        propagate: bool = True,
    ):
        self.w_s = float(w_s)
        self.w_c = float(w_c)
        self.clip = float(clip)
        self.propagate = propagate
        self.claims: List[Claim] = []

    def add(
        self,
        text: str,
        answer_bearing: bool,
        depends_on: List[int],
        c0: float,
        ev: Dict[str, Any],
    ) -> None:
        """Add a claim node to the graph. Only backwards edges (0 <= d < idx) are retained to guarantee acyclicity."""
        idx = len(self.claims)
        deps = [d for d in depends_on if isinstance(d, int) and 0 <= d < idx]
        self.claims.append(
            Claim(
                idx=idx,
                text=text,
                answer_bearing=answer_bearing,
                depends_on=deps,
                c0=c0,
                status=ev.get("status", "unresolved"),
                sup=list(ev.get("sup", [])),
                con=list(ev.get("con", [])),
                conflict=bool(ev.get("conflict", False)),
            )
        )

    def update(self) -> "BeliefGraph":
        """Compute logit-space belief update from evidence and propagate through dependencies."""
        # Step 1: Evidence update in logit space
        for c in self.claims:
            sup_sum = sum(p for _, p in c.sup)
            con_sum = sum(p for _, p in c.con)
            z = logit(c.c0) + self.w_s * sup_sum - self.w_c * con_sum
            z_clipped = max(-self.clip, min(self.clip, z))
            c.conf = sigmoid(z_clipped)

        # Step 2: Weakest-link propagation in topological order (earlier claims first)
        for c in self.claims:
            if self.propagate and c.depends_on:
                parents_eff = [self.claims[d].eff for d in c.depends_on]
                c.eff = min([c.conf] + parents_eff)
            else:
                c.eff = c.conf
        return self

    def answer_claims(self) -> List[Claim]:
        """Return all answer-bearing claim nodes (or final claim as fallback)."""
        ans_nodes = [c for c in self.claims if c.answer_bearing]
        return ans_nodes if ans_nodes else ([self.claims[-1]] if self.claims else [])

    def answer_critical_claims(self) -> List[Claim]:
        """Return all answer-bearing claims and their upstream dependency ancestors."""
        ans_nodes = self.answer_claims()
        critical_indices = set()
        queue = [c.idx for c in ans_nodes]
        while queue:
            idx = queue.pop(0)
            if idx not in critical_indices:
                critical_indices.add(idx)
                if 0 <= idx < len(self.claims):
                    queue.extend(self.claims[idx].depends_on)
        return [self.claims[i] for i in sorted(critical_indices)]

    def answer_claim(self) -> Claim:
        """Return the answer-bearing claim node with lowest effective confidence (or final claim as fallback)."""
        ans_nodes = self.answer_claims()
        return min(ans_nodes, key=lambda c: c.eff) if ans_nodes else self.claims[-1]

    def gaps(self, theta_gap: float, use_status: bool = True) -> List[Claim]:
        """Identify claims that have insufficient support or fall below theta_gap."""
        return [
            c for c in self.claims
            if (use_status and c.status != "supported") or c.eff < theta_gap
        ]

    def weakest(self, candidate_claims: List[Claim]) -> Claim:
        """Return the claim with the lowest effective confidence."""
        return min(candidate_claims, key=lambda c: c.eff)

    def min_eff(self) -> float:
        """Return minimum effective confidence across all claims in graph."""
        if not self.claims:
            return 0.0
        return min(c.eff for c in self.claims)

    def to_json(self) -> List[Dict[str, Any]]:
        """Serialize belief graph nodes into inspectable dictionaries."""
        return [vars(c).copy() for c in self.claims]

