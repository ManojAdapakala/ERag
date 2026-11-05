"""Verified-Belief Episodic Memory module with verification gating, surprise triggers, and override deletion."""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class BeliefMemory:
    """Episodic belief store maintaining verified, surprising beliefs with contradiction override and decay."""

    def __init__(self, embedder: Any, cfg: Dict[str, Any]):
        self.emb = embedder
        self.m = cfg.get("memory", {})
        self.items: List[Dict[str, Any]] = []
        self.t: int = 0  # Question counter across stream

    def insert(self, text: str, conf: float = 0.9):
        """Plant a synthetic belief (used for wrong-belief override test E5)."""
        vec = self.emb.encode([text])[0]
        self.items.append({
            "text": text,
            "vec": vec,
            "conf": float(conf),
            "created": self.t,
            "last_used": self.t,
        })

    def read(
        self,
        question: str,
        fresh_pool: List[Dict[str, Any]],
        nli: Any,
        cfg: Dict[str, Any],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Query memory for relevant beliefs; verify against fresh evidence and delete contradicted beliefs.
        
        Returns:
            (memory_passages, mem_events)
        """
        self.t += 1
        if not self.items:
            return [], []

        qv = self.emb.encode([question])[0]
        sims = np.stack([it["vec"] for it in self.items]) @ qv
        passages: List[Dict[str, Any]] = []
        events: List[Dict[str, Any]] = []
        drop_indices: List[int] = []

        # Inspect top-3 most similar memory beliefs
        top_k = min(3, len(self.items))
        order = np.argsort(-sims)[:top_k]

        theta_mem = self.m.get("theta_mem", 0.75)
        w_mem = self.m.get("w_mem", 0.5)
        theta_s = cfg.get("nli", {}).get("theta_s", 0.5)
        theta_c = cfg.get("nli", {}).get("theta_c", 0.5)

        for i in order:
            if sims[i] < theta_mem:
                continue
            it = self.items[i]
            ev = nli.verify(it["text"], fresh_pool, theta_s=theta_s, theta_c=theta_c)
            
            # Contradiction-driven deletion (override)
            if ev["status"] == "contradicted":
                events.append({
                    "event": "overridden",
                    "belief": it["text"],
                    "by": [t for t, _ in ev.get("con", [])],
                })
                drop_indices.append(int(i))
                continue

            it["last_used"] = self.t
            passages.append({
                "title": "memory",
                "sentences": [it["text"]],
                "text": it["text"],
                "weight": float(w_mem * it["conf"]),
                "is_memory": True,
            })

        # Remove contradicted beliefs in reverse index order
        for idx in sorted(drop_indices, reverse=True):
            del self.items[idx]

        return passages, events

    def write(self, graph: Any, c0: float, seen_contra: bool, cfg: Dict[str, Any]):
        """Write supported, surprising beliefs; apply decay to unused beliefs; evict LRU over capacity."""
        m = self.m
        theta_surprise = m.get("theta_surprise", 0.6)
        theta_write = m.get("theta_write", 0.7)
        decay = m.get("decay", 0.98)
        capacity = m.get("capacity", 200)

        surprising = (c0 < theta_surprise) or seen_contra
        if surprising:
            for c in graph.claims:
                if c.status == "supported" and c.eff >= theta_write:
                    v = self.emb.encode([c.text])[0]
                    # Check for duplicate belief in memory
                    dup = next((it for it in self.items if float(it["vec"] @ v) > 0.95), None)
                    if dup:
                        dup["conf"] = max(dup["conf"], float(c.eff))
                        dup["last_used"] = self.t
                    else:
                        self.items.append({
                            "text": c.text,
                            "vec": v,
                            "conf": float(c.eff),
                            "created": self.t,
                            "last_used": self.t,
                        })

        # Memory decay: decay confidence of beliefs not accessed this step
        for it in self.items:
            if it["last_used"] < self.t:
                it["conf"] *= decay

        # Prune beliefs with confidence below retention floor (0.2)
        self.items = [it for it in self.items if it["conf"] >= 0.2]

        # Fixed capacity LRU eviction
        if len(self.items) > capacity:
            self.items.sort(key=lambda it: it["last_used"], reverse=True)
            self.items = self.items[:capacity]

