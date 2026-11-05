"""Natural Language Inference (NLI) evidence verifier using Cross-Encoder models."""

import hashlib
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class NLI:
    """NLI verifier computing entailment and contradiction probabilities with sliding windows and dynamic label mapping."""

    def __init__(
        self,
        name: str = "cross-encoder/nli-deberta-v3-base",
        window: int = 2,
        batch: int = 16,
        device: Optional[str] = None,
    ):
        self.name = name
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForSequenceClassification.from_pretrained(name).eval()
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.dev)

        # Dynamic label mapping from model configuration (never hardcode label indices)
        labels = {int(i): str(l).lower() for i, l in self.model.config.id2label.items()}
        self.i_ent = next(i for i, l in labels.items() if "entail" in l)
        self.i_con = next(i for i, l in labels.items() if "contra" in l)
        self.window = window
        self.batch = batch
        self.memo: Dict[str, Tuple[float, float]] = {}

    @torch.no_grad()
    def _probs(self, premises: List[str], hypothesis: str) -> np.ndarray:
        """Batch-score premise-hypothesis pairs through the NLI cross-encoder."""
        out = []
        for s in range(0, len(premises), self.batch):
            chunk = premises[s : s + self.batch]
            enc = self.tok(
                chunk,
                [hypothesis] * len(chunk),
                truncation=True,
                max_length=512,
                padding=True,
                return_tensors="pt",
            ).to(self.dev)
            logits = self.model(**enc).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            out.append(probs)
        return np.concatenate(out, axis=0) if out else np.zeros((0, 3))

    def _windows(self, p: Dict[str, Any]) -> List[str]:
        """Construct sliding sentence windows of size w with prepended paragraph title."""
        s = p.get("sentences", [])
        if not s and "text" in p:
            s = [p["text"]]
        w = self.window
        title = p.get("title", "")
        prefix = f"{title}: " if title else ""
        if len(s) <= w:
            return [prefix + " ".join(s)]
        return [prefix + " ".join(s[i : i + w]) for i in range(len(s) - w + 1)]

    def best(self, claim: str, passage: Dict[str, Any]) -> Tuple[float, float]:
        """Return maximum entailment and contradiction probability across passage sentence windows."""
        wins = self._windows(passage)
        key = hashlib.md5((claim + "||" + "||".join(wins)).encode("utf-8")).hexdigest()
        if key not in self.memo:
            P = self._probs(wins, claim)
            max_ent = float(P[:, self.i_ent].max()) if len(P) > 0 else 0.0
            max_con = float(P[:, self.i_con].max()) if len(P) > 0 else 0.0
            self.memo[key] = (max_ent, max_con)
        return self.memo[key]

    def verify(
        self,
        claim: str,
        passages: List[Dict[str, Any]],
        theta_s: float = 0.5,
        theta_c: float = 0.5,
    ) -> Dict[str, Any]:
        """Classify relationship between claim and evidence passages: supported, contradicted, or unresolved."""
        sup: List[Tuple[str, float]] = []
        con: List[Tuple[str, float]] = []

        for p in passages:
            e, c = self.best(claim, p)
            w = float(p.get("weight", 1.0))
            e, c = e * w, c * w
            title = p.get("title", "unknown")
            if e >= theta_s:
                sup.append((title, e))
            if c >= theta_c:
                con.append((title, c))

        be = max((x for _, x in sup), default=0.0)
        bc = max((x for _, x in con), default=0.0)

        if sup and con:
            status = "supported" if be > bc else "contradicted"
            conflict = True
        elif sup:
            status = "supported"
            conflict = False
        elif con:
            status = "contradicted"
            conflict = False
        else:
            status = "unresolved"
            conflict = False

        return {"status": status, "conflict": conflict, "sup": sup, "con": con}
