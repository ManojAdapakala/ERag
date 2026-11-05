"""Natural Language Inference (NLI) evidence verifier using Cross-Encoder models."""

import hashlib
import pickle
import re
from pathlib import Path
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
        self.memo_path = Path("results/nli_memo.pkl")
        self.memo: Dict[str, Tuple[float, float]] = {}
        if self.memo_path.exists():
            try:
                with open(self.memo_path, "rb") as mf:
                    self.memo = pickle.load(mf)
            except Exception:
                pass
        self._dirty_count = 0

    def _save_memo(self):
        try:
            self.memo_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.memo_path, "wb") as mf:
                pickle.dump(self.memo, mf)
            self._dirty_count = 0
        except Exception:
            pass

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
            self._dirty_count += 1
            if self._dirty_count >= 20:
                self._save_memo()
        return self.memo[key]

    @staticmethod
    def _extract_subject_entities(claim: str) -> List[str]:
        """Extract core subject entities, proper noun phrases, and titles from the claim's subject clause."""
        verb_re = re.compile(
            r'\b(is|was|were|are|has|have|had|served|initiated|became|directed|starred|defeated|won|played|share|shares|co-wrote|managed|competed|lasted|born)\b',
            re.IGNORECASE,
        )
        m = verb_re.search(claim)
        subj_clause = claim[:m.start()].strip() if m else claim

        entities = []
        # 1. Quoted strings in subject clause
        quoted = re.findall(r'"([^"]+)"', subj_clause)
        entities.extend([q.strip() for q in quoted if len(q.strip()) >= 2])

        # 2. Capitalized phrases in subject clause
        cap_phrases = re.findall(r'\b(?:[A-Z0-9][a-zA-Z0-9/\'-]*(?:\s+[A-Z0-9][a-zA-Z0-9/\'-]*)*)\b', subj_clause)
        ignore = {
            "The", "Both", "Because", "What", "Are", "Were", "And", "Or", "In", "On", "At",
            "From", "To", "A", "An", "This", "That", "It", "Yes", "No", "When", "Where", "Who", "Why", "How",
            "American", "British", "English", "French", "German", "Canadian", "Italian", "Japanese", "Australian",
            "Russian", "Spanish", "Chinese", "South Korean", "Korean", "Mexican", "Indian"
        }
        for p in cap_phrases:
            words = [w for w in p.split() if w not in ignore]
            if words:
                cleaned = " ".join(words)
                if len(cleaned) >= 2 and cleaned not in entities:
                    entities.append(cleaned)

        # If no entities found in subject clause, extract from full claim with demonyms/countries ignored
        if not entities:
            cap_full = re.findall(r'\b(?:[A-Z0-9][a-zA-Z0-9/\'-]*(?:\s+[A-Z0-9][a-zA-Z0-9/\'-]*)*)\b', claim)
            ignore_full = ignore | {"United States", "USA", "UK", "New York", "California"}
            for p in cap_full:
                words = [w for w in p.split() if w not in ignore_full]
                if words:
                    cleaned = " ".join(words)
                    if len(cleaned) >= 2 and cleaned not in entities:
                        entities.append(cleaned)

        return entities

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
        ents = self._extract_subject_entities(claim)

        for p in passages:
            e, c = self.best(claim, p)
            w = float(p.get("weight", 1.0))
            e, c = e * w, c * w
            title = p.get("title", "unknown")
            base_title = re.sub(r'\s*\([^)]*\)', '', title).strip().lower()
            if e >= theta_s:
                sup.append((title, e))

            # Contradiction requires that passage actually discusses the subject entity of the claim.
            # Topic-disjoint passages with different entities produce spurious cross-encoder contradictions.
            p_text = (
                title + " " + (" ".join(p.get("sentences", [])) if isinstance(p.get("sentences"), list) else p.get("text", ""))
            ).lower()

            if ents:
                title_in_claim = bool(len(base_title) >= 3 and base_title in claim.lower())
                ent_in_passage = any(
                    ent.lower() in title.lower() or re.search(rf"\b{re.escape(ent.lower())}\b", p_text)
                    for ent in ents
                )
                mentions_subject = title_in_claim or ent_in_passage
            else:
                subj_tokens = [w.lower() for w in re.findall(r"\b[a-zA-Z]{3,}\b", claim) if w.lower() not in {"the", "and", "was", "were", "are", "for", "with"}]
                mentions_subject = any(re.search(rf"\b{re.escape(tok)}\b", p_text) for tok in subj_tokens) if subj_tokens else True

            if c >= theta_c and mentions_subject:
                con.append((title, c))

        be = max((x for _, x in sup), default=0.0)
        bc = max((x for _, x in con), default=0.0)

        if sup and con:
            status = "supported" if be >= (bc - 0.05) else "contradicted"
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

