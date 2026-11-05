"""Dense paragraph retrieval using SentenceTransformers and FAISS IndexFlatIP."""

import hashlib
from typing import Dict, List, Optional
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class Embedder:
    """SentenceTransformer wrapper with in-memory memoization and normalized embeddings."""

    def __init__(self, name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(name)
        self.memo: Dict[str, np.ndarray] = {}

    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts into L2-normalized float32 vectors with memoization."""
        need = [t for t in dict.fromkeys(texts) if t not in self.memo]
        if need:
            vecs = self.model.encode(
                need,
                normalize_embeddings=True,
                batch_size=64,
                show_progress_bar=False,
            )
            for t, v in zip(need, vecs):
                self.memo[t] = v.astype("float32")
        return np.stack([self.memo[t] for t in texts])


def paragraphs_from_item(item: dict) -> List[dict]:
    """Extract standard paragraphs with prepended title matching Phase 1 baseline."""
    return [
        {
            "title": t,
            "sentences": s,
            "text": f"{t}. " + " ".join(s),
        }
        for t, s in item.get("context", [])
    ]


def build_pooled(items: List[dict]) -> List[dict]:
    """Deduplicate paragraphs across multiple questions for the pooled setting."""
    seen = set()
    out = []
    for it in items:
        for p in paragraphs_from_item(it):
            key = hashlib.md5((p["title"] + "||" + p["text"]).encode("utf-8")).hexdigest()
            if key not in seen:
                seen.add(key)
                out.append(p)
    return out


class Retriever:
    """FAISS dense retriever executing cosine similarity via inner product on normalized vectors."""

    def __init__(self, paragraphs: List[dict], embedder: Embedder):
        self.paras = paragraphs
        self.emb = embedder
        if not paragraphs:
            self.index = None
            return

        X = embedder.encode([p["text"] for p in paragraphs])
        self.index = faiss.IndexFlatIP(X.shape[1])
        self.index.add(X)

    def search(self, query: str, k: int = 3) -> List[dict]:
        """Retrieve top-k most similar paragraphs for query."""
        if not self.paras or self.index is None:
            return []
        q = self.emb.encode([query])
        k_search = min(k, len(self.paras))
        scores, ids = self.index.search(q, k_search)
        return [
            dict(self.paras[i], score=float(s))
            for s, i in zip(scores[0], ids[0])
            if i >= 0
        ]


class RetrieverFactory:
    """Factory supporting distractor (per-question index) and pooled (shared corpus index)."""

    def __init__(self, setting: str, embedder: Embedder, pooled_items: Optional[List[dict]] = None):
        self.setting = setting
        self.emb = embedder
        self.pooled: Optional[Retriever] = None
        if setting == "pooled" and pooled_items is not None:
            pooled_paras = build_pooled(pooled_items)
            self.pooled = Retriever(pooled_paras, embedder)

    def for_item(self, item: dict) -> Retriever:
        """Return the shared pooled index or construct a per-question distractor index."""
        if self.pooled is not None:
            return self.pooled
        return Retriever(paragraphs_from_item(item), self.emb)
