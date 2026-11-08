"""Paragraph retrieval using SentenceTransformers/FAISS IndexFlatIP (Dense) or rank-bm25 BM25Plus (Sparse)."""

import hashlib
import re
from typing import Any, Dict, List, Optional
import faiss
import numpy as np
from rank_bm25 import BM25Plus
from sentence_transformers import SentenceTransformer


MODEL_ALIASES: Dict[str, str] = {
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "all-minilm-l6-v2": "sentence-transformers/all-MiniLM-L6-v2",
    "dense": "sentence-transformers/all-MiniLM-L6-v2",
    "e5": "intfloat/e5-base-v2",
    "e5-base-v2": "intfloat/e5-base-v2",
    "intfloat/e5-base-v2": "intfloat/e5-base-v2",
    "bge": "BAAI/bge-base-en-v1.5",
    "bge-base-en-v1.5": "BAAI/bge-base-en-v1.5",
    "baai/bge-base-en-v1.5": "BAAI/bge-base-en-v1.5",
}


class Embedder:
    """SentenceTransformer wrapper with in-memory memoization, asymmetric prefix formatting, and normalized embeddings."""

    def __init__(self, name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.name = MODEL_ALIASES.get(name.lower(), name)
        self.model = SentenceTransformer(self.name)
        self.memo: Dict[str, np.ndarray] = {}

    def format_query(self, query: str) -> str:
        """Apply model-specific asymmetric query prefix if required."""
        low = self.name.lower()
        if "e5" in low:
            if not query.startswith("query: "):
                return f"query: {query}"
        elif "bge" in low:
            instruction = "Represent this sentence for searching relevant passages: "
            if not query.startswith(instruction):
                return f"{instruction}{query}"
        return query

    def format_passage(self, passage: str) -> str:
        """Apply model-specific passage prefix if required."""
        low = self.name.lower()
        if "e5" in low:
            if not passage.startswith("passage: "):
                return f"passage: {passage}"
        return passage

    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode raw texts into L2-normalized float32 vectors with memoization."""
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

    def encode_queries(self, queries: List[str]) -> np.ndarray:
        """Format queries with model-specific prefixes and encode."""
        formatted = [self.format_query(q) for q in queries]
        return self.encode(formatted)

    def encode_passages(self, passages: List[str]) -> np.ndarray:
        """Format passages with model-specific prefixes and encode."""
        formatted = [self.format_passage(p) for p in passages]
        return self.encode(formatted)


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

        passages = [p["text"] for p in paragraphs]
        X = embedder.encode_passages(passages)
        self.index = faiss.IndexFlatIP(X.shape[1])
        self.index.add(X)

    def search(self, query: str, k: int = 3) -> List[dict]:
        """Retrieve top-k most similar paragraphs for query."""
        if not self.paras or self.index is None:
            return []
        q = self.emb.encode_queries([query])
        k_search = min(k, len(self.paras))
        scores, ids = self.index.search(q, k_search)
        return [
            dict(self.paras[i], score=float(s))
            for s, i in zip(scores[0], ids[0])
            if i >= 0
        ]


class BM25Retriever:
    """Sparse lexical BM25 retriever using rank-bm25 BM25Plus (Lucene-compatible lower-bound IDF)."""

    def __init__(self, paragraphs: List[dict]):
        self.paras = paragraphs
        if not paragraphs:
            self.bm25 = None
            return
        self.corpus_tokens = [self._tokenize(p["text"]) for p in paragraphs]
        self.bm25 = BM25Plus(self.corpus_tokens)

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Simple, standard word-level tokenizer for BM25: lowercase alphanumeric tokens."""
        return [w.strip() for w in re.findall(r"\w+", text.lower()) if w.strip()]

    def search(self, query: str, k: int = 3) -> List[dict]:
        """Retrieve top-k highest-scoring BM25 paragraphs for query."""
        if not self.paras or self.bm25 is None:
            return []
        q_tokens = self._tokenize(query)
        if not q_tokens:
            return [dict(p, score=0.0) for p in self.paras[:k]]
        scores = self.bm25.get_scores(q_tokens)
        k_search = min(k, len(self.paras))
        top_indices = np.argsort(scores)[::-1][:k_search]
        return [
            dict(self.paras[i], score=float(scores[i]))
            for i in top_indices
        ]


class RetrieverFactory:
    """Factory supporting distractor (per-question index) and pooled (shared corpus index)."""

    def __init__(
        self,
        setting: str,
        embedder: Optional[Embedder] = None,
        pooled_items: Optional[List[dict]] = None,
        retriever_type: str = "dense",
    ):
        self.setting = setting
        self.retriever_type = retriever_type.lower()

        if self.retriever_type == "bm25":
            self.emb = None
        elif embedder is not None:
            self.emb = embedder
        else:
            resolved_model = MODEL_ALIASES.get(self.retriever_type, self.retriever_type)
            self.emb = Embedder(resolved_model)

        self.pooled: Optional[Any] = None
        if setting == "pooled" and pooled_items is not None:
            pooled_paras = build_pooled(pooled_items)
            if self.retriever_type == "bm25":
                self.pooled = BM25Retriever(pooled_paras)
            else:
                self.pooled = Retriever(pooled_paras, self.emb)

    def for_item(self, item: dict) -> Any:
        """Return the shared pooled index or construct a per-question distractor index."""
        if self.pooled is not None:
            return self.pooled
        paras = paragraphs_from_item(item)
        if self.retriever_type == "bm25":
            return BM25Retriever(paras)
        return Retriever(paras, self.emb)
