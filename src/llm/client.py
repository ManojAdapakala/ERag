"""Groq LLM client with SQLite caching, retry backoff, and usage tracking."""

import hashlib
import json
import os
import sqlite3
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Tracker:
    """One tracker instance per question. The runner logs these metrics into output records."""
    llm_calls: int = 0
    in_tokens: int = 0
    out_tokens: int = 0
    latency: float = 0.0      # Seconds spent waiting on external API (cache hits add 0.0)
    cache_hits: int = 0
    retrievals: int = 0       # Retrievals that add external evidence to the pool
    probe_searches: int = 0   # Candidate queries evaluated during decision-making
    failures: int = 0         # Transient API failures prior to retry success/exhaustion


class LLM:
    """LLM inference interface with persistent SQLite caching and exponential retry backoff."""

    def __init__(
        self,
        cfg: Dict[str, Any],
        use_cache: bool = True,
        namespace: str = "default",
        cache_path: str = "results/cache.sqlite",
    ):
        self.m = cfg["model"]
        self.use_cache = use_cache
        self.namespace = namespace
        self.cache_path = cache_path

        # Lazy / safe client initialization
        self.api_key = os.environ.get("GROQ_API_KEY")
        self._client = None

        if self.use_cache:
            os.makedirs(os.path.dirname(self.cache_path) or ".", exist_ok=True)
            self.db = sqlite3.connect(self.cache_path, check_same_thread=False)
            self.db.execute("CREATE TABLE IF NOT EXISTS c (k TEXT PRIMARY KEY, v TEXT)")
            self.db.commit()
        else:
            self.db = None

    def close(self):
        """Close SQLite database connection if open."""
        if self.db is not None:
            self.db.close()
            self.db = None

    @property
    def client(self):
        if self._client is None:
            if not self.api_key:
                self.api_key = os.environ.get("GROQ_API_KEY")
            if not self.api_key:
                raise ValueError(
                    "GROQ_API_KEY environment variable is not set. "
                    "Please set GROQ_API_KEY in your environment or .env file."
                )
            from groq import Groq
            self._client = Groq(api_key=self.api_key, timeout=30.0)
        return self._client

    def _key(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        sample_idx: int,
    ) -> str:
        """Compute deterministic SHA-256 cache key covering all generation parameters."""
        payload = {
            "ns": self.namespace,
            "model": self.m["name"],
            "messages": messages,
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
            "reasoning_effort": self.m.get("reasoning_effort"),
            "sample_idx": int(sample_idx),
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()

    def chat(
        self,
        messages: List[Dict[str, str]],
        tracker: Tracker,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        sample_idx: int = 0,
    ) -> str:
        """Execute chat completion with cache-lookup, retry backoff, and usage tracking."""
        actual_max_tokens = max_tokens or self.m.get("max_tokens", 1024)
        key = self._key(messages, temperature, actual_max_tokens, sample_idx)

        # 1. Check persistent SQLite cache
        if self.use_cache and self.db is not None:
            row = self.db.execute("SELECT v FROM c WHERE k=?", (key,)).fetchone()
            if row:
                out = json.loads(row[0])
                tracker.cache_hits += 1
                tracker.llm_calls += 1
                tracker.in_tokens += out.get("in", 0)
                tracker.out_tokens += out.get("out", 0)
                return out["text"]

        # 2. Call Groq API with exponential backoff (1 try + 5 retries = 6 attempts)
        out = None
        dt = 0.0
        for attempt in range(6):
            try:
                kw: Dict[str, Any] = {
                    "model": self.m["name"],
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": actual_max_tokens,
                }
                if self.m.get("reasoning_effort"):
                    kw["reasoning_effort"] = self.m["reasoning_effort"]

                t0 = time.time()
                r = self.client.chat.completions.create(**kw)
                dt = time.time() - t0

                prompt_tokens = r.usage.prompt_tokens if (r.usage and r.usage.prompt_tokens) else 0
                completion_tokens = (
                    r.usage.completion_tokens if (r.usage and r.usage.completion_tokens) else 0
                )
                text_content = (r.choices[0].message.content or "").strip()

                out = {
                    "text": text_content,
                    "in": prompt_tokens,
                    "out": completion_tokens,
                }
                break
            except Exception as ex:
                tracker.failures += 1
                if attempt == 5:
                    raise RuntimeError(f"LLM call failed after 6 attempts: {repr(ex)}") from ex
                # Exponential backoff with rate-limit protection
                sleep_secs = min(2 ** attempt * 3, 45)
                time.sleep(sleep_secs)

        if out is None:
            raise RuntimeError("LLM completion returned None without exception")

        # 3. Update tracker metrics
        tracker.llm_calls += 1
        tracker.latency += dt
        tracker.in_tokens += out["in"]
        tracker.out_tokens += out["out"]

        # 4. Save to persistent cache
        if self.use_cache and self.db is not None:
            self.db.execute(
                "INSERT OR REPLACE INTO c (k, v) VALUES (?, ?)",
                (key, json.dumps(out)),
            )
            self.db.commit()

        return out["text"]

