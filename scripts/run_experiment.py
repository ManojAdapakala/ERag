"""Main experiment runner for Standard RAG, Closed-book, IRCoT, and E-RAG."""

import argparse
import json
import os
import subprocess
import time
from typing import Any, Dict, Generator, List, Tuple
import mlflow
import numpy as np
import yaml
from tqdm import tqdm

from src.config import load_cfg
from src.eval.metrics import exact_match, f1_score, is_unknown, retrieval_recall
from src.llm.client import LLM, Tracker
from src.retrieval.index import Embedder, RetrieverFactory
from src.systems.standard import run_closed_book, run_standard


def flat(d: Dict[str, Any], prefix: str = "") -> Generator[Tuple[str, str], None, None]:
    """Flatten nested config dictionary for MLflow logging."""
    for k, v in d.items():
        if isinstance(v, dict):
            yield from flat(v, f"{prefix}{k}.")
        else:
            yield f"{prefix}{k}", str(v)


def apply_overrides(cfg: Dict[str, Any], kvs: List[str]) -> Dict[str, Any]:
    """Apply command-line overrides of the form key.subkey=value."""
    for kv in kvs:
        if "=" not in kv:
            continue
        key, val = kv.split("=", 1)
        node = cfg
        *path, last = key.split(".")
        for p in path:
            if p not in node or not isinstance(node[p], dict):
                node[p] = {}
            node = node[p]
        node[last] = yaml.safe_load(val)
    return cfg


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """Read a JSONL file into a list of dictionaries."""
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def git_commit() -> str:
    """Return short hash of current Git commit."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return "untracked"


def dispatch(
    system: str,
    item: Dict[str, Any],
    retriever: Any,
    llm: Any,
    nli: Any,
    cfg: Dict[str, Any],
    tr: Tracker,
    memory: Any,
) -> Dict[str, Any]:
    """Route question execution to the selected answering system."""
    if system == "standard":
        return run_standard(item, retriever, llm, cfg, tr)
    if system == "closed_book":
        return run_closed_book(item, retriever, llm, cfg, tr)
    if system == "ircot":
        from src.systems.ircot import run_ircot
        return run_ircot(item, retriever, llm, cfg, tr)
    if system == "erag":
        from src.systems.erag import run_erag
        return run_erag(item, retriever, llm, nli, cfg, tr, memory)
    raise ValueError(f"Unknown system identifier: {system}")


def summarize(path: str) -> Dict[str, Any]:
    """Calculate aggregate research metrics from a raw prediction JSONL file."""
    rows = load_jsonl(path)
    if not rows:
        return {"n": 0}

    ok = [r for r in rows if not r.get("failed", False)]
    ans = [r for r in ok if not r.get("unknown", False)]
    lat = [r.get("wall_latency", 0.0) for r in ok]

    em_scores = [r["em"] for r in rows if "em" in r]
    f1_scores = [r["f1"] for r in rows if "f1" in r]
    rec_scores = [r["recall"] for r in ok if r.get("recall") is not None]

    return {
        "n": len(rows),
        "failed": len(rows) - len(ok),
        "em": float(np.mean(em_scores)) if em_scores else 0.0,
        "f1": float(np.mean(f1_scores)) if f1_scores else 0.0,
        "coverage": len(ans) / max(len(rows), 1),
        "selective_em": float(np.mean([r["em"] for r in ans])) if ans else float("nan"),
        "recall": float(np.mean(rec_scores)) if rec_scores else float("nan"),
        "llm_calls": float(np.mean([r["llm_calls"] for r in ok])) if ok else 0.0,
        "retrievals": float(np.mean([r["retrievals"] for r in ok])) if ok else 0.0,
        "tokens_in": float(np.mean([r["in_tokens"] for r in ok])) if ok else 0.0,
        "tokens_out": float(np.mean([r["out_tokens"] for r in ok])) if ok else 0.0,
        "latency_median": float(np.median(lat)) if lat else 0.0,
        "latency_p95": float(np.quantile(lat, 0.95)) if lat else 0.0,
        "api_failures": int(sum(r.get("failures", 0) for r in rows)),
    }


def main():
    ap = argparse.ArgumentParser(description="Execute E-RAG research experiments.")
    ap.add_argument("--config", nargs="+", required=True, help="YAML config files (in order)")
    ap.add_argument("--set", nargs="*", default=[], help="Config overrides (key=val)")
    ap.add_argument("--limit", type=int, help="Optional question count limit")
    a = ap.parse_args()

    cfg = apply_overrides(load_cfg(*a.config), a.set)
    manifest_dir = cfg["data"]["manifest_dir"]
    system = cfg["run"]["system"]
    split = cfg["data"]["split"]

    manifest_file = os.path.join(manifest_dir, f"{split}.jsonl")
    items = load_jsonl(manifest_file)[: a.limit]

    pooled = None
    if cfg["retrieval"]["setting"] == "pooled":
        dev_path = os.path.join(manifest_dir, "dev.jsonl")
        test_path = os.path.join(manifest_dir, "test.jsonl")
        if split in ("dev", "test") and os.path.exists(dev_path) and os.path.exists(test_path):
            pooled = load_jsonl(dev_path) + load_jsonl(test_path)
        else:
            pooled = items

    emb = Embedder(cfg["retrieval"]["embedder"])
    factory = RetrieverFactory(cfg["retrieval"]["setting"], emb, pooled)
    llm = LLM(
        cfg,
        use_cache=cfg["run"]["cache"],
        namespace=cfg["run"]["cache_namespace"],
    )

    nli = None
    if system == "erag":
        from src.epistemic.nli import NLI
        nli = NLI(cfg["nli"]["model"], cfg["nli"]["window"])

    memory = None
    if cfg.get("memory", {}).get("enabled"):
        from src.systems.memory import BeliefMemory
        memory = BeliefMemory(emb, cfg)
        if cfg["memory"].get("plant"):
            with open(cfg["memory"]["plant"], "r", encoding="utf-8") as pf:
                for b in json.load(pf):
                    memory.insert(b["text"], b.get("conf", 0.9))

    os.makedirs(cfg["run"]["out_dir"], exist_ok=True)
    out_path = os.path.join(cfg["run"]["out_dir"], f"{cfg['run']['name']}.jsonl")

    # Crash recovery: skip questions already successfully completed in output file
    done_ids = set()
    if os.path.exists(out_path):
        for line in load_jsonl(out_path):
            if not line.get("failed", False):
                done_ids.add(line["id"])

    mlflow.set_experiment("erag")
    with mlflow.start_run(run_name=cfg["run"]["name"]):
        mlflow.log_params(dict(flat(cfg)))
        mlflow.set_tags({
            "git_commit": git_commit(),
            "system": system,
            "split": split,
            "retrieval_setting": cfg["retrieval"]["setting"],
        })

        with open(out_path, "a", encoding="utf-8") as f:
            for item in tqdm(items, desc=f"{system}:{split}"):
                if item["_id"] in done_ids:
                    continue

                tr = Tracker()
                t0 = time.time()
                failed = False

                try:
                    retriever = factory.for_item(item)
                    r = dispatch(system, item, retriever, llm, nli, cfg, tr, memory)
                except Exception as ex:
                    r = {
                        "answer": "",
                        "pool_titles": [],
                        "hops": 0,
                        "error": repr(ex),
                    }
                    failed = True

                wall = time.time() - t0
                if system != "closed_book":
                    rec_score, all_gold = retrieval_recall(r.get("pool_titles", []), item)
                else:
                    rec_score, all_gold = None, None

                rec = {
                    "id": item["_id"],
                    "system": system,
                    "setting": cfg["retrieval"]["setting"],
                    "split": split,
                    "type": item.get("type"),
                    "question": item["question"],
                    "gold": item["answer"],
                    "em": exact_match(r["answer"], item["answer"]),
                    "f1": f1_score(r["answer"], item["answer"]),
                    "unknown": is_unknown(r["answer"]),
                    "recall": rec_score,
                    "all_gold": all_gold,
                    "llm_calls": tr.llm_calls,
                    "retrievals": tr.retrievals,
                    "probe_searches": tr.probe_searches,
                    "in_tokens": tr.in_tokens,
                    "out_tokens": tr.out_tokens,
                    "api_latency": tr.latency,
                    "wall_latency": wall,
                    "cache_hits": tr.cache_hits,
                    "failures": tr.failures,
                    "failed": failed,
                }
                rec.update(r)

                f.write(json.dumps(rec) + "\n")
                f.flush()

        summary = summarize(out_path)
        mlflow.log_metrics({k: v for k, v in summary.items() if isinstance(v, (int, float)) and not np.isnan(v)})
        mlflow.log_artifact(out_path)

        print("\n=== Experiment Summary ===")
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
