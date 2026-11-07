"""Independent Verification of Observational E-RAG Parity with Standard RAG on Test-400 (Gate G1)."""

import hashlib
import json
import os
import sys
from typing import Any, Dict, List
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.eval.metrics import exact_match, f1_score


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def compute_sha256_answers(records: List[Dict[str, Any]]) -> str:
    h = hashlib.sha256()
    for r in records:
        h.update(r.get("id", "").encode("utf-8"))
        h.update(b"::")
        h.update(r.get("answer", "").strip().encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def main():
    std_path = "results/standard_test.jsonl"
    obs_path = "results/erag_observe_test.jsonl"

    std_records = load_jsonl(std_path)
    obs_records = load_jsonl(obs_path)

    print(f"Loaded Standard RAG: {len(std_records)} records from {std_path}")
    print(f"Loaded Observe RAG:  {len(obs_records)} records from {obs_path}")

    std_dict = {r["id"]: r for r in std_records}
    obs_dict = {r["id"]: r for r in obs_records}

    mismatches = []
    em_mismatches = []
    f1_mismatches = []

    for qid, std_rec in std_dict.items():
        if qid not in obs_dict:
            mismatches.append((qid, "MISSING_IN_OBSERVE", "", ""))
            continue
        obs_rec = obs_dict[qid]
        
        std_ans = std_rec.get("answer", "").strip()
        obs_ans = obs_rec.get("answer", "").strip()

        if std_ans != obs_ans:
            mismatches.append((qid, "ANSWER_MISMATCH", std_ans, obs_ans))

        if abs(std_rec.get("em", 0.0) - obs_rec.get("em", 0.0)) > 1e-6:
            em_mismatches.append((qid, std_rec.get("em"), obs_rec.get("em")))

        if abs(std_rec.get("f1", 0.0) - obs_rec.get("f1", 0.0)) > 1e-6:
            f1_mismatches.append((qid, std_rec.get("f1"), obs_rec.get("f1")))

    # Metric calculations
    std_em = float(np.mean([r.get("em", 0.0) for r in std_records]))
    std_f1 = float(np.mean([r.get("f1", 0.0) for r in std_records]))
    obs_em = float(np.mean([r.get("em", 0.0) for r in obs_records]))
    obs_f1 = float(np.mean([r.get("f1", 0.0) for r in obs_records]))

    std_hash = compute_sha256_answers(std_records)
    obs_hash = compute_sha256_answers(obs_records)

    print("\n================ GATE G1 VERIFICATION AUDIT ================")
    print(f"1. Standard RAG sample count: {len(std_records)}")
    print(f"2. Observe RAG sample count:  {len(obs_records)}")
    print(f"3. Answer string mismatches:  {len(mismatches)}")
    print(f"4. EM metric mismatches:      {len(em_mismatches)}")
    print(f"5. F1 metric mismatches:      {len(f1_mismatches)}")
    print(f"6. Standard RAG EM:           {std_em:.4f}")
    print(f"7. Observe RAG EM:            {obs_em:.4f}")
    print(f"8. Standard RAG F1:           {std_f1:.4f}")
    print(f"9. Observe RAG F1:            {obs_f1:.4f}")
    print(f"10. Standard Answers SHA-256: {std_hash}")
    print(f"11. Observe Answers SHA-256:  {obs_hash}")
    print(f"12. SHA-256 Parity Match:     {'YES (IDENTICAL)' if std_hash == obs_hash else 'NO (MISMATCH)'}")

    gate_g1_passed = (
        len(std_records) == 400
        and len(obs_records) == 400
        and len(mismatches) == 0
        and std_hash == obs_hash
        and abs(std_em - obs_em) < 1e-9
        and abs(std_f1 - obs_f1) < 1e-9
    )

    print(f"13. Gate G1 Status:           {'PASSED (100.0% EQUIVALENCE)' if gate_g1_passed else 'FAILED'}")
    print("============================================================\n")

    if mismatches:
        print("First 5 mismatches:")
        for m in mismatches[:5]:
            print(m)
        sys.exit(1)


if __name__ == "__main__":
    main()

