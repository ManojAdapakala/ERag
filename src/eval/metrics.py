"""Evaluation metrics for HotpotQA answer correctness, retrieval quality, and confidence calibration."""

import re
import string
from collections import Counter
from typing import List, Set, Tuple
import numpy as np
from sklearn.metrics import roc_auc_score


def normalize_answer(s: str) -> str:
    """Official HotpotQA evaluation answer normalizer: lowercasing, punctuation removal, article removal."""
    if not isinstance(s, str):
        s = str(s or "")
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match(pred: str, gold: str) -> float:
    """Exact match metric: 1.0 if normalized answers match exactly, else 0.0."""
    return float(normalize_answer(pred) == normalize_answer(gold))


def f1_score(pred: str, gold: str) -> float:
    """Official token-level F1 score with atomic handling of yes/no/noanswer special cases."""
    p, g = normalize_answer(pred), normalize_answer(gold)
    special = ("yes", "no", "noanswer")
    if (p in special or g in special) and p != g:
        return 0.0

    pt, gt = p.split(), g.split()
    common = Counter(pt) & Counter(gt)
    same = sum(common.values())
    if same == 0 or len(pt) == 0 or len(gt) == 0:
        return 0.0

    prec = same / len(pt)
    rec = same / len(gt)
    return 2.0 * prec * rec / (prec + rec)


def is_unknown(pred: str) -> bool:
    """Check if prediction corresponds to an abstention / unknown answer."""
    return normalize_answer(pred) in ("unknown", "")


def gold_titles(item: dict) -> Set[str]:
    """Extract set of gold supporting paragraph titles from HotpotQA question item."""
    return {t for t, _ in item.get("supporting_facts", [])}


def retrieval_recall(retrieved_titles: List[str], item: dict) -> Tuple[float, bool]:
    """Compute retrieval recall and all-gold-retrieved boolean flag."""
    gold = gold_titles(item)
    if not gold:
        return 1.0, True
    got = gold & set(retrieved_titles)
    recall = len(got) / len(gold)
    all_gold = len(got) == len(gold)
    return recall, all_gold


def auroc(conf: List[float], correct: List[int]) -> float:
    """Compute Area Under the Receiver Operating Characteristic curve for confidence calibration."""
    if len(set(correct)) < 2:
        return float("nan")
    return float(roc_auc_score(correct, conf))


def ece_by_level(conf: List[float], correct: List[int]) -> Tuple[float, List[dict]]:
    """Compute Expected Calibration Error grouped by distinct confidence values (self-consistency levels)."""
    conf_arr = np.asarray(conf, dtype=float)
    corr_arr = np.asarray(correct, dtype=float)
    if len(conf_arr) == 0:
        return 0.0, []

    total = len(conf_arr)
    err = 0.0
    rows = []
    for c in sorted(set(conf_arr)):
        m = (conf_arr == c)
        acc = corr_arr[m].mean()
        err += m.sum() / total * abs(acc - c)
        rows.append({
            "confidence": float(c),
            "n": int(m.sum()),
            "accuracy": float(acc),
        })
    return float(err), rows


def brier(conf: List[float], correct: List[int]) -> float:
    """Compute Brier score (mean squared error of probabilistic predictions)."""
    if len(conf) == 0:
        return 0.0
    conf_arr = np.asarray(conf, dtype=float)
    corr_arr = np.asarray(correct, dtype=float)
    return float(np.mean((conf_arr - corr_arr) ** 2))

