import math
import numpy as np
import pytest
from src.eval.metrics import (
    auroc,
    brier,
    ece_by_level,
    exact_match,
    f1_score,
    is_unknown,
    normalize_answer,
    retrieval_recall,
)
from src.eval.stats import bootstrap_ci, holm, mcnemar_exact, paired_bootstrap


def test_normalisation():
    assert normalize_answer("The Beatles!") == "beatles"
    assert normalize_answer("An Apple a day.") == "apple day"
    assert normalize_answer("  hello   world  ") == "hello world"


def test_em_f1_known_cases():
    assert exact_match("The Beatles", "beatles") == 1.0
    assert exact_match("The Beatles", "The Rolling Stones") == 0.0

    # Special binary tokens
    assert f1_score("yes", "no") == 0.0
    assert f1_score("yes", "yes") == 1.0

    # Partial token overlap: "New York City" vs "New York"
    # precision = 2/3, recall = 2/2 = 1.0, F1 = 2 * (2/3) * 1 / (2/3 + 1) = 4/5 = 0.8
    assert abs(f1_score("New York City", "New York") - 0.8) < 1e-9


def test_unknown_counts_as_wrong():
    assert is_unknown("unknown") is True
    assert is_unknown("Unknown.") is True
    assert is_unknown("") is True
    assert is_unknown("Paris") is False

    # Unknown answers strictly count as 0.0 for EM and F1
    assert exact_match("unknown", "Paris") == 0.0
    assert f1_score("unknown", "Paris") == 0.0
    assert exact_match("", "Paris") == 0.0


def test_ece_levels():
    # 2 predictions with conf=1.0 and correct=1 (acc=1.0, diff=0)
    # 2 predictions with conf=0.2 and correct=0 (acc=0.0, diff=0.2)
    # Total ECE = 2/4 * 0 + 2/4 * 0.2 = 0.1
    err, rows = ece_by_level([1.0, 1.0, 0.2, 0.2], [1, 1, 0, 0])
    assert abs(err - 0.1) < 1e-9
    assert len(rows) == 2


def test_brier_score():
    # (1.0 - 1)^2 = 0; (0.0 - 0)^2 = 0 -> MSE = 0.0
    assert abs(brier([1.0, 0.0], [1, 0]) - 0.0) < 1e-9
    # (0.8 - 1)^2 = 0.04; (0.2 - 0)^2 = 0.04 -> MSE = 0.04
    assert abs(brier([0.8, 0.2], [1, 0]) - 0.04) < 1e-9


def test_auroc():
    # Perfect separation
    score = auroc([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0])
    assert abs(score - 1.0) < 1e-9

    # Degenerate case (all correct or all wrong) must return NaN
    nan_score = auroc([0.8, 0.9], [1, 1])
    assert math.isnan(nan_score)


def test_retrieval_recall():
    item = {
        "supporting_facts": [["Doc A", 0], ["Doc B", 1]],
    }
    rec, all_gold = retrieval_recall(["Doc A", "Doc C"], item)
    assert rec == 0.5
    assert all_gold is False

    rec, all_gold = retrieval_recall(["Doc A", "Doc B", "Doc C"], item)
    assert rec == 1.0
    assert all_gold is True


def test_bootstrap_ci_and_paired():
    x = [1.0, 1.0, 1.0, 0.0, 0.0]  # mean = 0.6
    mean, lo, hi = bootstrap_ci(x, n=5000, seed=42)
    assert abs(mean - 0.6) < 1e-9
    assert lo <= 0.6 <= hi

    a = [0.0, 0.0, 1.0, 1.0]
    b = [1.0, 1.0, 1.0, 1.0]
    diff_mean, diff_lo, diff_hi = paired_bootstrap(a, b, n=5000, seed=42)
    assert abs(diff_mean - 0.5) < 1e-9
    assert diff_lo >= 0.0
    assert diff_hi <= 1.0


def test_mcnemar_exact_and_holm():
    # a: [F, F, T, T], b: [T, T, T, T] -> discordant: b_only = 2, a_only = 0
    # Two-sided binomial test p-value for k=2, n=2, p=0.5 is P(X=0) + P(X=2) = 0.25 + 0.25 = 0.5
    res = mcnemar_exact([False, False, True, True], [True, True, True, True])
    assert res["a_only"] == 0
    assert res["b_only"] == 2
    assert abs(res["p"] - 0.5) < 1e-9

    # Holm correction
    pvals = [0.01, 0.04, 0.03]
    adj = holm(pvals)
    # Sorted: 0.01 (x3 -> 0.03), 0.03 (x2 -> 0.06), 0.04 (x1 -> 0.06)
    assert abs(adj[0] - 0.03) < 1e-9
    assert abs(adj[1] - 0.06) < 1e-9
    assert abs(adj[2] - 0.06) < 1e-9
