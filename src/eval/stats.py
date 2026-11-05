"""Statistical significance testing, bootstrap confidence intervals, and multiple-comparison corrections."""

from typing import Dict, List, Tuple
import numpy as np
from scipy.stats import binomtest


def bootstrap_ci(
    x: List[float],
    n: int = 10000,
    seed: int = 0,
    alpha: float = 0.05,
) -> Tuple[float, float, float]:
    """Compute non-parametric bootstrap confidence interval: (mean, lower_ci, upper_ci)."""
    arr = np.asarray(x, dtype=float)
    if len(arr) == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(arr), size=(n, len(arr)))
    means = arr[indices].mean(axis=1)
    return (
        float(arr.mean()),
        float(np.quantile(means, alpha / 2.0)),
        float(np.quantile(means, 1.0 - alpha / 2.0)),
    )


def paired_bootstrap(
    a: List[float],
    b: List[float],
    n: int = 10000,
    seed: int = 0,
    alpha: float = 0.05,
) -> Tuple[float, float, float]:
    """Compute paired bootstrap difference (b - a) over identical question ordering."""
    arr_a = np.asarray(a, dtype=float)
    arr_b = np.asarray(b, dtype=float)
    if len(arr_a) != len(arr_b):
        raise ValueError("Paired bootstrap inputs must have identical length")
    d = arr_b - arr_a
    return bootstrap_ci(d, n=n, seed=seed, alpha=alpha)


def mcnemar_exact(a_correct: List[bool], b_correct: List[bool]) -> Dict[str, float]:
    """Exact paired McNemar test using binomial test on discordant pairs."""
    a = np.asarray(a_correct, dtype=bool)
    b = np.asarray(b_correct, dtype=bool)
    if len(a) != len(b):
        raise ValueError("McNemar test inputs must have identical length")
    only_a = int((a & ~b).sum())
    only_b = int((~a & b).sum())
    n = only_a + only_b
    p = 1.0 if n == 0 else float(binomtest(only_b, n, 0.5).pvalue)
    return {"a_only": only_a, "b_only": only_b, "p": p}


def holm(pvals: List[float]) -> List[float]:
    """Holm-Bonferroni step-down multiple-comparison correction."""
    if not pvals:
        return []
    p_arr = np.asarray(pvals, dtype=float)
    order = np.argsort(p_arr)
    adj = [0.0] * len(pvals)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(pvals) - rank) * p_arr[i])
        adj[i] = min(1.0, float(running))
    return adj

