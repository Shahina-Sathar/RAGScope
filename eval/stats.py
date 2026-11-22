"""Statistics for the ablation study: bootstrap CIs and paired significance tests.

All resampling is over *questions* (the unit of analysis). Per-question scores
are first averaged over repeated runs, so run-to-run noise is absorbed into the
per-question value and reported separately as run-to-run variance.
"""
from __future__ import annotations

import numpy as np

N_BOOT = 10_000
N_PERM = 10_000
SEED = 12345


def _clean(x) -> np.ndarray:
    a = np.asarray(x, dtype=float)
    return a[~np.isnan(a)]


def bootstrap_ci(values, n_boot: int = N_BOOT, alpha: float = 0.05, seed: int = SEED):
    """Mean and percentile bootstrap CI of the mean. NaNs are dropped."""
    a = _clean(values)
    if a.size == 0:
        return float("nan"), float("nan"), float("nan"), 0
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, a.size, size=(n_boot, a.size))
    means = a[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(a.mean()), float(lo), float(hi), int(a.size)


def paired_test(a, b, n_boot: int = N_BOOT, n_perm: int = N_PERM, seed: int = SEED):
    """Paired comparison of b vs a on the same questions.

    Returns dict with mean_a, mean_b, abs delta (b - a), relative delta,
    95% paired-bootstrap CI of the delta, and a two-sided sign-flip
    permutation-test p-value. Only questions where both values exist are used.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = ~np.isnan(a) & ~np.isnan(b)
    a, b = a[mask], b[mask]
    n = int(a.size)
    if n == 0:
        nan = float("nan")
        return dict(n=0, mean_a=nan, mean_b=nan, delta=nan, rel=nan, ci_lo=nan, ci_hi=nan, p=nan)
    d = b - a
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = d[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    obs = d.mean()
    if np.allclose(d, 0):
        p = 1.0
    else:
        signs = rng.choice([-1.0, 1.0], size=(n_perm, n))
        perm = (signs * d).mean(axis=1)
        # +1 smoothing so p is never reported as exactly 0
        p = (np.sum(np.abs(perm) >= abs(obs) - 1e-12) + 1) / (n_perm + 1)
    ma = float(a.mean())
    return dict(
        n=n,
        mean_a=ma,
        mean_b=float(b.mean()),
        delta=float(obs),
        rel=float(obs / ma) if ma != 0 else float("nan"),
        ci_lo=float(lo),
        ci_hi=float(hi),
        p=float(p),
    )


def holm(pvalues: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (NaNs passed through)."""
    p = np.asarray(pvalues, dtype=float)
    out = np.full_like(p, np.nan)
    ok = np.where(~np.isnan(p))[0]
    if ok.size == 0:
        return out.tolist()
    order = ok[np.argsort(p[ok])]
    m = ok.size
    running = 0.0
    for rank, i in enumerate(order):
        adj = min(1.0, (m - rank) * p[i])
        running = max(running, adj)
        out[i] = running
    return out.tolist()


def percentile(values, q: float) -> float:
    a = _clean(values)
    return float(np.percentile(a, q)) if a.size else float("nan")
