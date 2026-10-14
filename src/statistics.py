"""Statistical comparison of physiological features across strain levels.

Unit of analysis = SESSION, not window. Consecutive 60-s windows from one recording are
strongly autocorrelated; treating thousands of them as independent observations
(pseudo-replication) would make almost any difference 'significant'. We therefore summarise
each session by its median feature value and test those medians.

H0: the feature's distribution (of session medians) is the same for Low, Moderate and High.
H1: at least one strain level differs.
Test: Kruskal-Wallis H (non-parametric; no normality assumption, suitable for n ~ 3-5 per group).
Post-hoc: pairwise two-sided Mann-Whitney U with Holm correction for 3 comparisons.
Effect size: epsilon^2 for Kruskal-Wallis (0 = no effect, 1 = perfect separation).
Only the DEVELOPMENT split is analysed, so the test sets stay untouched.
"""
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

from .config import STRAIN_ORDER


def holm(pvals: list[float]) -> list[float]:
    order = np.argsort(pvals)
    adjusted, running = np.empty(len(pvals)), 0.0
    for rank, idx in enumerate(order):
        running = max(running, (len(pvals) - rank) * pvals[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted.tolist()


def session_level_tests(windows: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    med = windows.groupby(["session_id", "strain_label"])[features].median().reset_index()
    kw_rows, post_rows = [], []
    for f in features:
        groups = {c: med.loc[med.strain_label == c, f].dropna().to_numpy() for c in STRAIN_ORDER}
        groups = {c: g for c, g in groups.items() if len(g) >= 2}
        if len(groups) < 2:
            continue
        h, p = stats.kruskal(*groups.values())
        n, k = sum(len(g) for g in groups.values()), len(groups)
        eps2 = max(0.0, (h - k + 1) / (n - k))
        kw_rows.append(dict(feature=f, n_sessions=n, groups=k, H=h, p_value=p, epsilon_sq=eps2,
                            **{f"median_{c}": np.median(g) for c, g in groups.items()}))
        pairs = list(combinations(groups, 2))
        raw = [stats.mannwhitneyu(groups[a], groups[b], alternative="two-sided").pvalue for a, b in pairs]
        for (a, b), pr, pa in zip(pairs, raw, holm(raw)):
            post_rows.append(dict(feature=f, comparison=f"{a} vs {b}", p_raw=pr, p_holm=pa,
                                  median_a=np.median(groups[a]), median_b=np.median(groups[b])))
    return pd.DataFrame(kw_rows), pd.DataFrame(post_rows)


def window_level_effect_sizes(windows: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Descriptive only (no p-values): Kruskal-Wallis epsilon^2 on windows, showing how
    strongly each feature separates the classes at the resolution the model sees."""
    rows = []
    for f in features:
        groups = [windows.loc[windows.strain_label == c, f].dropna() for c in STRAIN_ORDER]
        groups = [g for g in groups if len(g) > 1]
        if len(groups) < 2:
            continue
        h, _ = stats.kruskal(*groups)
        n, k = sum(len(g) for g in groups), len(groups)
        rows.append(dict(feature=f, epsilon_sq_windows=(h - k + 1) / (n - k)))
    return pd.DataFrame(rows)
