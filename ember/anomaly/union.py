"""Label-free score combination and FAR-budgeted operating points.

EMBER-U combines detectors without fitted weights: each member score is converted
to an empirical p-value against a reference set of *held-out* background scores,
and the union statistic is the minimum p-value over members (a Bonferroni-style
union; with equal weights every member receives an equal share of the false-alarm
budget).  Thresholds are set so that at most ``floor(far * N_bg)`` reference
background bursts are flagged, and Clopper-Pearson intervals quantify the counting
uncertainty of both the recovery and the false-alarm rate.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
from scipy.stats import beta


def empirical_pvalues(reference: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """P(reference >= score) estimated from the reference background scores."""

    ref = np.sort(np.asarray(reference, dtype=float))
    x = np.asarray(scores, dtype=float)
    return 1.0 - np.searchsorted(ref, x, side="left") / len(ref)


def min_p_union(
    reference: Mapping[str, np.ndarray],
    scores: Mapping[str, np.ndarray],
    *,
    weights: Mapping[str, float] | None = None,
) -> np.ndarray:
    """Union statistic ``-min_m p_m / w_m`` (higher = more anomalous).

    Parameters
    ----------
    reference : {member: (N_bg,) held-out background scores}
    scores : {member: (...,) scores to combine}, same keys
    weights : optional share of the false-alarm budget per member (default equal)
    """

    members = list(reference)
    w = (
        np.ones(len(members))
        if weights is None
        else np.array([weights[m] for m in members], float)
    )
    p = np.stack(
        [
            empirical_pvalues(reference[m], scores[m]) / w[i]
            for i, m in enumerate(members)
        ],
        axis=-1,
    )
    return -p.min(axis=-1)


def threshold_at_far(background_scores: np.ndarray, far: float) -> float:
    """Largest threshold such that exactly floor(far*N) background scores exceed it.

    Detections are ``score > threshold``.
    """

    s = np.sort(np.asarray(background_scores, dtype=float))[::-1]
    k = int(np.floor(far * len(s)))
    return float(s[k]) if k < len(s) else float(s[-1] - 1e-12)


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact binomial confidence interval for k successes out of n."""

    lo = beta.ppf(alpha / 2, k, n - k + 1) if k > 0 else 0.0
    hi = beta.ppf(1 - alpha / 2, k + 1, n - k) if k < n else 1.0
    return float(lo), float(hi)


def recovery_at_far(
    background_scores: np.ndarray,
    event_scores: np.ndarray,
    far: float,
    *,
    event_labels: Sequence[int] | None = None,
) -> dict:
    """Recovery and false-alarm count at a FAR budget with Clopper-Pearson intervals.

    ``event_scores`` may be (N_ev,) or (K, N_ev) for K fold models; in the latter
    case the recovery is averaged over fold models and its spread is reported.
    """

    bg = np.asarray(background_scores, float)
    ev = np.atleast_2d(np.asarray(event_scores, float))
    thr = threshold_at_far(bg, far)
    det = ev > thr
    n_ev = ev.shape[1]
    tp = det.sum(axis=1).mean()
    out = {
        "threshold": thr,
        "tpr": float(det.mean()),
        "tpr_fold_sd": float(det.mean(axis=1).std()),
        "tp": float(tp),
        "tpr_ci": clopper_pearson(int(round(tp)), n_ev),
        "fp": int((bg > thr).sum()),
        "far": float((bg > thr).mean()),
        "far_ci": clopper_pearson(int((bg > thr).sum()), len(bg)),
        "detected": det.mean(axis=0) >= 0.5,
    }
    if event_labels is not None:
        lab = np.asarray(event_labels)
        for value in np.unique(lab):
            out[f"tpr_label{int(value)}"] = float(det[:, lab == value].mean())
    return out
