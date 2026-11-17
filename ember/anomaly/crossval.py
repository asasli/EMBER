"""Out-of-fold background scoring.

The evaluation protocol of the EMBER paper: the background bursts are split into
K folds; for each fold a detector is fitted on the other folds and scores the
held-out background fold plus every event.  Each background burst therefore
receives one score from a model that never saw it, and every event receives K
scores (one per fold model).  Thresholds, empirical p-values and false-alarm rates
are then computed on the pooled out-of-fold background scores.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np


@dataclass
class OutOfFoldScores:
    """Container returned by :func:`out_of_fold_scores`."""

    names: list[str]
    background_idx: np.ndarray  # (N_bg,) dataset indices
    event_idx: np.ndarray  # (N_ev,)
    background: dict[str, np.ndarray]  # name -> (N_bg,) out-of-fold scores
    events: dict[str, np.ndarray]  # name -> (K, N_ev) per-fold-model scores
    test_folds: list[np.ndarray]


def background_folds(
    background_idx: Sequence[int], k: int = 5, seed: int = 42
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Shuffled K-fold split of the background indices -> [(train_idx, test_idx), ...].

    Identical to ``sklearn.model_selection.KFold(k, shuffle=True, random_state=seed)``
    (the split used for the numbers in the EMBER paper), without the dependency.
    """

    bg = np.asarray(background_idx)
    perm = np.arange(len(bg))
    np.random.RandomState(seed).shuffle(perm)
    chunks = np.array_split(perm, k)
    return [
        (
            bg[np.sort(np.concatenate([c for j, c in enumerate(chunks) if j != i]))],
            bg[np.sort(chunks[i])],
        )
        for i in range(k)
    ]


def out_of_fold_scores(
    fit_and_score: Callable[[np.ndarray], dict[str, np.ndarray]],
    background_idx: Sequence[int],
    event_idx: Sequence[int],
    *,
    k: int = 5,
    seed: int = 42,
    verbose: bool = False,
) -> OutOfFoldScores:
    """Run the K-fold protocol.

    Parameters
    ----------
    fit_and_score : callable(train_idx) -> {detector_name: scores over ALL dataset rows}
        Fits every detector on the given background training indices and returns
        scores for the whole dataset (only held-out background and events are kept).
    """

    bg = np.asarray(background_idx)
    ev = np.asarray(event_idx)
    folds = background_folds(bg, k=k, seed=seed)
    pos = {int(i): r for r, i in enumerate(bg)}
    background: dict[str, np.ndarray] = {}
    events: dict[str, np.ndarray] = {}
    for j, (tr, te) in enumerate(folds):
        scores = fit_and_score(tr)
        for name, s in scores.items():
            s = np.asarray(s, float)
            background.setdefault(name, np.full(len(bg), np.nan))[
                [pos[int(i)] for i in te]
            ] = s[te]
            events.setdefault(name, np.zeros((k, len(ev))))[j] = s[ev]
        if verbose:
            print(
                f"fold {j + 1}/{k} done ({len(tr)} train, {len(te)} held-out background)",
                flush=True,
            )
    for name, arr in background.items():
        if np.isnan(arr).any():
            raise RuntimeError(f"detector {name!r} left background bursts unscored")
    return OutOfFoldScores(
        sorted(background), bg, ev, background, events, [te for _, te in folds]
    )
