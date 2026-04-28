"""Ensemble helpers for combining anomaly detector scores."""

from __future__ import annotations

from scipy.stats import rankdata
import numpy as np


def _as_score_arrays(scores_dict: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        name: np.asarray(scores, dtype=float) for name, scores in scores_dict.items()
    }


def _normalize_against_reference(
    reference_scores: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    reference_scores = np.asarray(reference_scores, dtype=float)
    values = np.asarray(values, dtype=float)
    span = reference_scores.max() - reference_scores.min()
    return (values - reference_scores.min()) / (span + 1e-10)


def _loo_rank(reference_scores: np.ndarray, test_score: float) -> float:
    combined = np.concatenate(
        [np.asarray(reference_scores, dtype=float), [float(test_score)]]
    )
    return float(rankdata(combined, method="average")[-1] / max(len(combined), 1))


def norm01(scores: np.ndarray) -> np.ndarray:
    """Min-max normalize scores into [0, 1]."""

    scores = np.asarray(scores, dtype=float)
    span = scores.max() - scores.min()
    return (scores - scores.min()) / (span + 1e-10)


def norm(scores: np.ndarray) -> np.ndarray:
    """Notebook-compatible alias for :func:`norm01`."""

    return norm01(scores)


def ens_rank(scores_dict: dict[str, np.ndarray]) -> np.ndarray:
    """Rank-based ensemble over a mapping of detector scores."""

    ranked = np.column_stack(
        [rankdata(np.asarray(scores)) for scores in scores_dict.values()]
    )
    return ranked.mean(axis=1)


def ens_rank_loo(scores_dict: dict[str, np.ndarray]) -> np.ndarray:
    """Strict rank-based ensemble with train-only ranks for each held-out point."""

    scores_dict = _as_score_arrays(scores_dict)
    keys = list(scores_dict)
    n_samples = len(next(iter(scores_dict.values()))) if keys else 0
    output = np.zeros(n_samples, dtype=float)

    for i in range(n_samples):
        mask = np.ones(n_samples, dtype=bool)
        mask[i] = False
        fold_ranks = [
            _loo_rank(scores_dict[key][mask], scores_dict[key][i]) for key in keys
        ]
        output[i] = float(np.mean(fold_ranks)) if fold_ranks else 0.0

    return output


def ens_mean(scores_dict: dict[str, np.ndarray]) -> np.ndarray:
    """Mean ensemble after min-max normalization."""

    normalized = np.column_stack(
        [norm01(np.asarray(scores)) for scores in scores_dict.values()]
    )
    return normalized.mean(axis=1)


def ens_mean_loo(scores_dict: dict[str, np.ndarray]) -> np.ndarray:
    """Strict mean ensemble with train-only min-max normalization."""

    scores_dict = _as_score_arrays(scores_dict)
    keys = list(scores_dict)
    n_samples = len(next(iter(scores_dict.values()))) if keys else 0
    output = np.zeros(n_samples, dtype=float)

    for i in range(n_samples):
        mask = np.ones(n_samples, dtype=bool)
        mask[i] = False
        fold_scores = [
            float(
                _normalize_against_reference(
                    scores_dict[key][mask], scores_dict[key][i]
                )
            )
            for key in keys
        ]
        output[i] = float(np.mean(fold_scores)) if fold_scores else 0.0

    return output


def ens_weighted(
    scores_dict: dict[str, np.ndarray],
    labels: np.ndarray,
) -> np.ndarray:
    """Notebook-style optimistic weighted ensemble using full-data AUCs."""

    try:
        from sklearn.metrics import roc_auc_score
    except ImportError as exc:
        raise ImportError(
            "ens_weighted requires scikit-learn. Install `pip install -e .[anomaly]`."
        ) from exc

    labels = np.asarray(labels, dtype=int)
    weights = []
    normalized_scores = []
    for name, scores in scores_dict.items():
        normalized = norm01(np.asarray(scores))
        normalized_scores.append(normalized)
        try:
            weights.append(float(roc_auc_score(labels, scores)))
        except Exception:
            weights.append(0.5)

    weight_arr = np.asarray(weights, dtype=float)
    if weight_arr.sum() < 1e-10:
        weight_arr = np.ones_like(weight_arr) / max(len(weight_arr), 1)
    else:
        weight_arr /= weight_arr.sum()

    stacked = np.column_stack(normalized_scores)
    return stacked @ weight_arr


def ens_weighted_loo(
    scores_dict: dict[str, np.ndarray],
    labels: np.ndarray,
) -> np.ndarray:
    """Weighted ensemble with leave-one-out weights derived from per-detector AUC."""

    try:
        from sklearn.metrics import roc_auc_score
    except ImportError as exc:
        raise ImportError(
            "ens_weighted_loo requires scikit-learn. Install `pip install -e .[anomaly]`."
        ) from exc

    labels = np.asarray(labels, dtype=int)
    scores_dict = _as_score_arrays(scores_dict)
    keys = list(scores_dict)
    final = np.zeros(len(labels), dtype=float)

    for i in range(len(labels)):
        mask = np.ones(len(labels), dtype=bool)
        mask[i] = False

        weights = []
        for j in range(len(keys)):
            try:
                auc_j = roc_auc_score(labels[mask], scores_dict[keys[j]][mask])
            except Exception:
                auc_j = 0.5
            weights.append(max(auc_j - 0.5, 0.0))

        weights = np.asarray(weights, dtype=float)
        if weights.sum() < 1e-10:
            weights = np.ones(len(keys), dtype=float) / max(len(keys), 1)
        else:
            weights /= weights.sum()

        fold_scores = np.asarray(
            [
                float(
                    _normalize_against_reference(
                        scores_dict[key][mask], scores_dict[key][i]
                    )
                )
                for key in keys
            ],
            dtype=float,
        )
        final[i] = float(weights @ fold_scores)

    return final


def ens_topk_by_auc(
    scores_dict: dict[str, np.ndarray],
    labels: np.ndarray,
    *,
    k: int = 5,
) -> tuple[np.ndarray, list[str]]:
    """Select the top-k detectors by full-data AUC before taking their mean ensemble.

    This reproduces the notebook helper and is convenient for comparison, but it is
    optimistic because it uses the evaluation labels for detector ranking.
    """

    try:
        from sklearn.metrics import roc_auc_score
    except ImportError as exc:
        raise ImportError(
            "ens_topk_by_auc requires scikit-learn. Install `pip install -e .[anomaly]`."
        ) from exc

    aucs: dict[str, float] = {}
    for name, scores in scores_dict.items():
        try:
            aucs[name] = float(roc_auc_score(labels, scores))
        except Exception:
            aucs[name] = 0.5

    top_names = sorted(aucs, key=lambda item: -aucs[item])[:k]
    return ens_mean({name: scores_dict[name] for name in top_names}), top_names


def ens_topk(
    scores_dict: dict[str, np.ndarray],
    labels: np.ndarray,
    *,
    k: int = 5,
) -> tuple[np.ndarray, list[str]]:
    """Notebook-compatible alias for the optimistic top-k AUC ensemble."""

    return ens_topk_by_auc(scores_dict, labels, k=k)


def ens_topk_loo(
    scores_dict: dict[str, np.ndarray],
    labels: np.ndarray,
    *,
    k: int = 5,
) -> tuple[np.ndarray, dict[str, object]]:
    """Strict top-k ensemble with train-only detector selection for each fold."""

    try:
        from sklearn.metrics import roc_auc_score
    except ImportError as exc:
        raise ImportError(
            "ens_topk_loo requires scikit-learn. Install `pip install -e .[anomaly]`."
        ) from exc

    labels = np.asarray(labels, dtype=int)
    scores_dict = _as_score_arrays(scores_dict)
    keys = list(scores_dict)
    final = np.zeros(len(labels), dtype=float)
    selection_counts = {name: 0 for name in keys}
    selected_per_sample: list[list[str]] = []

    for i in range(len(labels)):
        mask = np.ones(len(labels), dtype=bool)
        mask[i] = False

        aucs: dict[str, float] = {}
        for name in keys:
            try:
                aucs[name] = float(roc_auc_score(labels[mask], scores_dict[name][mask]))
            except Exception:
                aucs[name] = 0.5

        selected = sorted(aucs, key=lambda item: (-aucs[item], item))[:k]
        selected_per_sample.append(selected)
        for name in selected:
            selection_counts[name] += 1

        if selected:
            fold_scores = np.asarray(
                [
                    float(
                        _normalize_against_reference(
                            scores_dict[name][mask], scores_dict[name][i]
                        )
                    )
                    for name in selected
                ],
                dtype=float,
            )
            final[i] = float(fold_scores.mean())
        else:
            final[i] = 0.0

    most_common = sorted(
        selection_counts,
        key=lambda name: (-selection_counts[name], name),
    )[:k]
    return final, {
        "mode": "loo",
        "k": int(k),
        "most_common": most_common,
        "selection_counts": selection_counts,
        "selected_per_sample": selected_per_sample,
    }


def score_lightgbm_meta_learner(
    scores_dict: dict[str, np.ndarray],
    labels: np.ndarray,
    *,
    feature_order: list[str] | None = None,
    normalize: bool = True,
    strict: bool = True,
    seed: int = 42,
) -> np.ndarray:
    """Fit the notebook's leave-one-out LightGBM meta-learner."""

    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise ImportError(
            "score_lightgbm_meta_learner requires LightGBM. "
            "Install `pip install -e .[anomaly-notebook]`."
        ) from exc

    labels = np.asarray(labels, dtype=int)
    score_arrays = _as_score_arrays(scores_dict)
    feature_order = (
        list(feature_order) if feature_order is not None else sorted(score_arrays)
    )
    meta_x_raw = np.column_stack([score_arrays[name] for name in feature_order])
    if normalize and not strict:
        meta_x_global = np.column_stack(
            [
                _normalize_against_reference(score_arrays[name], score_arrays[name])
                for name in feature_order
            ]
        )
    else:
        meta_x_global = meta_x_raw
    output = np.zeros(len(labels), dtype=float)

    for i in range(len(labels)):
        mask = np.ones(len(labels), dtype=bool)
        mask[i] = False

        if normalize and strict:
            train_x = meta_x_raw[mask]
            test_x = meta_x_raw[i : i + 1]
            col_min = train_x.min(axis=0, keepdims=True)
            col_span = train_x.max(axis=0, keepdims=True) - col_min
            train_x = (train_x - col_min) / (col_span + 1e-10)
            test_x = (test_x - col_min) / (col_span + 1e-10)
        else:
            train_x = meta_x_global[mask]
            test_x = meta_x_global[i : i + 1]

        train_y = labels[mask]
        if len(np.unique(train_y)) < 2:
            output[i] = 0.5
            continue

        n_pos = int((train_y == 1).sum())
        n_neg = int((train_y == 0).sum())
        scale_pos_weight = n_neg / max(1, n_pos)

        model = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=150,
            num_leaves=10,
            learning_rate=0.05,
            min_child_samples=3,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=seed,
            verbose=-1,
            scale_pos_weight=scale_pos_weight,
            n_jobs=1,
        )
        model.fit(train_x, train_y)
        output[i] = float(model.predict_proba(test_x)[0, 1])

    return output


def rank_normalise(
    bg_cal_scores: np.ndarray,
    query_scores: np.ndarray,
) -> np.ndarray:
    """Calibration-based rank normalisation via empirical CDF.

    Each query score is mapped to its percentile rank within the background
    calibration set: ``searchsorted(sorted(bg_cal), query) / len(bg_cal)``.
    Scores above all calibration background receive a value approaching 1.0;
    scores below all receive 0.0.

    Parameters
    ----------
    bg_cal_scores : (M,) array
        Anomaly scores for held-out background calibration samples.
    query_scores : (N,) array
        Anomaly scores to normalise.

    Returns
    -------
    np.ndarray, shape (N,)
        Rank-normalised scores in [0, 1].
    """
    bg = np.sort(np.asarray(bg_cal_scores, dtype=np.float64))
    q = np.asarray(query_scores, dtype=np.float64)
    return np.searchsorted(bg, q, side="left").astype(np.float64) / max(len(bg), 1)


def optimise_ensemble_weights(
    norm_scores_bg_cal: np.ndarray,
    norm_scores_eval: np.ndarray,
    labels_binary: np.ndarray,
    *,
    target_far: float = 0.01,
    entropy_lambda: float = 0.08,
    seed: int = 42,
    maxiter: int = 500,
) -> np.ndarray:
    """Optimise ensemble weights via differential evolution.

    Maximises TPR at ``target_far`` on the eval set, with an entropy
    regularisation term to prevent single-detector collapse.

    Parameters
    ----------
    norm_scores_bg_cal : (M, D) array
        Rank-normalised detector scores for background calibration samples
        (one column per detector).
    norm_scores_eval : (N, D) array
        Rank-normalised detector scores for eval samples.
    labels_binary : (N,) int array
        Binary labels for eval samples (0 = background, 1 = anomaly).
    target_far : float
        False alarm rate budget (e.g. 0.01 = 1%).
    entropy_lambda : float
        Weight of the entropy regularisation term.
    seed : int
        Random seed for reproducibility.
    maxiter : int
        Maximum number of differential evolution iterations.

    Returns
    -------
    np.ndarray, shape (D,)
        Optimised non-negative weights (sum to 1).
    """
    try:
        from scipy.optimize import differential_evolution
    except ImportError as exc:
        raise ImportError(
            "optimise_ensemble_weights requires scipy. "
            "Install with `pip install scipy`."
        ) from exc

    X_cal = np.asarray(norm_scores_bg_cal, dtype=np.float64)
    X_eval = np.asarray(norm_scores_eval, dtype=np.float64)
    y = np.asarray(labels_binary, dtype=int)

    n_det = X_cal.shape[1]
    bg_eval_idx = np.where(y == 0)[0]
    an_eval_idx = np.where(y == 1)[0]
    n_bg_eval = len(bg_eval_idx)
    max_fp = int(np.floor(target_far * n_bg_eval))

    def _neg_tpr(w):
        w = np.asarray(w, dtype=np.float64)
        w = np.abs(w)
        w = w / (w.sum() + 1e-15)

        # threshold at target_far on bg_cal
        cal_ens = X_cal @ w
        eval_ens = X_eval @ w

        bg_cal_sorted = np.sort(cal_ens)[::-1]
        if max_fp < len(bg_cal_sorted):
            thr = bg_cal_sorted[max_fp]
        else:
            thr = bg_cal_sorted[-1] - 1e-9

        tpr = (eval_ens[an_eval_idx] >= thr).mean() if len(an_eval_idx) else 0.0

        # entropy regularisation: penalise collapsed weights
        entropy = -np.sum(w * np.log(w + 1e-15)) / np.log(n_det + 1e-15)
        return -(tpr + entropy_lambda * entropy)

    bounds = [(0.0, 1.0)] * n_det
    result = differential_evolution(
        _neg_tpr,
        bounds,
        seed=seed,
        maxiter=maxiter,
        tol=1e-4,
        popsize=15,
        mutation=(0.5, 1.0),
        recombination=0.7,
    )
    w_opt = np.abs(result.x)
    w_opt /= w_opt.sum() + 1e-15
    return w_opt


def greedy_cascade_detection(
    scores_dict: dict[str, np.ndarray],
    bg_idx: np.ndarray,
    anom_idx: np.ndarray,
    *,
    target_far: float = 0.01,
    order: list[str] | None = None,
) -> tuple[list[dict], dict]:
    """Greedy cascade: each detector adds unique anomalies within a global FAR budget.

    The combined system flags a sample if ANY detector fires.  Each
    detector's threshold is set from the **unflagged** background using the
    remaining FP budget so the union FAR stays ``<= target_far`` globally.

    Parameters
    ----------
    scores_dict:
        ``{name: scores_for_all_samples}`` — raw or normalised scores.
    bg_idx:
        Integer indices into the score arrays that correspond to background.
    anom_idx:
        Integer indices into the score arrays that correspond to anomalies.
    target_far:
        Global false-alarm rate budget (fraction, e.g. 0.01 for 1 %).
    order:
        Detector names in the order they should be applied.  If ``None``
        detectors are sorted by individual TPR at *target_far* (best first).

    Returns
    -------
    steps : list[dict]
        One entry per detector with keys
        ``detector, thr, new_tp, new_fp, cumulative_tp, cumulative_fp,
        tpr, far, skipped, detected_anom_mask``.
    summary : dict
        ``{total_tp, total_fp, tpr, far}``.
    """
    bg_idx = np.asarray(bg_idx, dtype=int)
    anom_idx = np.asarray(anom_idx, dtype=int)
    N_bg = len(bg_idx)
    N_anom = len(anom_idx)
    global_budget = int(np.floor(target_far * N_bg))

    def _thr_and_tpr(name: str) -> tuple[float, float]:
        s = np.asarray(scores_dict[name], float)
        bg_s = s[bg_idx]
        bg_sorted = np.sort(bg_s)[::-1]
        k = global_budget
        thr = float(bg_sorted[k] if k < len(bg_sorted) else bg_sorted[-1] - 1e-9)
        tpr = float((s[anom_idx] >= thr).mean())
        return thr, tpr

    if order is None:
        order = sorted(
            scores_dict.keys(), key=lambda n: _thr_and_tpr(n)[1], reverse=True
        )

    # Boolean masks over bg_idx / anom_idx positions
    flagged_bg = np.zeros(N_bg, dtype=bool)
    flagged_anom = np.zeros(N_anom, dtype=bool)
    union_fp = 0
    union_tp = 0
    steps: list[dict] = []

    for det_name in order:
        scores = np.asarray(scores_dict[det_name], float)
        bg_scores = scores[bg_idx]
        anom_scores = scores[anom_idx]

        remaining = global_budget - union_fp
        if remaining <= 0:
            steps.append(
                {
                    "detector": det_name,
                    "thr": float("nan"),
                    "new_tp": 0,
                    "new_fp": 0,
                    "cumulative_tp": union_tp,
                    "cumulative_fp": union_fp,
                    "tpr": union_tp / max(N_anom, 1),
                    "far": union_fp / max(N_bg, 1),
                    "skipped": True,
                    "detected_anom_mask": flagged_anom.copy(),
                }
            )
            continue

        # Threshold from UNFLAGGED background only
        unflagged_bg_s = bg_scores[~flagged_bg]
        if len(unflagged_bg_s) == 0:
            unflagged_sorted = np.array([float("inf")])
        else:
            unflagged_sorted = np.sort(unflagged_bg_s)[::-1]
        k = remaining - 1  # allow at most `remaining` new FPs (0-indexed)
        thr = float(
            unflagged_sorted[k]
            if k < len(unflagged_sorted)
            else unflagged_sorted[-1] - 1e-9
        )

        new_bg_mask = (bg_scores >= thr) & (~flagged_bg)
        new_anom_mask = (anom_scores >= thr) & (~flagged_anom)
        new_fp = int(new_bg_mask.sum())
        new_tp = int(new_anom_mask.sum())

        flagged_bg |= new_bg_mask
        flagged_anom |= new_anom_mask
        union_fp += new_fp
        union_tp += new_tp

        steps.append(
            {
                "detector": det_name,
                "thr": thr,
                "new_tp": new_tp,
                "new_fp": new_fp,
                "cumulative_tp": union_tp,
                "cumulative_fp": union_fp,
                "tpr": union_tp / max(N_anom, 1),
                "far": union_fp / max(N_bg, 1),
                "skipped": False,
                "detected_anom_mask": flagged_anom.copy(),
            }
        )

    summary = {
        "total_tp": union_tp,
        "total_fp": union_fp,
        "tpr": union_tp / max(N_anom, 1),
        "far": union_fp / max(N_bg, 1),
        "order": order,
    }
    return steps, summary


def compute_default_ensembles(
    scores_dict: dict[str, np.ndarray],
    labels: np.ndarray,
    *,
    topk_values: tuple[int, ...] = (3, 5),
    strict: bool = True,
) -> tuple[dict[str, np.ndarray], dict[int, dict[str, object]]]:
    """Build the default ensemble suite.

    By default this uses strict leave-one-out evaluation. Set ``strict=False``
    only when reproducing the older optimistic notebook behavior.
    """

    if strict:
        ensembles = {
            "Ens-Rank": ens_rank_loo(scores_dict),
            "Ens-Mean": ens_mean_loo(scores_dict),
            "Ens-Weighted": ens_weighted_loo(scores_dict, labels),
        }
    else:
        ensembles = {
            "Ens-Rank": ens_rank(scores_dict),
            "Ens-Mean": ens_mean(scores_dict),
            "Ens-Weighted": ens_weighted(scores_dict, labels),
        }
    topk_names: dict[int, dict[str, object]] = {}

    for k in topk_values:
        if k < 1:
            continue
        if strict:
            topk_scores, selected = ens_topk_loo(scores_dict, labels, k=k)
        else:
            topk_scores, selected_names = ens_topk_by_auc(scores_dict, labels, k=k)
            selected = {
                "mode": "full-data",
                "k": int(k),
                "most_common": list(selected_names),
                "selection_counts": {
                    name: int(name in selected_names) for name in scores_dict
                },
                "selected_per_sample": [list(selected_names)] * len(labels),
            }
        ensembles[f"Ens-Top{k}"] = topk_scores
        topk_names[int(k)] = selected

    return ensembles, topk_names
