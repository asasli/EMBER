"""Evaluation and detector-combination helpers for PSP anomaly analysis."""

from __future__ import annotations

from collections import OrderedDict, defaultdict
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd


def summarize_feature_discrimination(
    feature_matrix: np.ndarray,
    noise_idx: np.ndarray,
    signal_idx: np.ndarray,
    *,
    feature_names: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Rank features by how strongly they separate noise from anomalies."""

    try:
        from scipy.stats import mannwhitneyu
    except ImportError as exc:
        raise ImportError(
            "summarize_feature_discrimination requires SciPy. "
            "Install `pip install -e .`."
        ) from exc

    values = np.asarray(feature_matrix, dtype=float)
    noise_values = values[np.asarray(noise_idx, dtype=int)]
    signal_values = values[np.asarray(signal_idx, dtype=int)]
    n_features = values.shape[1]

    if feature_names is not None and len(feature_names) != n_features:
        raise ValueError(
            "feature_names must match the number of columns in feature_matrix."
        )

    rows: list[dict[str, object]] = []
    for feature_idx in range(n_features):
        stat, p_value = mannwhitneyu(
            noise_values[:, feature_idx],
            signal_values[:, feature_idx],
            alternative="two-sided",
        )
        n1 = len(noise_values)
        n2 = len(signal_values)
        effect_size = abs(1 - (2 * stat) / max(1, n1 * n2))
        rows.append(
            {
                "feature_idx": int(feature_idx),
                "feature_name": (
                    feature_names[feature_idx]
                    if feature_names is not None
                    else f"Feature {feature_idx}"
                ),
                "effect_size": float(effect_size),
                "p_value": float(p_value),
                "significant": bool(p_value < 0.05),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            ["effect_size", "p_value"],
            ascending=[False, True],
        )
        .reset_index(drop=True)
    )


def bootstrap_eval(
    y_true: np.ndarray,
    scores: np.ndarray,
    *,
    n_boot: int = 2000,
    seed: int = 42,
) -> dict[str, float | tuple[float, float]]:
    """Compute bootstrap AUC/AP estimates with a 95% AUC interval."""

    try:
        from sklearn.metrics import average_precision_score, roc_auc_score
    except ImportError as exc:
        raise ImportError(
            "bootstrap_eval requires scikit-learn. Install `pip install -e .[anomaly]`."
        ) from exc

    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    scores = np.nan_to_num(scores, nan=np.nanmedian(scores))

    try:
        auc = float(roc_auc_score(y_true, scores))
    except Exception:
        return {"auc": 0.5, "ap": 0.0, "auc_ci": (0.5, 0.5)}

    try:
        ap = float(average_precision_score(y_true, scores))
    except Exception:
        ap = 0.0

    aucs: list[float] = []
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        idx = rng.choice(len(y_true), len(y_true), replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        try:
            aucs.append(float(roc_auc_score(y_true[idx], scores[idx])))
        except Exception:
            pass

    if aucs:
        auc_ci = (float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5)))
    else:
        auc_ci = (auc, auc)

    return {"auc": auc, "ap": ap, "auc_ci": auc_ci}


def analyze_fpr(
    scores: np.ndarray,
    *args,
    fprs: Sequence[float] = (0.01, 0.025, 0.05, 0.10),
) -> dict[float, dict[str, float]]:
    """Measure true-positive recovery at noise-derived FPR thresholds.

    This accepts both the packaged signature
    ``analyze_fpr(scores, noise_idx, signal_idx)`` and the original notebook
    signature ``analyze_fpr(scores, labels_binary, noise_idx, signal_idx)``.
    """

    if len(args) == 2:
        noise_idx, signal_idx = args
    elif len(args) == 3:
        _, noise_idx, signal_idx = args
    else:
        raise TypeError(
            "analyze_fpr expects (scores, noise_idx, signal_idx) or "
            "(scores, labels_binary, noise_idx, signal_idx)."
        )

    scores = np.asarray(scores, dtype=float)
    noise_scores = scores[noise_idx]
    signal_scores = scores[signal_idx]

    results: dict[float, dict[str, float]] = {}
    for fpr in fprs:
        threshold = np.percentile(noise_scores, 100 * (1 - fpr))
        detections = signal_scores > threshold
        results[float(fpr)] = {
            "tpr": float(detections.sum() / max(1, len(signal_scores))),
            "n_det": int(detections.sum()),
            "n_tot": int(len(signal_scores)),
        }
    return results


def analyze_class(
    scores: np.ndarray,
    *args,
    fpr: float = 0.01,
) -> dict[str, float]:
    """Compute per-anomaly-class TPR at a shared noise-derived threshold.

    This accepts both the packaged signature
    ``analyze_class(scores, noise_idx, class1_idx, class2_idx)`` and the
    original notebook signature
    ``analyze_class(scores, labels_original, noise_idx, class1_idx, class2_idx)``.
    """

    if len(args) == 3:
        noise_idx, class1_idx, class2_idx = args
    elif len(args) == 4:
        _, noise_idx, class1_idx, class2_idx = args
    else:
        raise TypeError(
            "analyze_class expects (scores, noise_idx, class1_idx, class2_idx) or "
            "(scores, labels_original, noise_idx, class1_idx, class2_idx)."
        )

    threshold = np.percentile(np.asarray(scores)[noise_idx], 100 * (1 - fpr))
    scores = np.asarray(scores, dtype=float)
    return {
        "tpr_c1": float(
            (scores[class1_idx] > threshold).sum() / max(1, len(class1_idx))
        ),
        "tpr_c2": float(
            (scores[class2_idx] > threshold).sum() / max(1, len(class2_idx))
        ),
    }


def summarize_methods(
    all_methods: Mapping[str, np.ndarray],
    labels_binary: np.ndarray,
    noise_idx: np.ndarray,
    signal_idx: np.ndarray,
    *,
    labels_original: np.ndarray | None = None,
    class1_idx: np.ndarray | None = None,
    class2_idx: np.ndarray | None = None,
    n_boot: int = 2000,
    seed: int = 42,
) -> pd.DataFrame:
    """Build the notebook-style results dataframe for a detector bank."""

    labels_binary = np.asarray(labels_binary, dtype=int)
    if labels_original is not None:
        labels_original = np.asarray(labels_original, dtype=int)
        if class1_idx is None:
            class1_idx = np.where(labels_original == 1)[0]
        if class2_idx is None:
            class2_idx = np.where(labels_original == 2)[0]

    rows: list[dict[str, float | str]] = []
    for name, scores in all_methods.items():
        base = bootstrap_eval(labels_binary, scores, n_boot=n_boot, seed=seed)
        fpr_res = analyze_fpr(scores, noise_idx, signal_idx)

        row: dict[str, float | str] = {
            "method": name,
            "auc": float(base["auc"]),
            "ap": float(base["ap"]),
            "ci_lo": float(base["auc_ci"][0]),
            "ci_hi": float(base["auc_ci"][1]),
            "tpr_1": float(fpr_res[0.01]["tpr"]),
            "tpr_5": float(fpr_res[0.05]["tpr"]),
        }

        if class1_idx is not None and class2_idx is not None:
            cls = analyze_class(scores, noise_idx, class1_idx, class2_idx)
            row["tpr_c1"] = float(cls["tpr_c1"])
            row["tpr_c2"] = float(cls["tpr_c2"])

        rows.append(row)

    return pd.DataFrame(rows).sort_values("auc", ascending=False).reset_index(drop=True)


def threshold_detections(
    scores: np.ndarray,
    noise_idx: np.ndarray,
    *,
    fpr: float = 0.01,
) -> tuple[float, np.ndarray]:
    """Threshold scores using a target FAR/FPR percentile on the noise set."""

    scores = np.asarray(scores, dtype=float)
    threshold = float(np.percentile(scores[noise_idx], 100 * (1 - fpr)))
    detected_idx = np.where(scores > threshold)[0]
    return threshold, detected_idx


def threshold_zero_fp(
    scores: np.ndarray,
    noise_idx: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Threshold scores at the max noise score so no noise point is selected."""

    scores = np.asarray(scores, dtype=float)
    threshold = float(scores[noise_idx].max())
    detected_idx = np.where(scores > threshold)[0]
    return threshold, detected_idx


def hits_at_fpr(
    scores: np.ndarray,
    noise_idx: np.ndarray,
    signal_idx: np.ndarray,
    *,
    fpr_target: float = 0.01,
) -> tuple[int, int]:
    """Return true and false detections at a chosen FPR threshold."""

    _, detected_idx = threshold_detections(scores, noise_idx, fpr=fpr_target)
    detected_set = set(map(int, detected_idx))
    signal_set = set(map(int, signal_idx))
    true_hits = len(detected_set.intersection(signal_set))
    false_hits = len(detected_set - signal_set)
    return true_hits, false_hits


def evaluate_budgeted_accumulation(
    method_order: Sequence[str],
    case_name: str,
    all_methods: Mapping[str, np.ndarray],
    noise_idx: np.ndarray,
    signal_idx: np.ndarray,
    *,
    labels_original: np.ndarray | None = None,
    first_fpr: float = 0.01,
) -> dict[str, object]:
    """Evaluate a 1%-starter / zero-FP follow-up detector accumulation case."""

    signal_set = set(map(int, signal_idx))
    seen_signal: set[int] = set()
    union_fp: set[int] = set()
    method_records: list[dict[str, object]] = []
    method_to_signal_hits: dict[str, list[int]] = {}
    method_to_unique_hits: dict[str, list[int]] = {}

    for step, method in enumerate(method_order):
        scores = all_methods[method]
        if step == 0:
            threshold, detected_idx = threshold_detections(
                scores, noise_idx, fpr=first_fpr
            )
            stage_rule = f"fpr={first_fpr:.1%}"
        else:
            threshold, detected_idx = threshold_zero_fp(scores, noise_idx)
            stage_rule = "fp=0"

        detected_idx = sorted(map(int, detected_idx))
        detected_signal = sorted(set(detected_idx).intersection(signal_set))
        unique_signal = [idx for idx in detected_signal if idx not in seen_signal]
        seen_signal.update(unique_signal)

        false_positives = sorted(set(detected_idx) - signal_set)
        new_false_positives = [idx for idx in false_positives if idx not in union_fp]
        union_fp.update(false_positives)

        method_to_signal_hits[method] = detected_signal
        method_to_unique_hits[method] = unique_signal
        method_records.append(
            {
                "method": method,
                "stage_rule": stage_rule,
                "threshold": float(threshold),
                "raw_detections": int(len(detected_idx)),
                "true_anomalies_found": int(len(detected_signal)),
                "unique_new_anomalies": int(len(unique_signal)),
                "false_positives": int(len(false_positives)),
                "new_false_positives": int(len(new_false_positives)),
                "cumulative_union_false_positives": int(len(union_fp)),
                "cumulative_unique_anomalies": int(len(seen_signal)),
            }
        )

    summary_df = pd.DataFrame(method_records)
    combined_unique = sorted(seen_signal)
    missed_true = sorted(signal_set - set(combined_unique))

    vote_counts: defaultdict[int, int] = defaultdict(int)
    vote_methods: defaultdict[int, list[str]] = defaultdict(list)
    for method in method_order:
        for idx in method_to_signal_hits[method]:
            vote_counts[idx] += 1
            vote_methods[idx].append(method)

    vote_rows: list[dict[str, object]] = []
    for idx in combined_unique:
        row: dict[str, object] = {
            "sample_idx": int(idx),
            "votes": int(vote_counts[idx]),
            "methods": ", ".join(vote_methods[idx]),
        }
        if labels_original is not None:
            row["label"] = int(labels_original[idx])
        vote_rows.append(row)

    votes_df = (
        pd.DataFrame(vote_rows).sort_values(
            ["votes", "sample_idx"], ascending=[False, True]
        )
        if vote_rows
        else pd.DataFrame(columns=["sample_idx", "votes", "methods", "label"])
    )

    return {
        "case_name": case_name,
        "method_order": list(method_order),
        "summary_df": summary_df,
        "combined_unique_anomalies": combined_unique,
        "missed_true_anomalies": missed_true,
        "union_fp": sorted(union_fp),
        "union_far": len(union_fp) / max(1, len(noise_idx)),
        "votes_df": votes_df,
        "method_to_signal_hits": method_to_signal_hits,
        "method_to_unique_hits": method_to_unique_hits,
    }


def print_budgeted_case(
    case_result: Mapping[str, object],
    *,
    total_anomalies: int | None = None,
) -> None:
    """Pretty-print a detector accumulation case in the notebook style."""

    print(render_budgeted_case(case_result, total_anomalies=total_anomalies))


def render_budgeted_case(
    case_result: Mapping[str, object],
    *,
    total_anomalies: int | None = None,
) -> str:
    """Render a detector accumulation case in the notebook-style text format."""

    summary_df = case_result["summary_df"]
    method_order = case_result["method_order"]
    total_anomalies = (
        int(total_anomalies)
        if total_anomalies is not None
        else int(
            len(case_result["combined_unique_anomalies"])
            + len(case_result["missed_true_anomalies"])
        )
    )

    lines = [
        "=" * 80,
        f"  {case_result['case_name']}",
        "=" * 80,
        f"Using methods: {method_order}",
        "Schedule: starter @ 1% FAR, all follow-up detectors @ fp=0",
        "",
    ]

    for row in summary_df.itertuples(index=False):
        lines.append(
            f"{row.method:<22} {row.stage_rule:<8} thr={row.threshold:>10.4f}  "
            f"raw={row.raw_detections:>3d}  true={row.true_anomalies_found:>3d}  "
            f"unique={row.unique_new_anomalies:>3d}  cum={row.cumulative_unique_anomalies:>3d}  "
            f"fp={row.false_positives:>3d}  union_fp={row.cumulative_union_false_positives:>3d}"
        )

    lines.extend(
        [
            "",
            summary_df.to_string(index=False),
            "",
            f"Total known anomalies: {total_anomalies}",
            f"Unique anomalies recovered by union: {len(case_result['combined_unique_anomalies'])}",
            f"Coverage: {len(case_result['combined_unique_anomalies']) / max(1, total_anomalies):.1%}",
            (
                f"Union false positives kept: {len(case_result['union_fp'])} "
                f"({case_result['union_far']:.1%} of the noise set)"
            ),
            f"Still missed: {len(case_result['missed_true_anomalies'])}",
        ]
    )
    if case_result["union_fp"]:
        lines.append(f"False-positive indices in union: {case_result['union_fp']}")

    votes_df = case_result["votes_df"]
    if not votes_df.empty:
        lines.extend(
            [
                "",
                "Recovered anomalies with detector votes:",
                votes_df.to_string(index=False),
                "",
                "Unique anomalies contributed by each detector:",
            ]
        )
        for method in method_order:
            unique_hits = case_result["method_to_unique_hits"][method]
            lines.append(f"{method:<22}: {len(unique_hits):>3d} -> {unique_hits}")
    else:
        lines.append("No anomalies were recovered by this detector combination.")

    return "\n".join(lines)


def greedy_zero_fp_order_oracle(
    start_method: str,
    ranking: Sequence[str],
    all_methods: Mapping[str, np.ndarray],
    noise_idx: np.ndarray,
    signal_idx: np.ndarray,
    *,
    first_fpr: float = 0.01,
    max_methods: int = 6,
) -> list[str]:
    """Oracle ranking that uses labels to add the best zero-FP follow-up detector.

    This is intentionally labeled as an oracle because it uses ground-truth anomaly
    hits to choose the next detector and is therefore not suitable as a fair model
    selection rule.
    """

    ranking = [method for method in ranking if method in all_methods]
    signal_set = set(map(int, signal_idx))
    order = [start_method]
    _, starter_idx = threshold_detections(
        all_methods[start_method], noise_idx, fpr=first_fpr
    )
    seen_signal = set(map(int, set(starter_idx).intersection(signal_set)))
    remaining = [method for method in ranking if method != start_method]
    zero_fp_cache = {
        method: threshold_zero_fp(all_methods[method], noise_idx) for method in ranking
    }

    while remaining and len(order) < max_methods:
        best_method = None
        best_new_hits: list[int] = []
        best_total_hits: list[int] = []

        for method in remaining:
            _, detected_idx = zero_fp_cache[method]
            signal_hits = sorted(set(map(int, detected_idx)).intersection(signal_set))
            new_hits = [idx for idx in signal_hits if idx not in seen_signal]

            if len(new_hits) > len(best_new_hits) or (
                len(new_hits) == len(best_new_hits)
                and len(signal_hits) > len(best_total_hits)
            ):
                best_method = method
                best_new_hits = new_hits
                best_total_hits = signal_hits

        if best_method is None or len(best_new_hits) == 0:
            break

        order.append(best_method)
        seen_signal.update(best_new_hits)
        remaining.remove(best_method)

    return order


def greedy_zero_fp_order(
    start_method: str,
    ranking: Sequence[str],
    all_methods: Mapping[str, np.ndarray],
    noise_idx: np.ndarray,
    signal_idx: np.ndarray,
    *,
    first_fpr: float = 0.01,
    max_methods: int = 6,
) -> list[str]:
    """Notebook-compatible alias for the label-aware greedy zero-FP order.

    This helper is exploratory and uses the known anomaly hits to choose the
    next detector, so it should be interpreted as an oracle investigation
    rather than a fair model-selection rule.
    """

    return greedy_zero_fp_order_oracle(
        start_method,
        ranking,
        all_methods,
        noise_idx,
        signal_idx,
        first_fpr=first_fpr,
        max_methods=max_methods,
    )


def build_default_case_orders(
    results_df: pd.DataFrame,
    all_methods: Mapping[str, np.ndarray],
    noise_idx: np.ndarray,
    signal_idx: np.ndarray,
    *,
    max_methods: int = 6,
    first_fpr: float = 0.01,
    include_oracle: bool = False,
) -> OrderedDict[str, list[str]]:
    """Create the default set of exploratory detector-order comparisons.

    These orders are post-hoc because they are built from the evaluation
    summary itself. They are useful for qualitative case analysis, but they are
    not unbiased model-selection rules.
    """

    ranked_by_auc = results_df.sort_values("auc", ascending=False)["method"].tolist()
    if "tpr_1" in results_df.columns:
        ranked_by_tpr = results_df.sort_values(
            ["tpr_1", "auc"], ascending=[False, False]
        )["method"].tolist()
    else:
        ranked_by_tpr = ranked_by_auc

    candidate_cases: list[tuple[str, list[str]]] = [
        ("Post-hoc order: top AUC", ranked_by_auc[:max_methods]),
        ("Post-hoc order: top TPR@1%", ranked_by_tpr[:max_methods]),
    ]

    if include_oracle and ranked_by_auc:
        candidate_cases.extend(
            [
                (
                    f"Greedy zero-FP (oracle) after {ranked_by_auc[0]}",
                    greedy_zero_fp_order_oracle(
                        ranked_by_auc[0],
                        ranked_by_auc,
                        all_methods,
                        noise_idx,
                        signal_idx,
                        first_fpr=first_fpr,
                        max_methods=max_methods,
                    ),
                )
            ]
        )
        if ranked_by_tpr and ranked_by_tpr[0] != ranked_by_auc[0]:
            candidate_cases.append(
                (
                    f"Greedy zero-FP (oracle) after {ranked_by_tpr[0]}",
                    greedy_zero_fp_order_oracle(
                        ranked_by_tpr[0],
                        ranked_by_tpr,
                        all_methods,
                        noise_idx,
                        signal_idx,
                        first_fpr=first_fpr,
                        max_methods=max_methods,
                    ),
                )
            )

    case_orders: OrderedDict[str, list[str]] = OrderedDict()
    seen_orders: set[tuple[str, ...]] = set()
    for case_name, method_order in candidate_cases:
        method_order = [method for method in method_order if method in all_methods]
        order_key = tuple(method_order)
        if method_order and order_key not in seen_orders:
            case_orders[case_name] = method_order
            seen_orders.add(order_key)

    return case_orders


def compare_budgeted_cases(
    case_orders: Mapping[str, Sequence[str]] | Sequence[tuple[str, Sequence[str]]],
    all_methods: Mapping[str, np.ndarray],
    noise_idx: np.ndarray,
    signal_idx: np.ndarray,
    *,
    labels_original: np.ndarray | None = None,
    first_fpr: float = 0.01,
) -> tuple[list[dict[str, object]], pd.DataFrame, dict[str, dict[str, object]]]:
    """Evaluate many detector orders under the same accumulation budget rule."""

    items = case_orders.items() if isinstance(case_orders, Mapping) else case_orders
    case_results = [
        evaluate_budgeted_accumulation(
            method_order,
            case_name,
            all_methods,
            noise_idx,
            signal_idx,
            labels_original=labels_original,
            first_fpr=first_fpr,
        )
        for case_name, method_order in items
    ]

    comparison_rows: list[dict[str, object]] = []
    total_anomalies = max(1, len(signal_idx))
    for result in case_results:
        comparison_rows.append(
            {
                "case": result["case_name"],
                "starter": result["method_order"][0],
                "methods": len(result["method_order"]),
                "recovered": len(result["combined_unique_anomalies"]),
                "coverage": len(result["combined_unique_anomalies"]) / total_anomalies,
                "union_fp": len(result["union_fp"]),
                "union_far": result["union_far"],
                "order": " -> ".join(result["method_order"]),
            }
        )

    comparison_df = (
        pd.DataFrame(comparison_rows)
        .sort_values(
            ["coverage", "recovered", "union_fp", "methods"],
            ascending=[False, False, True, True],
        )
        .reset_index(drop=True)
    )
    case_results_by_name = {result["case_name"]: result for result in case_results}
    return case_results, comparison_df, case_results_by_name


def _minmax01(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    span = values.max() - values.min()
    if span == 0:
        return np.zeros_like(values)
    return (values - values.min()) / span


def build_case_scores(
    case_result: Mapping[str, object],
    all_methods: Mapping[str, np.ndarray],
    noise_idx: np.ndarray,
    *,
    first_fpr: float = 0.01,
    n_samples: int | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, list[int]]]:
    """Build vote and soft scores for a detector combination case."""

    if n_samples is None:
        n_samples = len(next(iter(all_methods.values())))

    vote_score = np.zeros(n_samples, dtype=float)
    soft_parts: list[np.ndarray] = []
    detection_map: dict[str, list[int]] = {}

    for step, method in enumerate(case_result["method_order"]):
        if step == 0:
            _, detected_idx = threshold_detections(
                all_methods[method], noise_idx, fpr=first_fpr
            )
        else:
            _, detected_idx = threshold_zero_fp(all_methods[method], noise_idx)

        detected_idx = np.asarray(detected_idx, dtype=int)
        vote_score[detected_idx] += 1.0
        soft_parts.append(_minmax01(all_methods[method]))
        detection_map[method] = sorted(map(int, detected_idx))

    soft_score = (
        np.mean(np.vstack(soft_parts), axis=0)
        if soft_parts
        else np.zeros(n_samples, dtype=float)
    )
    return vote_score, soft_score, detection_map
