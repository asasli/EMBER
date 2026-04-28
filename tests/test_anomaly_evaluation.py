from __future__ import annotations

import numpy as np
import pandas as pd

from ember.anomaly import (
    build_case_scores,
    build_default_case_orders,
    compare_budgeted_cases,
    evaluate_budgeted_accumulation,
    greedy_zero_fp_order,
    render_budgeted_case,
    summarize_feature_discrimination,
    threshold_detections,
    threshold_zero_fp,
)


def _toy_scores():
    noise_idx = np.array([0, 1, 2])
    signal_idx = np.array([3, 4])
    labels_original = np.array([0, 0, 0, 1, 2])
    all_methods = {
        "Detector-A": np.array([0.10, 0.20, 0.30, 0.95, 0.99]),
        "Detector-B": np.array([0.05, 0.10, 0.11, 0.20, 0.90]),
    }
    return all_methods, noise_idx, signal_idx, labels_original


def test_threshold_helpers() -> None:
    all_methods, noise_idx, _, _ = _toy_scores()

    threshold, idx = threshold_detections(
        all_methods["Detector-A"], noise_idx, fpr=0.01
    )
    zero_fp_threshold, zero_fp_idx = threshold_zero_fp(
        all_methods["Detector-A"], noise_idx
    )

    assert threshold > 0.29
    assert idx.tolist() == [2, 3, 4]
    assert zero_fp_threshold == 0.30
    assert zero_fp_idx.tolist() == [3, 4]


def test_budgeted_case_and_scores() -> None:
    all_methods, noise_idx, signal_idx, labels_original = _toy_scores()
    case = evaluate_budgeted_accumulation(
        ["Detector-A", "Detector-B"],
        "Toy case",
        all_methods,
        noise_idx,
        signal_idx,
        labels_original=labels_original,
        first_fpr=0.01,
    )

    assert case["combined_unique_anomalies"] == [3, 4]
    assert case["union_fp"] == [2]
    assert list(case["votes_df"]["sample_idx"]) == [3, 4]

    vote_score, soft_score, detection_map = build_case_scores(
        case, all_methods, noise_idx
    )
    assert vote_score.tolist() == [0.0, 0.0, 1.0, 2.0, 2.0]
    assert soft_score.shape == (5,)
    assert detection_map["Detector-B"] == [3, 4]


def test_compare_cases_and_default_orders() -> None:
    all_methods, noise_idx, signal_idx, labels_original = _toy_scores()
    results_df = pd.DataFrame(
        [
            {"method": "Detector-A", "auc": 0.91, "tpr_1": 1.0},
            {"method": "Detector-B", "auc": 0.80, "tpr_1": 0.5},
        ]
    )

    case_orders = build_default_case_orders(
        results_df,
        all_methods,
        noise_idx,
        signal_idx,
        include_oracle=False,
    )
    assert list(case_orders) == ["Post-hoc order: top AUC"]

    case_results, comparison_df, by_name = compare_budgeted_cases(
        case_orders,
        all_methods,
        noise_idx,
        signal_idx,
        labels_original=labels_original,
    )

    assert len(case_results) == 1
    assert comparison_df.iloc[0]["recovered"] == 2
    assert by_name["Post-hoc order: top AUC"]["union_fp"] == [2]


def test_render_budgeted_case_and_greedy_alias() -> None:
    all_methods, noise_idx, signal_idx, labels_original = _toy_scores()
    order = greedy_zero_fp_order(
        "Detector-A",
        ["Detector-A", "Detector-B"],
        all_methods,
        noise_idx,
        signal_idx,
        max_methods=2,
    )
    assert order == ["Detector-A"]

    case = evaluate_budgeted_accumulation(
        order,
        "Toy case",
        all_methods,
        noise_idx,
        signal_idx,
        labels_original=labels_original,
    )
    rendered = render_budgeted_case(case, total_anomalies=len(signal_idx))
    assert "Toy case" in rendered
    assert "Recovered anomalies with detector votes:" in rendered
    assert "Unique anomalies contributed by each detector:" in rendered


def test_summarize_feature_discrimination() -> None:
    feature_matrix = np.array(
        [
            [0.0, 1.0, 2.0],
            [0.1, 1.1, 2.1],
            [0.2, 1.2, 2.2],
            [3.0, 1.0, 2.0],
            [3.1, 1.1, 2.1],
        ],
        dtype=float,
    )
    summary_df = summarize_feature_discrimination(
        feature_matrix,
        noise_idx=np.array([0, 1, 2]),
        signal_idx=np.array([3, 4]),
    )

    assert list(summary_df.columns) == [
        "feature_idx",
        "feature_name",
        "effect_size",
        "p_value",
        "significant",
    ]
    assert summary_df.iloc[0]["feature_idx"] == 0
    assert summary_df.iloc[0]["effect_size"] >= summary_df.iloc[-1]["effect_size"]
