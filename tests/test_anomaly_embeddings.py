from __future__ import annotations

import numpy as np
import pytest

from ember.anomaly import (
    analyze_class,
    analyze_fpr,
    compute_default_ensembles,
    run_efficient_loo_cv,
)


def test_notebook_signature_compatibility() -> None:
    scores = np.array([0.1, 0.2, 0.3, 0.9, 1.0], dtype=float)
    labels_binary = np.array([0, 0, 0, 1, 1], dtype=int)
    labels_original = np.array([0, 0, 0, 1, 2], dtype=int)
    noise_idx = np.array([0, 1, 2], dtype=int)
    signal_idx = np.array([3, 4], dtype=int)
    class1_idx = np.array([3], dtype=int)
    class2_idx = np.array([4], dtype=int)

    with_labels = analyze_fpr(scores, labels_binary, noise_idx, signal_idx)
    without_labels = analyze_fpr(scores, noise_idx, signal_idx)
    assert with_labels == without_labels

    with_orig = analyze_class(scores, labels_original, noise_idx, class1_idx, class2_idx)
    without_orig = analyze_class(scores, noise_idx, class1_idx, class2_idx)
    assert with_orig["tpr_c1"] == without_orig["tpr_c1"]
    assert with_orig["tpr_c2"] == without_orig["tpr_c2"]


def test_run_efficient_loo_cv_smoke() -> None:
    pytest.importorskip("sklearn")

    rng = np.random.default_rng(7)
    labels_binary = np.array([0, 0, 0, 1, 1], dtype=int)
    noise_idx = np.where(labels_binary == 0)[0]
    aug_sources = np.repeat(noise_idx, 2)

    orig = np.vstack(
        [
            rng.normal(0.0, 0.1, size=(3, 6)),
            rng.normal(1.5, 0.1, size=(2, 6)),
        ]
    ).astype(np.float32)
    aug = rng.normal(0.0, 0.1, size=(len(aug_sources), 6)).astype(np.float32)

    scores = run_efficient_loo_cv(
        {"resnet18": orig},
        {"resnet18": aug},
        aug_sources,
        labels_binary,
        noise_idx,
        embeddings=["resnet18"],
        detectors_to_use=["IForest", "Mahal"],
        n_pca=3,
    )

    assert sorted(scores) == ["resnet18_IForest", "resnet18_Mahal"]
    for values in scores.values():
        assert values.shape == (5,)
        assert np.isfinite(values).all()


def test_compute_default_ensembles_returns_strict_topk_summary() -> None:
    pytest.importorskip("sklearn")

    labels = np.array([0, 0, 0, 1, 1], dtype=int)
    scores = {
        "A": np.array([0.05, 0.10, 0.11, 0.90, 0.95], dtype=float),
        "B": np.array([0.02, 0.03, 0.04, 0.50, 0.60], dtype=float),
        "C": np.array([0.20, 0.21, 0.22, 0.30, 0.31], dtype=float),
    }

    ensembles, topk_info = compute_default_ensembles(scores, labels, topk_values=(2,))

    assert {"Ens-Rank", "Ens-Mean", "Ens-Weighted", "Ens-Top2"} <= set(ensembles)
    assert ensembles["Ens-Top2"].shape == labels.shape
    assert 2 in topk_info
    assert topk_info[2]["mode"] == "loo"
    assert len(topk_info[2]["selected_per_sample"]) == len(labels)
    assert sum(topk_info[2]["selection_counts"].values()) == len(labels) * 2
