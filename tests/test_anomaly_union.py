import numpy as np
import pytest

from ember.anomaly.crossval import background_folds, out_of_fold_scores
from ember.anomaly.narrowband import (
    NARROWBAND_FEATURE_NAMES,
    extract_feature_bank_v2,
    narrowband_statistics,
)
from ember.anomaly.union import (
    clopper_pearson,
    empirical_pvalues,
    min_p_union,
    recovery_at_far,
    threshold_at_far,
)


def test_empirical_pvalues_are_ranks():
    ref = np.arange(10, dtype=float)
    assert empirical_pvalues(ref, [9.5])[0] == pytest.approx(0.0)
    assert empirical_pvalues(ref, [-1.0])[0] == pytest.approx(1.0)
    assert empirical_pvalues(ref, [4.5])[0] == pytest.approx(0.5)


def test_threshold_at_far_admits_floor_far_n():
    rng = np.random.default_rng(0)
    bg = rng.normal(size=500)
    thr = threshold_at_far(bg, 0.01)
    assert (bg > thr).sum() == 5


def test_min_p_union_flags_if_any_member_does():
    rng = np.random.default_rng(1)
    ref = {"a": rng.normal(size=1000), "b": rng.normal(size=1000)}
    scores = {"a": np.array([0.0, 10.0]), "b": np.array([10.0, 0.0])}
    u = min_p_union(ref, scores)
    assert u[0] == pytest.approx(u[1])
    assert u[0] > min_p_union(ref, {"a": np.array([0.0]), "b": np.array([0.0])})[0]


def test_recovery_at_far_with_fold_models():
    rng = np.random.default_rng(2)
    bg = rng.normal(size=496)
    ev = np.stack([rng.normal(loc=4, size=42) for _ in range(5)])
    r = recovery_at_far(bg, ev, 0.01, event_labels=[1] * 19 + [2] * 23)
    assert r["fp"] == 4
    assert 0.9 < r["tpr"] <= 1.0
    lo, hi = r["tpr_ci"]
    assert lo <= r["tpr"] <= hi
    assert "tpr_label1" in r and "tpr_label2" in r


def test_clopper_pearson_edges():
    assert clopper_pearson(0, 10)[0] == 0.0
    assert clopper_pearson(10, 10)[1] == 1.0
    lo, hi = clopper_pearson(31, 42)
    assert 0.57 < lo < 0.60 and 0.85 < hi < 0.88


def test_narrowband_statistics_shape_and_line_sensitivity():
    rng = np.random.default_rng(3)
    quiet = rng.normal(size=(128, 200)).astype(np.float32)
    line = quiet.copy()
    line[60, 90:110] += 12.0  # a narrow line, localised in time
    q, ln = narrowband_statistics(quiet), narrowband_statistics(line)
    assert q.shape == (len(NARROWBAND_FEATURE_NAMES),)
    assert ln[1] > q[1]  # line_contrast_p99
    assert ln[3] > q[3]  # column_excess_p99
    assert extract_feature_bank_v2(quiet).shape[0] == 75 + 14


def test_out_of_fold_scores_cover_every_background_once():
    bg = np.arange(0, 50)
    ev = np.arange(50, 60)

    def fit_and_score(train_idx):
        # a "detector" whose score is the distance to the nearest training index
        all_idx = np.arange(60)
        d = (
            np.abs(all_idx[:, None] - np.asarray(train_idx)[None, :])
            .min(1)
            .astype(float)
        )
        return {"toy": d}

    oof = out_of_fold_scores(fit_and_score, bg, ev, k=5, seed=0)
    assert oof.background["toy"].shape == (50,)
    assert oof.events["toy"].shape == (5, 10)
    assert not np.isnan(oof.background["toy"]).any()
    assert all(len(te) == 10 for te in oof.test_folds)
    folds = background_folds(bg, k=5, seed=0)
    assert sorted(np.concatenate([te for _, te in folds]).tolist()) == bg.tolist()
