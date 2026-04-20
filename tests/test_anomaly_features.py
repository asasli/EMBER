from __future__ import annotations

import numpy as np
import pandas as pd

from ember.anomaly import (
    PhysicsAugmenter,
    build_feature_bank,
    extract_coupling_feature_matrix,
    extract_coupling_features,
    extract_physics_features,
    prepare_multimodel_spectrograms,
    prepare_anomaly_dataset,
)


def _make_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Label": [0, 1, 2],
            "Log Amplitude": [
                np.arange(20, dtype=np.float32).reshape(4, 5),
                np.linspace(0, 1, 20, dtype=np.float32).reshape(4, 5),
                np.ones((4, 5), dtype=np.float32),
            ],
            "Aux Channel": [
                np.arange(20, dtype=np.float32).reshape(4, 5) * 0.5,
                np.linspace(1, 2, 20, dtype=np.float32).reshape(4, 5),
                np.full((4, 5), 2.0, dtype=np.float32),
            ],
        }
    )


def test_prepare_dataset_and_feature_bank() -> None:
    df = _make_dataframe()
    dataset = prepare_anomaly_dataset(df)

    assert len(dataset.specs_raw) == 3
    assert dataset.noise_idx.tolist() == [0]
    assert dataset.signal_idx.tolist() == [1, 2]

    feature_bank = build_feature_bank(dataset)
    physics = feature_bank["physics_features"]
    all_features = feature_bank["all_features"]

    assert feature_bank["coupling_column"] == "Aux Channel"
    assert physics.shape == (3, 75)
    assert all_features.shape == (3, 85)
    assert np.isfinite(all_features).all()


def test_feature_extractors_and_augmenter() -> None:
    spec = np.arange(20, dtype=np.float32).reshape(4, 5)
    aux = np.flipud(spec)

    physics = extract_physics_features(spec)
    coupling = extract_coupling_features(spec, aux)

    assert physics.shape == (75,)
    assert coupling.shape == (10,)
    assert np.isfinite(physics).all()
    assert np.isfinite(coupling).all()

    augmenter = PhysicsAugmenter(seed=7)
    aug_specs, aug_sources, aug_features = augmenter.generate_all([spec, aux], [10, 11], n_per_sample=2)

    assert aug_specs.shape == (4, 4, 5)
    assert aug_sources.tolist() == [10, 10, 11, 11]
    assert aug_features.shape == (4, 75)


def test_coupling_feature_matrix_handles_mixed_heights() -> None:
    specs_a = [
        np.zeros((4, 5), dtype=np.float32),
        np.ones((5, 5), dtype=np.float32),
        np.full((9, 5), 2.0, dtype=np.float32),
    ]
    specs_b = [
        np.zeros((4, 5), dtype=np.float32),
        np.ones((5, 5), dtype=np.float32) * 3.0,
        np.full((9, 5), 4.0, dtype=np.float32),
    ]

    coupling = extract_coupling_feature_matrix(specs_a, specs_b)
    assert coupling.shape == (3, 10)
    assert np.isfinite(coupling).all()


def test_prepare_multimodel_spectrograms() -> None:
    df = _make_dataframe()
    dataset = prepare_anomaly_dataset(df)
    prepared = prepare_multimodel_spectrograms(dataset.specs_raw, noise_indices=dataset.noise_idx, h=16, w=12)

    assert prepared.specs.shape == (3, 16, 12)
    assert prepared.target_shape == (16, 12)
    assert np.isfinite(prepared.specs).all()
    assert prepared.noise_std > 0
