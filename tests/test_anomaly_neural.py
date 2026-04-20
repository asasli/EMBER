from __future__ import annotations

import numpy as np
import pytest

from ember.anomaly import neural


def test_score_with_ae_loo_excludes_held_out_noise(monkeypatch) -> None:
    train_calls: list[tuple[int, ...]] = []

    def fake_train_autoencoder(noise_specs_np, **kwargs):
        ids = tuple(int(np.asarray(spec)[0, 0]) for spec in noise_specs_np)
        train_calls.append(ids)
        return ids, None, None

    def fake_score_with_ae(ae, all_specs_np, mu, std, **kwargs):
        scores = np.asarray([float(sum(ae))] * len(all_specs_np), dtype=np.float32)
        latents = np.asarray([[float(len(ae))]] * len(all_specs_np), dtype=np.float32)
        return scores, latents

    monkeypatch.setattr(neural, "train_autoencoder", fake_train_autoencoder)
    monkeypatch.setattr(neural, "score_with_ae", fake_score_with_ae)

    specs = [np.full((2, 2), idx, dtype=np.float32) for idx in range(4)]
    scores, latents = neural.score_with_ae_loo(
        specs,
        noise_idx=np.array([0, 1, 2], dtype=int),
        epochs=1,
        verbose=False,
    )

    assert scores.shape == (4,)
    assert latents.shape == (4, 1)
    assert train_calls == [
        (0, 1, 2),
        (1, 2),
        (0, 2),
        (0, 1),
    ]


def test_train_autoencoder_accepts_validation_and_early_stopping() -> None:
    rng = np.random.default_rng(0)
    specs = [rng.normal(size=(8, 8)).astype(np.float32) for _ in range(6)]

    ae, mu, std = neural.train_autoencoder(
        specs,
        epochs=2,
        batch_size=2,
        latent_dim=4,
        h=8,
        w=8,
        validation_fraction=0.25,
        early_stopping_patience=1,
        early_stopping_min_delta=1e-4,
        seed=0,
        verbose=False,
    )

    assert isinstance(ae, neural.SpectrogramAE)
    assert mu.shape == ()
    assert std.shape == ()


def test_train_autoencoder_rejects_invalid_early_stopping_args() -> None:
    specs = [np.zeros((4, 4), dtype=np.float32) for _ in range(3)]

    with pytest.raises(ValueError, match="validation_fraction"):
        neural.train_autoencoder(specs, epochs=1, validation_fraction=1.0, verbose=False)

    with pytest.raises(ValueError, match="early_stopping_patience"):
        neural.train_autoencoder(specs, epochs=1, early_stopping_patience=0, verbose=False)
