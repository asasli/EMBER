"""Narrowband-line statistics and the v2 feature bank.

Ion acoustic waves appear in a log-log DBM spectrogram as lines that are narrow
in frequency and localised in time.  After removing each burst's own time-median
spectrum, a handful of per-frame statistics of the residual capture how "line-like"
the most unusual frames of the burst are.  All statistics are label-free and
training-free; they are appended to the spectral-morphology features of
:mod:`ember.anomaly.features` to form the 89-dimensional v2 feature bank used by
the EMBER-U operating detector.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.ndimage import uniform_filter1d

from ember.anomaly.features import extract_physics_features

NARROWBAND_FEATURE_NAMES = (
    "line_contrast_p90",
    "line_contrast_p99",
    "column_excess_p90",
    "column_excess_p99",
    "active_frac_z6",
    "active_frac_z8",
    "smoothed_column_excess_p99",
    "temporal_std_p99",
    "temporal_std_max",
    "resid_band_power_p99_b0",
    "resid_band_power_p99_b1",
    "resid_band_power_p99_b2",
    "resid_band_power_p99_b3",
    "global_std",
)


def residual_spectrogram(spec: np.ndarray) -> np.ndarray:
    """Subtract the time-median spectrum of the burst from every frame."""

    spec = np.asarray(spec, dtype=np.float32)
    return spec - np.median(spec, axis=1, keepdims=True)


def narrowband_statistics(
    spec: np.ndarray, *, n_bands: int = 4, smooth_frames: int = 5
) -> np.ndarray:
    """Return the 14 narrowband-line statistics of one (F, T) log-power spectrogram."""

    resid = residual_spectrogram(spec)
    mx = resid.max(axis=0)
    med = np.median(resid, axis=0)
    mean = resid.mean(axis=0)
    std = resid.std(axis=0) + 1e-6
    contrast = mx - med
    excess = (mx - mean) / std

    n_freq = resid.shape[0]
    edges = np.linspace(0, n_freq, n_bands + 1, dtype=int)
    band_power = [
        float(np.percentile(resid[edges[b] : edges[b + 1]].mean(axis=0), 99))
        for b in range(n_bands)
    ]

    smoothed = uniform_filter1d(resid, smooth_frames, axis=1)
    smoothed_excess = (smoothed.max(axis=0) - smoothed.mean(axis=0)) / (
        smoothed.std(axis=0) + 1e-6
    )
    temporal_std = resid.std(axis=1)

    values = [
        np.percentile(contrast, 90),
        np.percentile(contrast, 99),
        np.percentile(excess, 90),
        np.percentile(excess, 99),
        (excess > 6).mean(),
        (excess > 8).mean(),
        np.percentile(smoothed_excess, 99),
        np.percentile(temporal_std, 99),
        temporal_std.max(),
        *band_power,
        np.asarray(spec, dtype=np.float32).std(),
    ]
    return np.nan_to_num(np.asarray(values, dtype=np.float32))


def extract_feature_bank_v2(spec: np.ndarray) -> np.ndarray:
    """Spectral-morphology features (75) followed by narrowband statistics (14)."""

    return np.concatenate([extract_physics_features(spec), narrowband_statistics(spec)])


def extract_feature_bank_v2_matrix(specs: Sequence[np.ndarray]) -> np.ndarray:
    """Vectorise :func:`extract_feature_bank_v2` over many spectrograms -> (N, 89)."""

    return np.nan_to_num(
        np.asarray([extract_feature_bank_v2(s) for s in specs], dtype=np.float32)
    )
