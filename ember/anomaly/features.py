"""Feature extraction and augmentation helpers for PSP anomaly detection."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.signal import coherence as sig_coherence
from scipy.signal import csd
from scipy.stats import kurtosis, skew


WAVELET_FEATURE_COUNT = 13


def collect_spectrograms(
    df: pd.DataFrame,
    *,
    column: str = "Log Amplitude",
) -> list[np.ndarray]:
    """Extract the raw 2D spectrogram arrays from a dataframe column."""

    return [np.asarray(value, dtype=np.float32) for value in df[column].tolist()]


def align_spectrogram_pair(
    spec_a: np.ndarray,
    spec_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Crop two spectrograms to a shared minimum shape."""

    spec_a = np.asarray(spec_a, dtype=np.float32)
    spec_b = np.asarray(spec_b, dtype=np.float32)
    min_f = min(spec_a.shape[0], spec_b.shape[0])
    min_t = min(spec_a.shape[1], spec_b.shape[1])
    return spec_a[:min_f, :min_t], spec_b[:min_f, :min_t]


def _wavelet_feature_vector(
    spec: np.ndarray,
    *,
    wavelet: str = "db4",
    max_level: int = 4,
) -> np.ndarray:
    """Return a fixed-length wavelet energy vector, padding when needed."""

    try:
        import pywt
    except ImportError:
        return np.zeros(WAVELET_FEATURE_COUNT, dtype=np.float32)

    supported_level = min(max_level, pywt.dwt_max_level(min(spec.shape), wavelet))
    if supported_level < 1:
        return np.zeros(WAVELET_FEATURE_COUNT, dtype=np.float32)

    try:
        coeffs = pywt.wavedec2(spec, wavelet, level=supported_level)
    except Exception:
        return np.zeros(WAVELET_FEATURE_COUNT, dtype=np.float32)

    energies = [np.abs(coeffs[0]).mean()]
    for level_coeffs in coeffs[1:]:
        for coeff in level_coeffs:
            energies.append(np.abs(coeff).mean())

    energies = np.asarray(energies[:WAVELET_FEATURE_COUNT], dtype=np.float64)
    if energies.size < WAVELET_FEATURE_COUNT:
        energies = np.pad(energies, (0, WAVELET_FEATURE_COUNT - energies.size))

    total = energies.sum()
    if total > 0:
        energies /= total

    return energies.astype(np.float32)


def extract_physics_features(
    spec_2d: np.ndarray,
    *,
    n_bands: int = 8,
) -> np.ndarray:
    """Extract the physics-aware handcrafted feature vector from a spectrogram."""

    spec = np.asarray(spec_2d, dtype=np.float64)
    n_freq, n_time = spec.shape
    feats: list[float] = []

    bands = np.array_split(spec, n_bands, axis=0)
    for band in bands:
        if band.size == 0:
            feats.extend([0.0, 0.0, 0.0, 0.0, 0.0])
            continue
        vals = band.ravel()
        feats.extend(
            [
                float(vals.mean()),
                float(vals.std()),
                float(kurtosis(vals)),
                float(skew(vals)),
                float(np.percentile(vals, 95) - np.percentile(vals, 5)),
            ]
        )

    temporal_mean = spec.mean(axis=0)
    temporal_std = spec.std(axis=0)
    temporal_diff = np.diff(temporal_mean) if n_time > 1 else np.zeros(1, dtype=np.float64)
    linear_trend = float(np.polyfit(np.arange(n_time), temporal_mean, 1)[0]) if n_time > 1 else 0.0
    feats.extend(
        [
            float(temporal_mean.std()),
            float(temporal_diff.std()),
            float(np.abs(temporal_diff).mean()),
            linear_trend,
            float(temporal_std.mean()),
        ]
    )

    freq_profile = spec.mean(axis=1)
    freq_profile_norm = freq_profile - freq_profile.min()
    freq_profile_norm /= freq_profile_norm.sum() + 1e-10
    spectral_entropy = -np.sum(freq_profile_norm * np.log(freq_profile_norm + 1e-10))
    freq_bins = np.arange(n_freq)
    centroid = float(np.sum(freq_bins * freq_profile_norm))
    bandwidth = float(np.sqrt(np.sum((freq_bins - centroid) ** 2 * freq_profile_norm)))
    feats.extend([float(spectral_entropy), centroid / max(n_freq, 1), bandwidth / max(n_freq, 1)])

    peak_freq_per_time = spec.argmax(axis=0).astype(float)
    peak_diff = np.diff(peak_freq_per_time) if n_time > 1 else np.zeros(1, dtype=np.float64)
    if n_time > 1:
        peak_auto = float(np.corrcoef(peak_freq_per_time[:-1], peak_freq_per_time[1:])[0, 1])
    else:
        peak_auto = 0.0
    feats.extend(
        [
            float(peak_freq_per_time.mean() / max(n_freq, 1)),
            float(peak_freq_per_time.std() / max(n_freq, 1)),
            float(peak_diff.std() / max(n_freq, 1)),
            float(np.abs(peak_diff).mean() / max(n_freq, 1)),
            peak_auto,
        ]
    )

    feats.extend(_wavelet_feature_vector(spec).tolist())

    valid_band_profiles = [band.mean(axis=0) for band in bands if band.size]
    if len(valid_band_profiles) >= 4:
        band_profiles = np.asarray(valid_band_profiles, dtype=np.float64)
        corr_matrix = np.corrcoef(band_profiles)
        upper = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
        feats.extend(
            [
                float(upper.mean()),
                float(upper.std()),
                float(upper.max()),
                float((upper > 0.7).mean()),
            ]
        )

    feats.extend(
        [
            float(spec.mean()),
            float(spec.std()),
            float(kurtosis(spec.ravel())),
            float(skew(spec.ravel())),
            float(np.percentile(spec, 99) - np.percentile(spec, 1)),
        ]
    )

    return np.nan_to_num(np.asarray(feats, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def extract_physics_feature_matrix(
    specs: Sequence[np.ndarray],
    *,
    n_bands: int = 8,
) -> np.ndarray:
    """Vectorize the physics feature extraction across many spectrograms."""

    return np.asarray([extract_physics_features(spec, n_bands=n_bands) for spec in specs], dtype=np.float32)


def extract_coupling_features(
    spec_ch1: np.ndarray,
    spec_ch2: np.ndarray,
) -> np.ndarray:
    """Extract cross-channel coupling features for a pair of spectrograms."""

    spec_ch1, spec_ch2 = align_spectrogram_pair(spec_ch1, spec_ch2)
    feats: list[float] = []

    # Sample a fixed number of frequency rows so the coupling vector length
    # stays constant even when different examples have different heights.
    row_indices = np.linspace(0, max(spec_ch1.shape[0] - 1, 0), num=4, dtype=int)
    for i in row_indices:
        t1 = spec_ch1[i]
        t2 = spec_ch2[i]
        if t1.std() > 1e-8 and t2.std() > 1e-8:
            corr = float(np.corrcoef(t1, t2)[0, 1])
        else:
            corr = 0.0
        feats.append(corr)

    flat_ch1 = spec_ch1.ravel()
    flat_ch2 = spec_ch2.ravel()
    nperseg = max(2, min(256, max(2, len(flat_ch1) // 4)))

    try:
        _, coherence = sig_coherence(flat_ch1, flat_ch2, nperseg=nperseg)
        feats.extend(
            [
                float(coherence.mean()),
                float(coherence.max()),
                float(coherence.std()),
                float((coherence > 0.5).mean()),
            ]
        )
    except Exception:
        feats.extend([0.0, 0.0, 0.0, 0.0])

    try:
        _, cross_spec = csd(flat_ch1, flat_ch2, nperseg=nperseg)
        phase_diff = np.angle(cross_spec)
        feats.extend([float(np.abs(phase_diff).mean()), float(phase_diff.std())])
    except Exception:
        feats.extend([0.0, 0.0])

    return np.nan_to_num(np.asarray(feats, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def extract_coupling_feature_matrix(
    primary_specs: Sequence[np.ndarray],
    secondary_specs: Sequence[np.ndarray],
) -> np.ndarray:
    """Vectorize the coupling feature extraction across spectrogram pairs."""

    if len(primary_specs) != len(secondary_specs):
        raise ValueError("primary_specs and secondary_specs must have the same length.")

    return np.asarray(
        [
            extract_coupling_features(spec_a, spec_b)
            for spec_a, spec_b in zip(primary_specs, secondary_specs)
        ],
        dtype=np.float32,
    )


def choose_coupling_channel(
    df: pd.DataFrame,
    *,
    reference_specs: Sequence[np.ndarray] | None = None,
    channel_columns: Sequence[str] | None = None,
    label_column: str = "Label",
    spec_column: str = "Log Amplitude",
) -> str | None:
    """Pick the first valid numeric 2D auxiliary channel for coupling features."""

    if df.empty:
        return None

    if reference_specs is None:
        reference_specs = collect_spectrograms(df, column=spec_column)

    ref_shape = reference_specs[0].shape
    if channel_columns is None:
        channel_columns = [col for col in df.columns if col not in {label_column, spec_column}]

    for column in channel_columns:
        try:
            sample_arr = np.asarray(df.iloc[0][column], dtype=np.float32)
        except (TypeError, ValueError):
            continue

        if sample_arr.ndim == 2 and sample_arr.shape == ref_shape:
            return column

    return None


class PhysicsAugmenter:
    """Physics-constrained augmentations for PSP spectrograms."""

    def __init__(self, seed: int = 42):
        self.seed = int(seed)

    @staticmethod
    def _crop_to_shape(spec: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
        return np.asarray(spec, dtype=np.float32)[: target_shape[0], : target_shape[1]]

    def augment_one(
        self,
        spec: np.ndarray,
        rng: np.random.Generator,
        *,
        all_noise_specs: Sequence[np.ndarray] | None = None,
    ) -> np.ndarray:
        """Apply one randomized augmentation chain to a single spectrogram."""

        augmented = np.asarray(spec, dtype=np.float32).copy()

        if rng.random() < 0.8:
            augmented = augmented * rng.uniform(0.85, 1.15)

        if rng.random() < 0.7:
            noise_level = rng.uniform(0.02, 0.08) * max(float(augmented.std()), 1e-6)
            augmented = augmented + rng.normal(0, noise_level, augmented.shape).astype(np.float32)

        if rng.random() < 0.5:
            augmented = np.roll(augmented, rng.integers(-8, 9), axis=0)

        if rng.random() < 0.4:
            augmented = np.roll(augmented, rng.integers(-10, 11), axis=-1)

        if rng.random() < 0.3:
            augmented = gaussian_filter(augmented, sigma=rng.uniform(0.3, 1.0))

        if all_noise_specs is not None and rng.random() < 0.25:
            partner = np.asarray(all_noise_specs[rng.integers(len(all_noise_specs))], dtype=np.float32)
            augmented, partner = align_spectrogram_pair(augmented, partner)
            lam = rng.beta(0.5, 0.5)
            augmented = lam * augmented + (1.0 - lam) * partner

        return np.asarray(augmented, dtype=np.float32)

    def generate_all(
        self,
        noise_specs: Sequence[np.ndarray],
        noise_indices: Sequence[int],
        *,
        n_per_sample: int = 15,
        n_bands: int = 8,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate augmented spectrograms and their physics features."""

        if len(noise_specs) != len(noise_indices):
            raise ValueError("noise_specs and noise_indices must have the same length.")
        if len(noise_specs) == 0:
            raise ValueError("noise_specs must not be empty.")

        min_f = min(np.asarray(spec).shape[0] for spec in noise_specs)
        min_t = min(np.asarray(spec).shape[1] for spec in noise_specs)
        common_shape = (min_f, min_t)

        aligned_noise_specs = [
            self._crop_to_shape(np.asarray(spec, dtype=np.float32), common_shape)
            for spec in noise_specs
        ]

        augmented_specs: list[np.ndarray] = []
        sources: list[int] = []
        augmented_features: list[np.ndarray] = []

        for i, (spec, orig_idx) in enumerate(zip(aligned_noise_specs, noise_indices)):
            for j in range(n_per_sample):
                rng = np.random.default_rng(hash((i, j, self.seed)) % (2**32))
                augmented = self._crop_to_shape(
                    self.augment_one(spec, rng, all_noise_specs=aligned_noise_specs),
                    common_shape,
                )
                augmented_specs.append(augmented)
                sources.append(int(orig_idx))
                augmented_features.append(extract_physics_features(augmented, n_bands=n_bands))

        return (
            np.stack(augmented_specs),
            np.asarray(sources, dtype=int),
            np.asarray(augmented_features, dtype=np.float32),
        )
