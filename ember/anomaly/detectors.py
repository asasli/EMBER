"""Production-oriented detector suite for background-only anomaly detection.

These detectors are designed for *deployment*: train on background only,
calibrate thresholds on held-out background, then score new data.
They complement the LOO-CV detectors in ``classical.py``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.ndimage import uniform_filter, zoom


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _missing_sklearn() -> ImportError:
    return ImportError(
        "EMBER anomaly detectors require scikit-learn. "
        "Install with `pip install -e .[anomaly]`."
    )


def _robust_patch_normalize(spec: np.ndarray) -> np.ndarray:
    """Robust z-score using median / IQR (not mean / std)."""
    s = np.asarray(spec, dtype=np.float64)
    center = float(np.nanmedian(s))
    q25, q75 = np.nanpercentile(s, [25, 75])
    scale = max(float(q75 - q25), 1e-6)
    return (s - center) / scale


def _resize_spec(
    spec: np.ndarray, target_h: int, target_w: int
) -> np.ndarray:
    """Bilinear resize to ``(target_h, target_w)``."""
    s = np.asarray(spec, dtype=np.float64)
    zf = (target_h / max(1, s.shape[0]), target_w / max(1, s.shape[1]))
    return zoom(s, zf, order=1)


# ---------------------------------------------------------------------------
# LocalPatchDetector
# ---------------------------------------------------------------------------

class LocalPatchDetector:
    """Local patch z-score detector (matches the batch5 notebook).

    For each spectrogram:

    1. Resize to ``(target_h, target_w)`` via bilinear interpolation.
    2. Robust-normalise (median / IQR centering).
    3. Compute per-pixel z-score against the pixel-wise mean / std
       estimated from the background training set.
    4. Apply a ``(win_h, win_w)`` uniform smoothing filter to positive
       excursions only (``max(z, 0)``).
    5. Score = ``score_q * quantile(sm, q) + (1 - score_q) * max(sm)``.

    Parameters
    ----------
    win_h, win_w : int
        Smoothing window size (pixels after resize).
    target_h, target_w : int or None
        Resize target (``None`` keeps original shape).
    q : float
        High quantile for the score combination (default 0.995).
    score_q : float
        Weight of the quantile term (default 0.7).
    """

    def __init__(
        self,
        win_h: int = 12,
        win_w: int = 24,
        target_h: int | None = 128,
        target_w: int | None = 256,
        q: float = 0.995,
        score_q: float = 0.7,
    ):
        self.win_h = win_h
        self.win_w = win_w
        self.target_h = target_h
        self.target_w = target_w
        self.q = q
        self.score_q = score_q
        self._mean_map: np.ndarray | None = None
        self._std_map: np.ndarray | None = None

    def _prepare(self, spec: np.ndarray) -> np.ndarray:
        s = _robust_patch_normalize(spec)
        if self.target_h is not None and self.target_w is not None:
            s = _resize_spec(s, self.target_h, self.target_w)
        return s

    def fit(self, specs: Sequence[np.ndarray]) -> "LocalPatchDetector":
        """Estimate pixel-wise mean / std from background training data."""
        stack = np.stack([self._prepare(s) for s in specs])
        self._mean_map = stack.mean(axis=0)
        self._std_map = np.maximum(stack.std(axis=0), 1e-3)
        return self

    def score(self, specs: Sequence[np.ndarray]) -> np.ndarray:
        """Return weighted quantile / max score for each spectrogram."""
        if self._mean_map is None:
            raise RuntimeError("Call fit() before score().")
        out = []
        for s in specs:
            z = (self._prepare(s) - self._mean_map) / self._std_map
            sm = uniform_filter(
                np.maximum(z, 0.0),
                size=(self.win_h, self.win_w),
                mode="nearest",
            )
            q_val = float(np.quantile(sm, self.q))
            sc = self.score_q * q_val + (1 - self.score_q) * float(sm.max())
            out.append(sc)
        return np.asarray(out, dtype=np.float64)


# ---------------------------------------------------------------------------
# BandDeviationDetector
# ---------------------------------------------------------------------------

class BandDeviationDetector:
    """Per-band temporal deviation detector.

    Divides the frequency axis into ``n_bands`` equal strips and computes
    the temporal mean profile for each strip.  Score = max over bands of
    the mean absolute z-score deviation (matches the batch5 notebook).

    Parameters
    ----------
    n_bands : int
        Number of frequency bands.
    """

    def __init__(self, n_bands: int = 8):
        self.n_bands = n_bands
        self._band_mean: np.ndarray | None = None
        self._band_std: np.ndarray | None = None

    def _band_profiles(
        self, specs: Sequence[np.ndarray]
    ) -> np.ndarray:
        """Return (N, n_bands, T_min) temporal profiles per band."""
        T_min = min(np.asarray(s).shape[1] for s in specs)
        profiles = []
        for s in specs:
            arr = np.asarray(s, dtype=np.float32)
            H = arr.shape[0]
            edges = np.linspace(0, H, self.n_bands + 1, dtype=int)
            row = np.array([
                arr[edges[b]:edges[b + 1], :T_min].mean(axis=0)
                for b in range(self.n_bands)
            ])
            profiles.append(row)
        return np.array(profiles)  # (N, n_bands, T_min)

    def fit(
        self, specs: Sequence[np.ndarray]
    ) -> "BandDeviationDetector":
        """Estimate per-band mean and std from background training data."""
        profiles = self._band_profiles(specs)  # (N, n_bands, T)
        self._band_mean = profiles.mean(axis=0)  # (n_bands, T)
        self._band_std = profiles.std(axis=0) + 1e-9  # (n_bands, T)
        return self

    def score(self, specs: Sequence[np.ndarray]) -> np.ndarray:
        """Return max-band mean-|z| score for each spectrogram."""
        if self._band_mean is None:
            raise RuntimeError("Call fit() before score().")
        profiles = self._band_profiles(specs)  # (N, n_bands, T)
        T = min(profiles.shape[2], self._band_mean.shape[1])
        p = profiles[:, :, :T]
        m = self._band_mean[:, :T]
        sd = self._band_std[:, :T]
        # mean |z| per band → (N, n_bands)
        band_scores = np.abs((p - m) / sd).mean(axis=2)
        return band_scores.max(axis=1)  # worst band per sample


# ---------------------------------------------------------------------------
# Detector suite
# ---------------------------------------------------------------------------

# Default LP configs match the batch5 notebook:
#   LP_Small: resize to 128x256, sliding 12x24 window
#   LP_Micro: resize to 32x64,   sliding 3x6  window
_DEFAULT_LP_CONFIGS: dict[str, dict] = {
    "LP_Small": {
        "win_h": 12, "win_w": 24, "target_h": 128, "target_w": 256,
    },
    "LP_Micro": {
        "win_h": 3, "win_w": 6, "target_h": 32, "target_w": 64,
    },
}


def fit_detector_suite(
    bg_train_specs: Sequence[np.ndarray],
    bg_train_features: np.ndarray,
    *,
    seed: int = 42,
    support_fraction: float = 0.90,
    contamination: float = 0.02,
    n_pca: int = 30,
    lp_configs: dict | None = None,
    n_bands: int = 8,
) -> dict:
    """Fit the full robust detector suite on background-only training data.

    Parameters
    ----------
    bg_train_specs : list of (H, W) arrays
        Background training spectrograms.
    bg_train_features : (N, D) array
        Physics feature matrix for ``bg_train_specs``.
    seed : int
        Random seed for reproducible fits.
    support_fraction : float
        ``MinCovDet`` support fraction (0.90 keeps 90 % of samples).
    contamination : float
        ``IsolationForest`` contamination estimate.
    n_pca : int
        PCA components for Mahal / IF / LOF / KNN.
    lp_configs : dict, optional
        ``{name: {win_h, win_w, target_h, target_w}}`` for local-patch
        detectors.  Defaults to LP_Small and LP_Micro (batch5 settings).
    n_bands : int
        Number of frequency bands for ``BandDeviationDetector``.

    Returns
    -------
    dict with keys:
        ``scaler``       – fitted ``StandardScaler``
        ``pca``          – fitted ``PCA`` (for Mahal / IF / LOF / KNN)
        ``pca_recon``    – fitted ``PCA`` (for reconstruction error)
        ``detectors``    – ``{name: fitted_estimator}``
        ``lp_detectors`` – ``{name: LocalPatchDetector}``
        ``band_dev``     – fitted ``BandDeviationDetector``
    """
    try:
        from sklearn.covariance import MinCovDet
        from sklearn.decomposition import PCA
        from sklearn.ensemble import IsolationForest
        from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise _missing_sklearn() from exc

    X = np.asarray(bg_train_features, dtype=np.float32)
    N = len(X)

    # ── Feature preprocessing ────────────────────────────────────────────
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)

    n_comp = min(n_pca, N - 1, Xs.shape[1])
    pca = PCA(n_components=n_comp, random_state=seed).fit(Xs)
    Xp = pca.transform(Xs)

    # ── Classical detectors on PCA-reduced features ──────────────────────
    sf = min(support_fraction, (N - n_comp - 1) / N)
    sf = max(sf, 0.6)

    try:
        mahal = MinCovDet(
            support_fraction=sf, random_state=seed
        ).fit(Xp)
    except Exception:
        from sklearn.covariance import EmpiricalCovariance
        mahal = EmpiricalCovariance().fit(Xp)

    iforest = IsolationForest(
        contamination=contamination, random_state=seed, n_estimators=500
    )
    iforest.fit(Xp)

    n_lof = min(35, max(2, N - 1))
    lof = LocalOutlierFactor(
        n_neighbors=n_lof,
        novelty=True,
        contamination=contamination,
    )
    lof.fit(Xp)

    k_knn = min(10, max(1, N - 1))
    knn = NearestNeighbors(n_neighbors=k_knn).fit(Xp)

    # PCA reconstruction error (separate PCA, more components)
    n_recon = min(N - 1, Xs.shape[1], 30)
    pca_recon = PCA(n_components=n_recon, random_state=seed).fit(Xs)

    detectors = {
        "RobustMahal": mahal,
        "IForest": iforest,
        "LOF": lof,
        "KNNDist": knn,
        "PCARecon": pca_recon,
    }

    # ── Local patch detectors ────────────────────────────────────────────
    if lp_configs is None:
        lp_configs = _DEFAULT_LP_CONFIGS

    lp_detectors: dict[str, LocalPatchDetector] = {}
    for name, cfg in lp_configs.items():
        lpd = LocalPatchDetector(
            win_h=cfg["win_h"],
            win_w=cfg["win_w"],
            target_h=cfg.get("target_h"),
            target_w=cfg.get("target_w"),
        )
        lpd.fit(bg_train_specs)
        lp_detectors[name] = lpd

    # ── Band deviation detector ──────────────────────────────────────────
    band_dev = BandDeviationDetector(n_bands=n_bands)
    band_dev.fit(bg_train_specs)

    return {
        "scaler": scaler,
        "pca": pca,
        "pca_recon": pca_recon,
        "detectors": detectors,
        "lp_detectors": lp_detectors,
        "band_dev": band_dev,
    }


def score_detector_suite(
    suite: dict,
    specs: Sequence[np.ndarray],
    features: np.ndarray,
) -> dict[str, np.ndarray]:
    """Score spectrograms with the fitted detector suite.

    Parameters
    ----------
    suite : dict
        Output of :func:`fit_detector_suite`.
    specs : list of (H, W) arrays
        Spectrograms to score.
    features : (N, D) array
        Physics features for ``specs``.

    Returns
    -------
    dict mapping detector name → anomaly score array (N,).
    Higher = more anomalous.
    """
    X = np.asarray(features, dtype=np.float32)
    Xs = suite["scaler"].transform(X)
    Xp = suite["pca"].transform(Xs)

    scores: dict[str, np.ndarray] = {}
    det = suite["detectors"]

    scores["RobustMahal"] = det["RobustMahal"].mahalanobis(Xp)
    scores["IForest"] = -det["IForest"].score_samples(Xp)
    scores["LOF"] = -det["LOF"].score_samples(Xp)

    dists, _ = det["KNNDist"].kneighbors(Xp)
    scores["KNNDist"] = dists.mean(axis=1)

    Xs_hat = suite["pca_recon"].inverse_transform(
        suite["pca_recon"].transform(Xs)
    )
    scores["PCARecon"] = np.mean((Xs - Xs_hat) ** 2, axis=1)

    for name, lpd in suite["lp_detectors"].items():
        scores[name] = lpd.score(specs)

    scores["BandDeviation"] = suite["band_dev"].score(specs)

    return scores


def make_recon_gate(
    pca_recon_norm: np.ndarray,
    vae_elbo_norm: np.ndarray,
) -> np.ndarray:
    """Geometric mean of PCARecon and VAE-ELBO rank-normalised scores.

    Suppresses false positives that score high on Mahalanobis but low on
    both reconstruction-based detectors.

    Parameters
    ----------
    pca_recon_norm, vae_elbo_norm : array of float in [0, 1]
        Rank-normalised scores from PCARecon and VAE-ELBO detectors.

    Returns
    -------
    np.ndarray
        Element-wise ``sqrt(pca_recon_norm * vae_elbo_norm)``.
    """
    return np.sqrt(
        np.asarray(pca_recon_norm, dtype=np.float64)
        * np.asarray(vae_elbo_norm, dtype=np.float64)
    )
