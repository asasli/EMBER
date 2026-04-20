"""Classical anomaly detectors and probing helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


def _missing_sklearn() -> ImportError:
    return ImportError(
        "EMBER anomaly classical methods require scikit-learn. "
        "Install the anomaly extras with `pip install -e .[anomaly]`."
    )


class MahalanobisDetector:
    """Robust Mahalanobis detector using a robust covariance estimate when possible."""

    def fit(self, X: np.ndarray):
        try:
            from sklearn.covariance import EmpiricalCovariance, MinCovDet
        except ImportError as exc:
            raise _missing_sklearn() from exc

        try:
            self.cov = MinCovDet(random_state=42, support_fraction=0.8).fit(X)
        except Exception:
            self.cov = EmpiricalCovariance().fit(X)
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        return -self.cov.mahalanobis(X)


def _build_embedding_detectors(
    *,
    n_neighbors: int,
    seed: int,
) -> dict[str, object]:
    try:
        from sklearn.ensemble import IsolationForest
        from sklearn.neighbors import LocalOutlierFactor
        from sklearn.svm import OneClassSVM
    except ImportError as exc:
        raise _missing_sklearn() from exc

    return {
        "OCSVM": OneClassSVM(nu=0.1, kernel="rbf", gamma="scale"),
        "IForest": IsolationForest(n_estimators=200, contamination=0.1, random_state=seed),
        "LOF": LocalOutlierFactor(
            n_neighbors=n_neighbors,
            novelty=True,
            contamination=0.1,
        ),
        "Mahal": MahalanobisDetector(),
    }


def run_physics_loo_cv(
    physics_feats_orig: np.ndarray,
    physics_feats_aug: np.ndarray,
    aug_sources: np.ndarray,
    ae_scores_orig: np.ndarray,
    ae_latents_orig: np.ndarray | None,
    labels_binary: np.ndarray,
    noise_idx: np.ndarray,
    *,
    use_pca: bool = True,
    n_pca: int = 30,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Run the notebook-style LOO-CV detector suite on feature vectors."""

    try:
        from sklearn.decomposition import PCA
        from sklearn.ensemble import IsolationForest, RandomForestClassifier
        from sklearn.neighbors import LocalOutlierFactor
        from sklearn.preprocessing import RobustScaler
        from sklearn.svm import OneClassSVM, SVC
    except ImportError as exc:
        raise _missing_sklearn() from exc

    del ae_latents_orig

    n_samples = len(labels_binary)
    scores_oc = {key: np.zeros(n_samples, dtype=float) for key in ["OCSVM", "IForest", "LOF", "Mahal"]}
    scores_ss = {"SVM_sup": np.zeros(n_samples, dtype=float), "RF_sup": np.zeros(n_samples, dtype=float)}

    for test_i in range(n_samples):
        train_noise_mask = np.asarray([idx != test_i for idx in noise_idx], dtype=bool)
        train_noise_idx = noise_idx[train_noise_mask]
        aug_mask = aug_sources != test_i

        x_noise_orig = physics_feats_orig[train_noise_idx]
        x_noise_aug = physics_feats_aug[aug_mask]
        x_train_oc = np.vstack([x_noise_orig, x_noise_aug])
        x_test = physics_feats_orig[test_i : test_i + 1]

        scaler = RobustScaler()
        x_train_s = scaler.fit_transform(x_train_oc)
        x_test_s = scaler.transform(x_test)

        if use_pca:
            n_comp = min(n_pca, x_train_s.shape[0] - 1, x_train_s.shape[1])
            if n_comp >= 1:
                pca = PCA(n_components=n_comp, random_state=seed)
                x_train_p = pca.fit_transform(x_train_s)
                x_test_p = pca.transform(x_test_s)
            else:
                x_train_p = x_train_s
                x_test_p = x_test_s
        else:
            x_train_p = x_train_s
            x_test_p = x_test_s

        max_neighbors = max(1, len(x_train_p) - 1)
        n_neighbors = min(15, max_neighbors, max(1, len(x_train_p) // 3))
        detectors = {
            "OCSVM": OneClassSVM(nu=0.08, kernel="rbf", gamma="scale"),
            "IForest": IsolationForest(n_estimators=300, contamination=0.1, random_state=seed),
            "LOF": LocalOutlierFactor(
                n_neighbors=n_neighbors,
                novelty=True,
                contamination=0.1,
            ),
            "Mahal": MahalanobisDetector(),
        }

        for name, detector in detectors.items():
            detector.fit(x_train_p)
            scores_oc[name][test_i] = float(-detector.score_samples(x_test_p)[0])

        train_all_idx = np.asarray([j for j in range(n_samples) if j != test_i], dtype=int)
        x_sup_raw = physics_feats_orig[train_all_idx]
        y_sup = labels_binary[train_all_idx]

        if len(np.unique(y_sup)) >= 2:
            scaler_sup = RobustScaler()
            x_sup_s = scaler_sup.fit_transform(x_sup_raw)
            x_test_sup_s = scaler_sup.transform(x_test)

            if use_pca:
                n_comp_sup = min(n_pca, x_sup_s.shape[0] - 1, x_sup_s.shape[1])
                if n_comp_sup >= 1:
                    pca_sup = PCA(n_components=n_comp_sup, random_state=seed)
                    x_sup_p = pca_sup.fit_transform(x_sup_s)
                    x_test_sup_p = pca_sup.transform(x_test_sup_s)
                else:
                    x_sup_p = x_sup_s
                    x_test_sup_p = x_test_sup_s
            else:
                x_sup_p = x_sup_s
                x_test_sup_p = x_test_sup_s

            svm = SVC(
                kernel="rbf",
                probability=True,
                class_weight="balanced",
                C=10,
                gamma="scale",
                random_state=seed,
            )
            svm.fit(x_sup_p, y_sup)
            scores_ss["SVM_sup"][test_i] = float(svm.predict_proba(x_test_sup_p)[0, 1])

            rf = RandomForestClassifier(
                n_estimators=200,
                class_weight="balanced",
                max_features="sqrt",
                random_state=seed,
                # Keep the strict workflow safe in constrained environments where
                # joblib thread pools can fail to start.
                n_jobs=1,
            )
            rf.fit(x_sup_p, y_sup)
            scores_ss["RF_sup"][test_i] = float(rf.predict_proba(x_test_sup_p)[0, 1])
        else:
            scores_ss["SVM_sup"][test_i] = 0.5
            scores_ss["RF_sup"][test_i] = 0.5

    return {**scores_oc, **scores_ss, "AE_recon": np.asarray(ae_scores_orig, dtype=float)}


def run_embedding_loo_cv(
    features_orig: Mapping[str, np.ndarray],
    features_aug: Mapping[str, np.ndarray],
    aug_sources: np.ndarray,
    labels_binary: np.ndarray,
    noise_idx: np.ndarray,
    *,
    embeddings: Sequence[str] | None = None,
    detectors_to_use: Sequence[str] | None = None,
    use_pca: bool = True,
    n_pca: int = 50,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Run the notebook's efficient embedding LOO-CV detector bank."""

    try:
        from sklearn.base import clone
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise _missing_sklearn() from exc

    labels_binary = np.asarray(labels_binary, dtype=int)
    noise_idx = np.asarray(noise_idx, dtype=int)
    aug_sources = np.asarray(aug_sources, dtype=int)

    if embeddings is None:
        emb_names = list(features_orig)
    else:
        emb_names = [name for name in embeddings if name in features_orig]

    if not emb_names:
        raise ValueError("No valid embeddings were provided.")

    all_scores: dict[str, np.ndarray] = {}
    for emb_name in emb_names:
        feats_orig = np.asarray(features_orig[emb_name], dtype=np.float32)
        feats_aug = np.asarray(features_aug[emb_name], dtype=np.float32)

        if len(feats_orig) != len(labels_binary):
            raise ValueError(
                f"Embedding '{emb_name}' has {len(feats_orig)} original rows but "
                f"{len(labels_binary)} labels were provided."
            )
        if len(feats_aug) != len(aug_sources):
            raise ValueError(
                f"Embedding '{emb_name}' has {len(feats_aug)} augmented rows but "
                f"{len(aug_sources)} augmentation sources were provided."
            )

        for test_i in range(len(labels_binary)):
            train_noise_idx = noise_idx[noise_idx != test_i]
            aug_mask = aug_sources != test_i

            x_train = np.vstack([feats_orig[train_noise_idx], feats_aug[aug_mask]])
            x_test = feats_orig[test_i : test_i + 1]

            scaler = StandardScaler()
            x_train_s = scaler.fit_transform(x_train)
            x_test_s = scaler.transform(x_test)

            if use_pca:
                n_comp = min(n_pca, x_train_s.shape[0] - 1, x_train_s.shape[1])
                if n_comp >= 1:
                    pca = PCA(n_components=n_comp, random_state=seed)
                    x_train_s = pca.fit_transform(x_train_s)
                    x_test_s = pca.transform(x_test_s)

            n_neighbors = min(15, max(1, len(x_train_s) - 1))
            detectors = _build_embedding_detectors(n_neighbors=n_neighbors, seed=seed)
            if detectors_to_use is not None:
                detectors = {name: detectors[name] for name in detectors_to_use if name in detectors}
            if not detectors:
                raise ValueError("No valid detectors were selected.")

            for det_name, det_template in detectors.items():
                key = f"{emb_name}_{det_name}"
                all_scores.setdefault(key, np.zeros(len(labels_binary), dtype=float))

                try:
                    detector = MahalanobisDetector() if det_name == "Mahal" else clone(det_template)
                    detector.fit(x_train_s)
                    all_scores[key][test_i] = float(-detector.score_samples(x_test_s)[0])
                except Exception:
                    all_scores[key][test_i] = 0.0

    return all_scores


def run_efficient_loo_cv(*args, **kwargs) -> dict[str, np.ndarray]:
    """Notebook-compatible alias for :func:`run_embedding_loo_cv`."""

    return run_embedding_loo_cv(*args, **kwargs)


def linear_probe_auc(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_splits: int = 5,
    seed: int = 42,
) -> dict[str, np.ndarray | float]:
    """Compute an out-of-fold linear probe benchmark."""

    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import average_precision_score, roc_auc_score
        from sklearn.model_selection import StratifiedKFold
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise _missing_sklearn() from exc

    y = np.asarray(y, dtype=int)
    X = np.asarray(X, dtype=np.float32)

    bincount = np.bincount(y)
    n_splits = min(n_splits, int(bincount.min())) if bincount.size else 0
    if n_splits < 2:
        return {"auc": np.nan, "ap": np.nan, "preds": np.full(len(y), 0.5, dtype=np.float32)}

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(y), dtype=np.float32)

    for train_idx, test_idx in skf.split(X, y):
        scaler = StandardScaler()
        x_train = scaler.fit_transform(X[train_idx])
        x_test = scaler.transform(X[test_idx])

        clf = LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            solver="liblinear",
            random_state=seed,
        )
        clf.fit(x_train, y[train_idx])
        oof[test_idx] = clf.predict_proba(x_test)[:, 1]

    return {
        "auc": float(roc_auc_score(y, oof)),
        "ap": float(average_precision_score(y, oof)),
        "preds": oof,
    }


def score_isolation_forest(
    train_features: np.ndarray,
    test_features: np.ndarray,
    *,
    contamination: float = 0.01,
    seed: int = 42,
) -> np.ndarray:
    """Fit an Isolation Forest on the training bank and score the test bank."""

    try:
        from sklearn.ensemble import IsolationForest
    except ImportError as exc:
        raise _missing_sklearn() from exc

    detector = IsolationForest(contamination=contamination, random_state=seed)
    detector.fit(np.asarray(train_features, dtype=np.float32))
    return -detector.score_samples(np.asarray(test_features, dtype=np.float32))
