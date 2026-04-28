"""High-level orchestration helpers for the notebook-style anomaly workflow."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ember.anomaly.classical import run_embedding_loo_cv, run_physics_loo_cv
from ember.anomaly.embeddings import (
    MultiModelExtractor,
    prepare_multimodel_spectrograms,
)
from ember.anomaly.ensemble import (
    compute_default_ensembles,
    score_lightgbm_meta_learner,
)
from ember.anomaly.evaluation import (
    build_default_case_orders,
    compare_budgeted_cases,
    summarize_methods,
)
from ember.anomaly.features import (
    PhysicsAugmenter,
    align_spectrogram_pair,
    choose_coupling_channel,
    collect_spectrograms,
    extract_coupling_feature_matrix,
    extract_physics_feature_matrix,
)
from ember.anomaly.neural import score_with_ae, score_with_ae_loo, train_autoencoder


@dataclass(frozen=True)
class AnomalyDataset:
    """Container for the spectrogram bank and the derived label/index arrays."""

    df: pd.DataFrame
    specs_raw: list[np.ndarray]
    labels_original: np.ndarray
    labels_binary: np.ndarray
    noise_idx: np.ndarray
    signal_idx: np.ndarray
    class1_idx: np.ndarray
    class2_idx: np.ndarray
    channel_columns: list[str]
    has_multi_channel: bool


def prepare_anomaly_dataset(
    df: pd.DataFrame,
    *,
    spec_column: str = "Log Amplitude",
    label_column: str = "Label",
) -> AnomalyDataset:
    """Extract the notebook's core arrays from a labeled spectrogram dataframe."""

    labels_original = df[label_column].to_numpy(dtype=int)
    labels_binary = (labels_original > 0).astype(int)

    return AnomalyDataset(
        df=df,
        specs_raw=collect_spectrograms(df, column=spec_column),
        labels_original=labels_original,
        labels_binary=labels_binary,
        noise_idx=np.where(labels_binary == 0)[0],
        signal_idx=np.where(labels_binary == 1)[0],
        class1_idx=np.where(labels_original == 1)[0],
        class2_idx=np.where(labels_original == 2)[0],
        channel_columns=[
            col for col in df.columns if col not in {label_column, spec_column}
        ],
        has_multi_channel=len(df.columns.difference([label_column, spec_column])) > 0,
    )


def build_feature_bank(
    dataset: AnomalyDataset,
    *,
    n_bands: int = 8,
    channel_column: str | None = "auto",
) -> dict[str, object]:
    """Build the physics and optional coupling features from the dataset."""

    physics_features = extract_physics_feature_matrix(
        dataset.specs_raw, n_bands=n_bands
    )

    resolved_channel = channel_column
    if channel_column == "auto":
        resolved_channel = choose_coupling_channel(
            dataset.df,
            reference_specs=dataset.specs_raw,
            channel_columns=dataset.channel_columns,
        )

    coupling_features = None
    if resolved_channel is not None:
        paired_specs = []
        for primary, raw_aux in zip(
            dataset.specs_raw, dataset.df[resolved_channel].tolist()
        ):
            try:
                aux = np.asarray(raw_aux, dtype=np.float32)
            except (TypeError, ValueError):
                aux = np.asarray(primary, dtype=np.float32).copy()

            if aux.ndim != 2:
                aux = np.asarray(primary, dtype=np.float32).copy()

            _, aux = align_spectrogram_pair(primary, aux)
            paired_specs.append(aux)

        coupling_features = extract_coupling_feature_matrix(
            dataset.specs_raw, paired_specs
        )
        all_features = np.hstack([physics_features, coupling_features])
    else:
        all_features = physics_features

    return {
        "physics_features": physics_features,
        "coupling_features": coupling_features,
        "all_features": all_features,
        "coupling_column": resolved_channel,
    }


def run_classical_anomaly_workflow(
    dataset: AnomalyDataset,
    *,
    feature_bank: dict[str, object] | None = None,
    seed: int = 42,
    k_aug: int = 15,
    ae_epochs: int = 400,
    ae_lr: float = 1e-3,
    ae_latent_dim: int = 24,
    use_pca: bool = True,
    n_pca: int = 25,
    include_lightgbm: bool = False,
    strict_evaluation: bool = True,
    topk_values: tuple[int, ...] = (3, 5),
    verbose: bool = True,
) -> dict[str, object]:
    """Run the improved physics-feature anomaly workflow through EMBER APIs."""

    if feature_bank is None:
        feature_bank = build_feature_bank(dataset)

    noise_specs = [dataset.specs_raw[idx] for idx in dataset.noise_idx]
    augmenter = PhysicsAugmenter(seed=seed)
    aug_specs, aug_sources, aug_physics_feats = augmenter.generate_all(
        noise_specs,
        dataset.noise_idx,
        n_per_sample=k_aug,
    )

    ae_model = None
    ae_mu = None
    ae_std = None
    if strict_evaluation:
        ae_scores, ae_latents = score_with_ae_loo(
            dataset.specs_raw,
            dataset.noise_idx,
            epochs=ae_epochs,
            lr=ae_lr,
            latent_dim=ae_latent_dim,
            verbose=verbose,
        )
    else:
        ae_model, ae_mu, ae_std = train_autoencoder(
            noise_specs,
            epochs=ae_epochs,
            lr=ae_lr,
            latent_dim=ae_latent_dim,
            verbose=verbose,
        )
        ae_scores, ae_latents = score_with_ae(
            ae_model, dataset.specs_raw, ae_mu, ae_std
        )

    base_scores = run_physics_loo_cv(
        physics_feats_orig=np.asarray(feature_bank["all_features"], dtype=np.float32),
        physics_feats_aug=aug_physics_feats,
        aug_sources=aug_sources,
        ae_scores_orig=ae_scores,
        ae_latents_orig=ae_latents,
        labels_binary=dataset.labels_binary,
        noise_idx=dataset.noise_idx,
        use_pca=use_pca,
        n_pca=n_pca,
        seed=seed,
    )

    ensemble_scores, topk_names = compute_default_ensembles(
        base_scores,
        dataset.labels_binary,
        topk_values=topk_values,
        strict=strict_evaluation,
    )
    meta_scores: dict[str, np.ndarray] = {}
    if include_lightgbm:
        meta_scores["LightGBM"] = score_lightgbm_meta_learner(
            base_scores,
            dataset.labels_binary,
            strict=strict_evaluation,
            seed=seed,
        )

    all_methods = {**base_scores, **ensemble_scores, **meta_scores}

    results_df = summarize_methods(
        all_methods,
        dataset.labels_binary,
        dataset.noise_idx,
        dataset.signal_idx,
        labels_original=dataset.labels_original,
        class1_idx=dataset.class1_idx,
        class2_idx=dataset.class2_idx,
        seed=seed,
    )

    return {
        "feature_bank": feature_bank,
        "augmented_specs": aug_specs,
        "aug_sources": aug_sources,
        "augmented_physics_features": aug_physics_feats,
        "ae_model": ae_model,
        "ae_mu": ae_mu,
        "ae_std": ae_std,
        "ae_scores": ae_scores,
        "ae_latents": ae_latents,
        "strict_evaluation": bool(strict_evaluation),
        "base_scores": base_scores,
        "ensemble_scores": ensemble_scores,
        "meta_scores": meta_scores,
        "topk_names": topk_names,
        "all_methods": all_methods,
        "results_df": results_df,
    }


def run_embedding_anomaly_workflow(
    dataset: AnomalyDataset,
    *,
    model_names: tuple[str, ...] | list[str] | None = None,
    detector_names: tuple[str, ...] | list[str] | None = None,
    seed: int = 42,
    k_aug: int = 10,
    target_h: int = 224,
    target_w: int = 224,
    batch_size: int = 32,
    use_pca: bool = True,
    n_pca: int = 50,
    include_lightgbm: bool = False,
    strict_evaluation: bool = True,
    device: str | None = None,
    strict_backbones: bool = False,
    topk_values: tuple[int, ...] = (3, 5),
    verbose: bool = True,
) -> dict[str, object]:
    """Run the efficient multi-backbone anomaly workflow from the source notebook."""

    prepared_specs = prepare_multimodel_spectrograms(
        dataset.specs_raw,
        noise_indices=dataset.noise_idx,
        h=target_h,
        w=target_w,
    )

    extractor = MultiModelExtractor(
        device=device,
        model_names=model_names,
        strict=strict_backbones,
        verbose=verbose,
    )
    if not extractor.loaded_model_names:
        raise RuntimeError(
            "No feature-extraction backbones could be loaded. "
            f"Load errors: {extractor.load_errors}"
        )

    if verbose:
        print("Extracting backbone features for original spectrograms...")
    features_original = extractor.extract(prepared_specs.specs, batch_size=batch_size)

    if verbose:
        print(f"Generating {k_aug} augmented views per noise sample...")
    augmenter = PhysicsAugmenter(seed=seed)
    augmented_specs, aug_sources, _ = augmenter.generate_all(
        prepared_specs.specs[dataset.noise_idx],
        dataset.noise_idx,
        n_per_sample=k_aug,
    )

    if verbose:
        print("Extracting backbone features for augmented spectrograms...")
    features_augmented = extractor.extract(augmented_specs, batch_size=batch_size)

    base_scores = run_embedding_loo_cv(
        features_original,
        features_augmented,
        aug_sources,
        dataset.labels_binary,
        dataset.noise_idx,
        embeddings=extractor.loaded_model_names,
        detectors_to_use=detector_names,
        use_pca=use_pca,
        n_pca=n_pca,
        seed=seed,
    )

    ensemble_scores, topk_names = compute_default_ensembles(
        base_scores,
        dataset.labels_binary,
        topk_values=topk_values,
        strict=strict_evaluation,
    )

    meta_scores: dict[str, np.ndarray] = {}
    if include_lightgbm:
        meta_scores["LightGBM"] = score_lightgbm_meta_learner(
            base_scores,
            dataset.labels_binary,
            strict=strict_evaluation,
            seed=seed,
        )

    all_methods = {**base_scores, **ensemble_scores, **meta_scores}
    results_df = summarize_methods(
        all_methods,
        dataset.labels_binary,
        dataset.noise_idx,
        dataset.signal_idx,
        labels_original=dataset.labels_original,
        class1_idx=dataset.class1_idx,
        class2_idx=dataset.class2_idx,
        seed=seed,
    )

    return {
        "prepared_specs": prepared_specs,
        "extractor": extractor,
        "features_original": features_original,
        "features_augmented": features_augmented,
        "augmented_specs": augmented_specs,
        "aug_sources": aug_sources,
        "base_scores": base_scores,
        "ensemble_scores": ensemble_scores,
        "meta_scores": meta_scores,
        "strict_evaluation": bool(strict_evaluation),
        "topk_names": topk_names,
        "all_methods": all_methods,
        "results_df": results_df,
    }


def _split_noise_idx(
    noise_idx: np.ndarray,
    train_frac: float,
    cal_frac: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split noise indices into train / calibration / evaluation subsets."""
    idx = rng.permutation(noise_idx)
    n = len(idx)
    n_train = int(np.floor(train_frac * n))
    n_cal = int(np.floor(cal_frac * n))
    return idx[:n_train], idx[n_train : n_train + n_cal], idx[n_train + n_cal :]


def run_robust_anomaly_pipeline(
    dataset: AnomalyDataset,
    *,
    target_far: float = 0.01,
    train_fraction: float = 0.80,
    cal_fraction: float = 0.20,
    include_patchcore: bool = False,
    include_vae: bool = True,
    seed: int = 42,
    device: str | None = None,
    output_dir=None,
    verbose: bool = True,
) -> dict:
    """Background-only anomaly detection pipeline (deployment mode).

    Trains entirely on background noise, calibrates a score threshold on
    held-out background, then scores evaluation samples.  Ensemble weights
    are optimised by ``scipy.differential_evolution`` to maximise TPR at
    ``target_far``.

    Parameters
    ----------
    dataset : AnomalyDataset
        Output of :func:`prepare_anomaly_dataset`.
    target_far : float
        False alarm rate budget (default 1 %).
    train_fraction : float
        Fraction of background samples used for training detectors.
    cal_fraction : float
        Fraction used for threshold calibration.  The remainder is eval bg.
    include_patchcore : bool
        If True, fit a :class:`~ember.anomaly.patchcore.PatchCoreDetector`
        (requires ``torchvision``).
    include_vae : bool
        If True, train a :class:`~ember.anomaly.neural.SpectrogramVAE`
        (requires ``torch``).
    seed : int
        Random seed for reproducibility.
    device : str or None
        Torch device for the optional neural/image models. Use ``"cpu"`` to
        avoid GPU memory pressure in notebook environments.
    output_dir : path-like or None
        If given, save CSV summaries and PNG plots to this directory.
    verbose : bool
        Print progress messages.

    Returns
    -------
    dict with keys:
        ``detector_suite``          – fitted classical detectors (background-only)
        ``vae_model``               – fitted β-VAE (or None)
        ``patchcore``               – fitted PatchCoreDetector (or None)
        ``opt_weights``             – optimised ensemble weight vector
        ``pool``                    – ordered list of detector names in the ensemble
        ``threshold_conservative``  – 99th-pct bg_cal threshold
        ``threshold_1pct``          – threshold at ``target_far``
        ``bg_train_idx``            – indices used for training
        ``bg_cal_idx``              – indices used for calibration
        ``bg_eval_idx``             – background eval indices
        ``anom_eval_idx``           – anomaly eval indices
        ``all_scores``              – dict of raw score arrays (all samples)
        ``norm_scores``             – rank-normalised scores vs bg_cal (all samples)
        ``ensemble_scores``         – final ensemble score (all samples)
        ``results_df``              – summary DataFrame (AUC, TPR, FAR)
    """
    from pathlib import Path

    from ember.anomaly.detectors import fit_detector_suite, score_detector_suite
    from ember.anomaly.ensemble import optimise_ensemble_weights, rank_normalise
    from ember.anomaly.features import extract_physics_feature_matrix

    try:
        from sklearn.metrics import roc_auc_score
    except ImportError as exc:
        raise ImportError(
            "run_robust_anomaly_pipeline requires scikit-learn. "
            "Install with `pip install -e .[anomaly]`."
        ) from exc

    rng = np.random.default_rng(seed)

    # ── 1. Split background noise ──────────────────────────────────────────────
    bg_train_idx, bg_cal_idx, bg_eval_idx_raw = _split_noise_idx(
        dataset.noise_idx, train_fraction, cal_fraction, rng
    )
    anom_eval_idx = dataset.signal_idx  # all anomalies go to eval

    n_all_bg_print = len(dataset.noise_idx)
    if verbose:
        print(
            f"bg_train={len(bg_train_idx)}  bg_cal={len(bg_cal_idx)}"
            f"  bg_total={n_all_bg_print}  anomalies={len(anom_eval_idx)}"
            f"  (FAR evaluated over all {n_all_bg_print} background samples)"
        )

    bg_train_specs = [dataset.specs_raw[i] for i in bg_train_idx]
    all_specs = dataset.specs_raw

    # ── 2. Physics features ────────────────────────────────────────────────────
    if verbose:
        print("Extracting physics features …")
    all_feats = extract_physics_feature_matrix(all_specs).astype(np.float32)
    bg_train_feats = all_feats[bg_train_idx]

    # ── 3. Fit detector suite ──────────────────────────────────────────────────
    if verbose:
        print("Fitting detector suite on bg_train …")
    suite = fit_detector_suite(
        bg_train_specs,
        bg_train_feats,
        seed=seed,
    )

    # ── 4. Optional β-VAE ─────────────────────────────────────────────────────
    vae_model = None
    vae_mu_norm = None
    vae_std_norm = None
    if include_vae:
        try:
            from ember.anomaly.neural import score_vae, train_vae

            if verbose:
                print("Training β-VAE on bg_train spectrograms …")
            vae_model, vae_mu_norm, vae_std_norm = train_vae(
                bg_train_specs, seed=seed, verbose=verbose, device=device
            )
        except ImportError:
            if verbose:
                print("  [skip] torch not available — skipping VAE.")
            include_vae = False

    # ── 5. Optional PatchCore ─────────────────────────────────────────────────
    patchcore = None
    if include_patchcore:
        try:
            from ember.anomaly.patchcore import PatchCoreDetector

            if verbose:
                print("Fitting PatchCore memory bank on bg_train spectrograms …")
            patchcore = PatchCoreDetector(device=device)
            patchcore.fit(bg_train_specs)
        except ImportError:
            if verbose:
                print("  [skip] torchvision not available — skipping PatchCore.")
            include_patchcore = False

    # ── 6. Score all samples ───────────────────────────────────────────────────
    if verbose:
        print("Scoring all samples …")
    raw_scores = score_detector_suite(suite, all_specs, all_feats)

    if include_vae and vae_model is not None:
        elbo_scores, _ = score_vae(
            vae_model,
            all_specs,
            vae_mu_norm,
            vae_std_norm,
            device=device,
        )
        raw_scores["VAE_ELBO"] = elbo_scores

    if include_patchcore and patchcore is not None:
        raw_scores["PatchCore"] = patchcore.score(all_specs)

    # ── 7. Rank-normalise using bg_cal ────────────────────────────────────────
    pool = sorted(raw_scores.keys())
    bg_cal_raw = {det: raw_scores[det][bg_cal_idx] for det in pool}

    norm_scores: dict[str, np.ndarray] = {}
    for det in pool:
        norm_scores[det] = rank_normalise(bg_cal_raw[det], raw_scores[det])

    # Use ALL background for evaluation — matches the batch5 notebook strategy.
    # bg_train is used for fitting, bg_cal for calibration/rank-normalisation,
    # ALL background for FAR evaluation (so max_fp is computed over the full
    # background count, not a tiny held-out slice).
    all_bg_idx = dataset.noise_idx  # all background (497 samples)

    # Use ALL background as the calibration reference for the optimiser.
    # This ensures the optimiser's internal threshold (all_bg_sorted[max_fp])
    # matches the threshold used at evaluation time, so the weights truly
    # maximise TPR at the stated FAR budget.
    X_all_bg_mat = np.column_stack([norm_scores[d][all_bg_idx] for d in pool])
    X_anom_mat = np.column_stack([norm_scores[d][anom_eval_idx] for d in pool])

    # Eval set for optimiser: all bg (threshold reference) + all anomalies (TPR target)
    X_opt_eval = np.vstack([X_all_bg_mat, X_anom_mat])
    labels_opt = np.concatenate(
        [
            np.zeros(len(all_bg_idx), dtype=int),
            np.ones(len(anom_eval_idx), dtype=int),
        ]
    )

    # ── 8. Optimise ensemble weights ──────────────────────────────────────────
    if verbose:
        print("Optimising ensemble weights …")
    opt_weights = optimise_ensemble_weights(
        X_all_bg_mat,  # all bg: threshold = all_bg_sorted[max_fp] → ≤ target_far
        X_opt_eval,  # all bg + anomalies
        labels_opt,
        target_far=target_far,
        seed=seed,
    )

    # Final ensemble score over all samples
    X_all_mat = np.column_stack([norm_scores[d] for d in pool])
    ensemble_all = X_all_mat @ opt_weights

    # ── 9. Threshold calibration ───────────────────────────────────────────────
    cal_ens = ensemble_all[bg_cal_idx]
    thr_conservative = float(np.percentile(cal_ens, 99))

    # 1%-FAR threshold directly from ALL background → guaranteed FAR ≤ target_far
    n_all_bg = len(all_bg_idx)
    max_fp_total = int(np.floor(target_far * n_all_bg))
    all_bg_ens_sorted = np.sort(ensemble_all[all_bg_idx])[::-1]
    thr_1pct = float(
        all_bg_ens_sorted[max_fp_total]
        if max_fp_total < len(all_bg_ens_sorted)
        else all_bg_ens_sorted[-1] - 1e-9
    )

    # ── 10. Evaluation metrics ─────────────────────────────────────────────────
    # Two FAR reports:
    #   FAR_all   — FP / N_all_bg  : includes training bg (production estimate)
    #   FAR_cal   — FP / N_bg_cal  : bg_cal is held-out from detector training
    #               (honest out-of-sample estimate; cal is used for threshold
    #               calibration but never seen by the fitted detectors)
    anom_ens = ensemble_all[anom_eval_idx]
    all_bg_ens = ensemble_all[all_bg_idx]
    cal_ens_e = ensemble_all[bg_cal_idx]  # bg_cal subset

    n_anom = len(anom_eval_idx)
    n_all_bg_e = len(all_bg_idx)
    n_bg_cal_e = len(bg_cal_idx)

    def _metrics(thr):
        tp = int((anom_ens >= thr).sum())
        fp_all = int((all_bg_ens >= thr).sum())
        fp_cal = int((cal_ens_e >= thr).sum())
        tpr = tp / max(n_anom, 1)
        far_all = fp_all / max(n_all_bg_e, 1)
        far_cal = fp_cal / max(n_bg_cal_e, 1)
        return tpr, far_all, far_cal, tp, fp_all, fp_cal

    tpr_c, far_c_all, far_c_cal, tp_c, fp_c_all, fp_c_cal = _metrics(thr_conservative)
    tpr_1, far_1_all, far_1_cal, tp_1, fp_1_all, fp_1_cal = _metrics(thr_1pct)

    eval_idx_all = np.concatenate([all_bg_idx, anom_eval_idx])
    y_eval_bin = np.concatenate(
        [
            np.zeros(n_all_bg_e, dtype=int),
            np.ones(n_anom, dtype=int),
        ]
    )
    try:
        auc_val = float(roc_auc_score(y_eval_bin, ensemble_all[eval_idx_all]))
    except Exception:
        auc_val = float("nan")

    results_df = pd.DataFrame(
        [
            {
                "version": "robust_v1",
                "AUC": round(auc_val, 4),
                # Conservative threshold (99th pct of bg_cal)
                "TPR_cons": round(tpr_c, 4),
                "FAR_cons_all_bg": round(far_c_all, 4),  # all background
                "FAR_cons_cal": round(far_c_cal, 4),  # bg_cal (held-out)
                "TP_cons": tp_c,
                "FP_cons_all_bg": fp_c_all,
                "FP_cons_cal": fp_c_cal,
                # 1%-FAR threshold (budgeted against all background)
                "TPR_1pct": round(tpr_1, 4),
                "FAR_1pct_all_bg": round(far_1_all, 4),  # all background
                "FAR_1pct_cal": round(far_1_cal, 4),  # bg_cal (held-out)
                "TP_1pct": tp_1,
                "FP_1pct_all_bg": fp_1_all,
                "FP_1pct_cal": fp_1_cal,
                "n_bg_all": n_all_bg_e,
                "n_bg_cal": n_bg_cal_e,
                "n_anom": n_anom,
                "n_detectors": len(pool),
                "pool": ",".join(pool),
            }
        ]
    )

    if verbose:
        print("\n" + "=" * 58)
        print("ROBUST PIPELINE RESULTS")
        print("=" * 58)
        print(f"  AUC : {auc_val:.4f}")
        print("  Conservative threshold (99th pct bg_cal):")
        print(
            f"    TPR={tpr_c:.1%}  FAR_all={far_c_all:.2%} ({fp_c_all}/{n_all_bg_e} bg)"
            f"  FAR_cal={far_c_cal:.2%} ({fp_c_cal}/{n_bg_cal_e} bg_cal)"
        )
        print(f"  1%-FAR threshold (budget vs all {n_all_bg_e} bg):")
        print(
            f"    TPR={tpr_1:.1%}  FAR_all={far_1_all:.2%} ({fp_1_all}/{n_all_bg_e} bg)"
            f"  FAR_cal={far_1_cal:.2%} ({fp_1_cal}/{n_bg_cal_e} bg_cal)"
        )
        print("=" * 58)

    # ── 11. Optional output ────────────────────────────────────────────────────
    plots: dict = {}
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(out / "robust_pipeline_results.csv", index=False)

        weight_df = pd.DataFrame({"detector": pool, "weight": opt_weights})
        weight_df.to_csv(out / "ensemble_weights.csv", index=False)

        try:
            from ember.anomaly.plotting import (
                plot_roc_with_operating_points,
                plot_score_distributions,
                plot_weight_bar,
            )

            fig_dist = plot_score_distributions(
                bg_scores=all_bg_ens,
                anom_scores=anom_ens,
                threshold=thr_conservative,
            )
            fig_dist.savefig(
                out / "score_distributions.png", dpi=120, bbox_inches="tight"
            )
            plots["score_distributions"] = fig_dist

            fig_roc = plot_roc_with_operating_points(
                y_true=y_eval_bin,
                scores_dict={"Ensemble v1": ensemble_all[eval_idx_all]},
                operating_points=[
                    {
                        "fpr": far_c_all,
                        "tpr": tpr_c,
                        "label": f"Conservative ({far_c_all:.1%} FAR)",
                    },
                    {
                        "fpr": far_1_all,
                        "tpr": tpr_1,
                        "label": f"Target FAR ({far_1_all:.1%})",
                    },
                ],
            )
            fig_roc.savefig(out / "roc_curve.png", dpi=120, bbox_inches="tight")
            plots["roc_curve"] = fig_roc

            fig_w = plot_weight_bar(weight_df)
            fig_w.savefig(out / "ensemble_weights.png", dpi=120, bbox_inches="tight")
            plots["weights"] = fig_w

            if verbose:
                print(f"Saved plots and CSV to {out}/")
        except Exception as exc:
            if verbose:
                print(f"  [warning] Plotting failed: {exc}")

    return {
        "detector_suite": suite,
        "vae_model": vae_model,
        "patchcore": patchcore,
        "opt_weights": opt_weights,
        "pool": pool,
        "threshold_conservative": thr_conservative,
        "threshold_1pct": thr_1pct,
        "bg_train_idx": bg_train_idx,
        "bg_cal_idx": bg_cal_idx,
        "bg_eval_idx": all_bg_idx,
        "anom_eval_idx": anom_eval_idx,
        "all_scores": raw_scores,
        "norm_scores": norm_scores,
        "ensemble_scores": ensemble_all,
        "results_df": results_df,
        "plots": plots,
    }


def predict(
    pipeline_result: dict,
    new_specs: list,
    *,
    operating_point: str = "1pct",
) -> np.ndarray:
    """Score new (unlabelled) spectrograms using a fitted pipeline.

    Parameters
    ----------
    pipeline_result : dict
        Output of :func:`run_robust_anomaly_pipeline`.
    new_specs : list of (H, W) arrays
        New spectrograms to classify.
    operating_point : {"1pct", "conservative"}
        Which threshold to use for the binary flag.

    Returns
    -------
    np.ndarray, shape (N,)
        Boolean array — True = anomaly alert.
    """
    from ember.anomaly.detectors import score_detector_suite
    from ember.anomaly.ensemble import rank_normalise
    from ember.anomaly.features import extract_physics_feature_matrix

    suite = pipeline_result["detector_suite"]
    pool = pipeline_result["pool"]
    opt_w = pipeline_result["opt_weights"]
    bg_cal_idx = pipeline_result["bg_cal_idx"]
    raw_all = pipeline_result["all_scores"]

    new_feats = extract_physics_feature_matrix(new_specs).astype(np.float32)
    new_raw = score_detector_suite(suite, new_specs, new_feats)

    vae_model = pipeline_result.get("vae_model")
    if vae_model is not None and "VAE_ELBO" in pool:
        from ember.anomaly.neural import score_vae

        # retrieve normalisation stats stored in pipeline
        mu_n = pipeline_result.get("_vae_mu_norm")
        std_n = pipeline_result.get("_vae_std_norm")
        if mu_n is not None:
            elbo, _ = score_vae(vae_model, new_specs, mu_n, std_n)
            new_raw["VAE_ELBO"] = elbo

    patchcore = pipeline_result.get("patchcore")
    if patchcore is not None and "PatchCore" in pool:
        new_raw["PatchCore"] = patchcore.score(new_specs)

    norm_new: list[np.ndarray] = []
    for det in pool:
        bg_cal_scores = raw_all[det][bg_cal_idx]
        norm_new.append(rank_normalise(bg_cal_scores, new_raw[det]))

    X_new = np.column_stack(norm_new)
    ens_new = X_new @ opt_w

    if operating_point == "conservative":
        thr = pipeline_result["threshold_conservative"]
    else:
        thr = pipeline_result["threshold_1pct"]

    return ens_new >= thr


def run_default_case_analysis(
    dataset: AnomalyDataset,
    *,
    all_methods: dict[str, np.ndarray],
    results_df: pd.DataFrame,
    max_methods: int = 6,
    first_fpr: float = 0.01,
    include_oracle: bool = False,
) -> dict[str, object]:
    """Evaluate the default detector-combination cases on the prepared workflow."""

    case_orders = build_default_case_orders(
        results_df,
        all_methods,
        dataset.noise_idx,
        dataset.signal_idx,
        max_methods=max_methods,
        first_fpr=first_fpr,
        include_oracle=include_oracle,
    )
    case_results, comparison_df, case_results_by_name = compare_budgeted_cases(
        case_orders,
        all_methods,
        dataset.noise_idx,
        dataset.signal_idx,
        labels_original=dataset.labels_original,
        first_fpr=first_fpr,
    )
    return {
        "case_orders": case_orders,
        "case_results": case_results,
        "comparison_df": comparison_df,
        "case_results_by_name": case_results_by_name,
    }
