"""Lazy anomaly-detection API extracted from the PSP analysis notebook."""

from __future__ import annotations

from importlib import import_module


_NAME_TO_MODULE = {
    "AffineCoupling": "ember.anomaly.neural",
    "AnomalyDataset": "ember.anomaly.pipeline",
    "ContrastiveEncoder": "ember.anomaly.neural",
    "MahalanobisDetector": "ember.anomaly.classical",
    "MultiModelExtractor": "ember.anomaly.embeddings",
    "PreparedSpectrogramBank": "ember.anomaly.embeddings",
    "PhysicsAugmenter": "ember.anomaly.features",
    "SimpleFlow": "ember.anomaly.neural",
    "SpectrogramAE": "ember.anomaly.neural",
    "align_spectrogram_pair": "ember.anomaly.features",
    "analyze_class": "ember.anomaly.evaluation",
    "analyze_fpr": "ember.anomaly.evaluation",
    "anomaly_score": "ember.anomaly.neural",
    "augment_batch": "ember.anomaly.neural",
    "blend_scores": "ember.anomaly.neural",
    "bootstrap_eval": "ember.anomaly.evaluation",
    "build_case_scores": "ember.anomaly.evaluation",
    "build_default_case_orders": "ember.anomaly.evaluation",
    "build_feature_bank": "ember.anomaly.pipeline",
    "build_noise_dataloaders": "ember.anomaly.neural",
    "choose_coupling_channel": "ember.anomaly.features",
    "collect_spectrograms": "ember.anomaly.features",
    "compute_umap_projection": "ember.anomaly.embeddings",
    "compare_budgeted_cases": "ember.anomaly.evaluation",
    "compute_default_ensembles": "ember.anomaly.ensemble",
    "contrastive_loss": "ember.anomaly.neural",
    "encode_specs": "ember.anomaly.neural",
    "ens_mean": "ember.anomaly.ensemble",
    "ens_mean_loo": "ember.anomaly.ensemble",
    "ens_rank": "ember.anomaly.ensemble",
    "ens_rank_loo": "ember.anomaly.ensemble",
    "ens_topk": "ember.anomaly.ensemble",
    "ens_topk_by_auc": "ember.anomaly.ensemble",
    "ens_topk_loo": "ember.anomaly.ensemble",
    "ens_weighted": "ember.anomaly.ensemble",
    "ens_weighted_loo": "ember.anomaly.ensemble",
    "evaluate_budgeted_accumulation": "ember.anomaly.evaluation",
    "extract_coupling_feature_matrix": "ember.anomaly.features",
    "extract_coupling_features": "ember.anomaly.features",
    "extract_physics_feature_matrix": "ember.anomaly.features",
    "extract_physics_features": "ember.anomaly.features",
    "fit_zuko_maf": "ember.anomaly.neural",
    "flow_score": "ember.anomaly.neural",
    "greedy_zero_fp_order": "ember.anomaly.evaluation",
    "greedy_zero_fp_order_oracle": "ember.anomaly.evaluation",
    "hits_at_fpr": "ember.anomaly.evaluation",
    "info_nce_loss": "ember.anomaly.neural",
    "linear_probe_auc": "ember.anomaly.classical",
    "log_prob": "ember.anomaly.neural",
    "norm": "ember.anomaly.ensemble",
    "norm01": "ember.anomaly.ensemble",
    "prepare_anomaly_dataset": "ember.anomaly.pipeline",
    "prepare_embedding_spectrograms": "ember.anomaly.embeddings",
    "prepare_multimodel_spectrograms": "ember.anomaly.embeddings",
    "prepare_specs_tensor": "ember.anomaly.neural",
    "print_budgeted_case": "ember.anomaly.evaluation",
    "run_efficient_loo_cv": "ember.anomaly.classical",
    "run_classical_anomaly_workflow": "ember.anomaly.pipeline",
    "run_default_case_analysis": "ember.anomaly.pipeline",
    "run_embedding_anomaly_workflow": "ember.anomaly.pipeline",
    "run_embedding_loo_cv": "ember.anomaly.classical",
    "run_physics_loo_cv": "ember.anomaly.classical",
    "render_budgeted_case": "ember.anomaly.evaluation",
    "score_isolation_forest": "ember.anomaly.classical",
    "score_lightgbm_meta_learner": "ember.anomaly.ensemble",
    "score_simple_flow": "ember.anomaly.neural",
    "score_with_ae": "ember.anomaly.neural",
    "score_with_ae_loo": "ember.anomaly.neural",
    "summarize_feature_discrimination": "ember.anomaly.evaluation",
    "summarize_methods": "ember.anomaly.evaluation",
    "threshold_detections": "ember.anomaly.evaluation",
    "threshold_zero_fp": "ember.anomaly.evaluation",
    "train_autoencoder": "ember.anomaly.neural",
    "train_contrastive_encoder": "ember.anomaly.neural",
    "train_simple_flow": "ember.anomaly.neural",
    # ── Robust deployment detectors ──────────────────────────────────────────
    "LocalPatchDetector":            "ember.anomaly.detectors",
    "BandDeviationDetector":         "ember.anomaly.detectors",
    "fit_detector_suite":            "ember.anomaly.detectors",
    "score_detector_suite":          "ember.anomaly.detectors",
    "make_recon_gate":               "ember.anomaly.detectors",
    # ── PatchCore ─────────────────────────────────────────────────────────────
    "PatchCoreDetector":             "ember.anomaly.patchcore",
    # ── β-VAE ─────────────────────────────────────────────────────────────────
    "SpectrogramVAE":                "ember.anomaly.neural",
    "train_vae":                     "ember.anomaly.neural",
    "score_vae":                     "ember.anomaly.neural",
    # ── Calibration-based ensemble ────────────────────────────────────────────
    "rank_normalise":                "ember.anomaly.ensemble",
    "optimise_ensemble_weights":     "ember.anomaly.ensemble",
    "greedy_cascade_detection":      "ember.anomaly.ensemble",
    # ── Production pipeline ───────────────────────────────────────────────────
    "run_robust_anomaly_pipeline":   "ember.anomaly.pipeline",
    "predict":                       "ember.anomaly.pipeline",
    # ── Publication plots ─────────────────────────────────────────────────────
    "plot_score_distributions":      "ember.anomaly.plotting",
    "plot_roc_with_operating_points":"ember.anomaly.plotting",
    "plot_spectrogram_grid":         "ember.anomaly.plotting",
    "plot_detector_heatmap":         "ember.anomaly.plotting",
    "plot_fp_cluster_timeseries":    "ember.anomaly.plotting",
    "plot_weight_bar":               "ember.anomaly.plotting",
}

__all__ = sorted(_NAME_TO_MODULE)


def __getattr__(name: str):
    if name not in _NAME_TO_MODULE:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = import_module(_NAME_TO_MODULE[name])
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
