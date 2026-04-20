"""Reusable EMBER package API."""

from __future__ import annotations

from importlib import import_module


_NAME_TO_MODULE = {
    "HOURS": "ember.download",
    "SpectrogramRecord": "ember.spectrograms",
    "anomaly": "ember.anomaly",
    "build_detection_matrix": "ember.reporting",
    "build_download_plan": "ember.download",
    "build_filename": "ember.download",
    "build_url": "ember.download",
    "create_dvac_spectrograms": "ember.spectrograms",
    "create_spectrograms": "ember.spectrograms",
    "create_vac_spectrograms": "ember.spectrograms",
    "date_range": "ember.download",
    "download_file": "ember.download",
    "download_products": "ember.download",
    "extract_label_examples": "ember.datasets",
    "iter_spectrogram_records": "ember.spectrograms",
    "load_case_summary": "ember.reporting",
    "load_detection_votes": "ember.reporting",
    "load_labeled_spectrogram_dataframe": "ember.datasets",
    "load_saved_spectrogram": "ember.spectrograms",
    "load_voltage_bursts": "ember.spectrograms",
    "plot_case_recovery_summary": "ember.reporting",
    "plot_detection_map": "ember.reporting",
    "plot_feature_discrimination": "ember.reporting",
    "plot_labeled_projection": "ember.reporting",
    "plot_spectrogram": "ember.spectrograms",
    "plot_three_class_examples": "ember.reporting",
    "save_repo_figures": "ember.reporting",
    "spectrogram_from_row": "ember.datasets",
}

__all__ = sorted(_NAME_TO_MODULE)


def __getattr__(name: str):
    if name not in _NAME_TO_MODULE:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = import_module(_NAME_TO_MODULE[name])
    value = module if name == "anomaly" else getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
