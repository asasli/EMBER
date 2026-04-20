"""Publication-quality plotting helpers for the robust anomaly pipeline.

All functions return a ``matplotlib.Figure`` so callers can save, display,
or embed them as needed.  Nothing is shown or saved automatically.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Score distributions
# ---------------------------------------------------------------------------

def plot_score_distributions(
    bg_scores: np.ndarray,
    anom_scores: np.ndarray,
    threshold: float,
    *,
    label1_mask: np.ndarray | None = None,
    label2_mask: np.ndarray | None = None,
    title: str = "Anomaly Score Distribution",
    xlabel: str = "Score",
) -> "matplotlib.figure.Figure":
    """Histogram of background vs anomaly scores with threshold line.

    Parameters
    ----------
    bg_scores : (N_bg,) array
        Scores for background (noise) eval samples.
    anom_scores : (N_anom,) array
        Scores for anomaly eval samples.
    threshold : float
        Detection threshold (e.g. 1% FAR operating point).
    label1_mask, label2_mask : bool arrays of length N_anom, optional
        Masks to colour label-1 and label-2 anomalies differently.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.hist(bg_scores,   bins=50, alpha=0.6, color="steelblue", label=f"Background (n={len(bg_scores)})")
    if label1_mask is not None and label2_mask is not None:
        ax.hist(anom_scores[label1_mask],  bins=20, alpha=0.75, color="orange", label="Anomaly label-1")
        ax.hist(anom_scores[label2_mask],  bins=20, alpha=0.75, color="tomato",  label="Anomaly label-2")
    else:
        ax.hist(anom_scores, bins=20, alpha=0.75, color="tomato", label=f"Anomaly (n={len(anom_scores)})")

    ax.axvline(threshold, color="red", lw=1.5, ls="--",
               label=f"Threshold ({threshold:.3f})")
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# ROC curve with operating points
# ---------------------------------------------------------------------------

def plot_roc_with_operating_points(
    y_true: np.ndarray,
    scores_dict: dict[str, np.ndarray],
    *,
    operating_points: list[tuple[float, float, str]] | None = None,
    far_xlim: float = 0.15,
    title: str = "ROC Curves",
) -> "matplotlib.figure.Figure":
    """ROC curves for multiple detectors with optional operating-point markers.

    Parameters
    ----------
    y_true : (N,) int array
        Binary labels (0 = background, 1 = anomaly).
    scores_dict : dict
        ``{label: score_array}`` — each array length N.
    operating_points : list of (fpr, tpr, label), optional
        Star markers placed at specific (FAR, TPR) coordinates.
    far_xlim : float
        X-axis upper limit (zoom into low-FAR region).
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_auc_score, roc_curve

    fig, ax = plt.subplots(figsize=(8, 6))

    colors = plt.cm.tab10(np.linspace(0, 0.9, len(scores_dict)))
    for (label, scores), color in zip(scores_dict.items(), colors):
        try:
            auc = roc_auc_score(y_true, scores)
            fpr, tpr, _ = roc_curve(y_true, scores)
            ax.plot(fpr, tpr, lw=1.8, color=color, label=f"{label} (AUC={auc:.3f})")
        except Exception:
            pass

    if operating_points:
        for fpr_op, tpr_op, op_label in operating_points:
            ax.scatter([fpr_op], [tpr_op], s=120, marker="*",
                       color="red", zorder=6, label=op_label)

    ax.axvline(0.01, color="gray", lw=1, ls="--", alpha=0.6, label="1% FAR")
    ax.set_xlim(-0.005, far_xlim)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("False Alarm Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Spectrogram grid
# ---------------------------------------------------------------------------

def plot_spectrogram_grid(
    groups: dict[str, Sequence[np.ndarray]],
    *,
    n_per_row: int = 4,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "plasma",
    scores: dict[str, Sequence[float]] | None = None,
    title: str = "Spectrogram Comparison",
) -> "matplotlib.figure.Figure":
    """Side-by-side spectrogram grid grouped by category.

    Parameters
    ----------
    groups : dict
        ``{row_label: [spec, spec, ...]}`` — each value is a list of 2D arrays.
    n_per_row : int
        Maximum number of spectrograms per row.
    vmin, vmax : float, optional
        Shared colour scale.  If ``None``, computed from all spectrograms (2nd / 98th pct).
    scores : dict, optional
        ``{row_label: [score, ...]}`` — displayed as subplot title if provided.
    """
    import matplotlib.pyplot as plt

    n_rows = len(groups)
    all_specs = [s for specs in groups.values() for s in specs]
    all_arr = np.array([np.asarray(s, dtype=np.float32) for s in all_specs])
    if vmin is None:
        vmin = float(np.percentile(all_arr, 2))
    if vmax is None:
        vmax = float(np.percentile(all_arr, 98))

    fig, axes = plt.subplots(
        n_rows, n_per_row,
        figsize=(n_per_row * 3.5, n_rows * 3.0),
    )
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for row, (glabel, specs) in enumerate(groups.items()):
        row_scores = (scores or {}).get(glabel, [None] * len(specs))
        for col in range(n_per_row):
            ax = axes[row, col]
            if col < len(specs):
                ax.imshow(np.asarray(specs[col]), aspect="auto", origin="lower",
                          vmin=vmin, vmax=vmax, cmap=cmap)
                sc = row_scores[col] if col < len(row_scores) else None
                subtitle = f"sc={sc:.3f}" if sc is not None else ""
                ax.set_title(subtitle, fontsize=8)
            else:
                ax.axis("off")
            ax.set_xticks([])
            ax.set_yticks([])
        axes[row, 0].set_ylabel(glabel, fontsize=9, labelpad=4)

    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Detector heatmap
# ---------------------------------------------------------------------------

def plot_detector_heatmap(
    score_matrix: np.ndarray,
    detector_names: list[str],
    row_labels: list[str],
    *,
    vmin: float = 0.80,
    vmax: float = 1.0,
    title: str = "Per-Detector Rank-Normalised Scores",
    divider_row: int | None = None,
    divider_label_top: str = "",
    divider_label_bottom: str = "",
) -> "matplotlib.figure.Figure":
    """Heatmap of rank-normalised detector scores (rows = samples, cols = detectors).

    Parameters
    ----------
    score_matrix : (N_samples, N_detectors) array
        Rank-normalised scores in [0, 1].
    detector_names : list of str
        Column labels.
    row_labels : list of str
        Row labels (e.g. ``"[✓] idx=42  ens=0.98"``).
    divider_row : int, optional
        Draw a white horizontal line after this row (e.g. stealth/detected boundary).
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(
        figsize=(len(detector_names) * 1.3, len(row_labels) * 0.65 + 1.5)
    )
    im = ax.imshow(score_matrix, aspect="auto", cmap="RdYlGn", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(detector_names)))
    ax.set_xticklabels(detector_names, rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=9)
    plt.colorbar(im, ax=ax, label="Rank-normalised score")

    if divider_row is not None and 0 < divider_row < len(row_labels):
        ax.axhline(divider_row - 0.5, color="white", lw=2, ls="--")
        mid_x = len(detector_names) / 2
        if divider_label_top:
            ax.text(mid_x, divider_row / 2 - 0.5, divider_label_top,
                    ha="center", va="center", color="white",
                    fontsize=10, fontweight="bold")
        if divider_label_bottom:
            ax.text(mid_x, divider_row + (len(row_labels) - divider_row) / 2 - 0.5,
                    divider_label_bottom,
                    ha="center", va="center", color="white",
                    fontsize=10, fontweight="bold")

    ax.set_title(title, fontweight="bold")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# FP cluster time-series
# ---------------------------------------------------------------------------

def plot_fp_cluster_timeseries(
    bg_scores: np.ndarray,
    threshold: float,
    fp_positions: np.ndarray,
    cluster_window: tuple[int, int] | None = None,
    *,
    title: str = "Ensemble Score — Background Eval",
    xlabel: str = "Background eval sample index",
) -> "matplotlib.figure.Figure":
    """Plot background eval scores with FP positions and cluster window marked.

    Parameters
    ----------
    bg_scores : (N_bg,) array
        Ensemble scores for background eval samples.
    threshold : float
        Detection threshold.
    fp_positions : (N_fp,) int array
        Positions within ``bg_scores`` that are false positives.
    cluster_window : (start, end), optional
        Contiguous FP cluster; drawn as an orange span.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(14, 4))
    t = np.arange(len(bg_scores))
    ax.plot(t, bg_scores, color="steelblue", lw=0.7, alpha=0.8, label="BG eval score")
    ax.axhline(threshold, color="red", lw=1.5, ls="--",
               label=f"Threshold ({threshold:.3f})")

    if cluster_window is not None:
        ax.axvspan(cluster_window[0], cluster_window[1],
                   alpha=0.2, color="orange",
                   label=f"Main FP cluster ({cluster_window[1]-cluster_window[0]+1} samples)")

    ax.scatter(fp_positions, bg_scores[fp_positions],
               color="red", s=25, zorder=5, label=f"FPs ({len(fp_positions)})")

    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel("Ensemble score", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Ensemble weight bar chart
# ---------------------------------------------------------------------------

def plot_weight_bar(
    weight_df: "pandas.DataFrame",
    *,
    detector_col: str = "detector",
    weight_col: str = "weight",
    title: str = "Ensemble Weights",
) -> "matplotlib.figure.Figure":
    """Horizontal bar chart of ensemble detector weights.

    Parameters
    ----------
    weight_df : DataFrame with columns ``detector_col`` and ``weight_col``
        Sorted descending by weight.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, max(3, len(weight_df) * 0.5)))
    bars = ax.barh(weight_df[detector_col], weight_df[weight_col],
                   color="steelblue", edgecolor="k", linewidth=0.5)
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
    ax.set_xlabel("Weight", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.invert_yaxis()
    fig.tight_layout()
    return fig
