"""Plotting helpers for the Solar-ADv2 repo notebook and README figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd

try:
    import seaborn as sns
except ImportError:  # pragma: no cover - optional styling dependency
    sns = None

from ember.datasets import (
    extract_label_examples,
    load_labeled_spectrogram_dataframe,
    spectrogram_from_row,
)


LABEL_TITLES = {
    0: "Noise (Label=0)",
    1: "Anomaly-1 (Label=1)",
    2: "Anomaly-2 (Label=2)",
}

LABEL_COLORS = {
    0: "#3498db",
    1: "#f39c12",
    2: "#e74c3c",
}

LABEL_MARKERS = {
    0: "o",
    1: "s",
    2: "^",
}


def load_case_summary(path: str | Path) -> pd.DataFrame:
    """Load the exported detector-combination case summary table."""

    return pd.read_csv(Path(path))


def load_detection_votes(path: str | Path) -> pd.DataFrame:
    """Load the exported recovered-anomaly vote table."""

    return pd.read_csv(Path(path))


def plot_three_class_examples(
    df: pd.DataFrame,
    *,
    output_path: str | Path | None = None,
    cmap: str = "inferno",
):
    """Plot one labeled spectrogram example for each class."""

    examples = extract_label_examples(df)
    fig, axes = plt.subplots(1, len(examples), figsize=(15, 4), constrained_layout=True)
    for ax, (label, row) in zip(axes, examples):
        ax.imshow(spectrogram_from_row(row), aspect="auto", origin="lower", cmap=cmap)
        ax.set_title(
            LABEL_TITLES.get(label, f"Label {label}"), fontsize=12, fontweight="bold"
        )
        ax.set_xlabel("Time Bins")
        ax.set_ylabel("Log Frequency Bins")
        _t = row.get("Time (UTC)", None)
        time_utc = _t[0] if _t is not None and len(_t) > 0 else "n/a"
        ax.text(
            0.01,
            0.01,
            str(time_utc),
            transform=ax.transAxes,
            color="white",
            fontsize=8,
            ha="left",
            va="bottom",
            bbox={"facecolor": "black", "alpha": 0.45, "pad": 3},
        )
    fig.suptitle(
        "Representative PSP spectrogram examples", fontsize=16, fontweight="bold"
    )
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
    return fig, axes


def plot_case_recovery_summary(
    case_summary_df: pd.DataFrame,
    *,
    output_path: str | Path | None = None,
):
    """Plot the recovered anomalies by detector-combination case."""

    df = case_summary_df.sort_values(
        ["recovered", "coverage"], ascending=[True, True]
    ).copy()
    fig, ax = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
    if sns is not None:
        palette = sns.color_palette("crest", n_colors=len(df))
    else:
        palette = plt.cm.viridis(np.linspace(0.2, 0.85, max(len(df), 1)))
    ax.barh(df["case"], df["recovered"], color=palette)
    for y, (_, row) in enumerate(df.iterrows()):
        ax.text(
            row["recovered"] + 0.15,
            y,
            f"{row['coverage']:.1%} | fp={int(row['union_fp'])}",
            va="center",
            fontsize=11,
        )
    if "total_anomalies" in df.columns:
        total = int(df["total_anomalies"].iloc[0])
        ax.axvline(total, color="0.45", linestyle="--", linewidth=1.5)
        ax.set_xlim(0, total + 2)
    ax.set_title("Recovered anomalies by detector-combination case", fontweight="bold")
    ax.set_xlabel("Unique anomalies recovered")
    ax.set_ylabel("")
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
    return fig, ax


def build_detection_matrix(
    votes_df: pd.DataFrame,
    *,
    method_order: list[str] | None = None,
) -> pd.DataFrame:
    """Convert the recovered-anomaly vote table into a binary method/sample matrix."""

    votes_df = votes_df.copy()
    votes_df["methods"] = votes_df["methods"].fillna("")

    if method_order is None:
        seen: list[str] = []
        for methods in votes_df["methods"]:
            for method in [part.strip() for part in methods.split(",") if part.strip()]:
                if method not in seen:
                    seen.append(method)
        method_order = seen

    matrix = pd.DataFrame(
        0, index=method_order, columns=votes_df["sample_idx"].tolist(), dtype=int
    )
    for _, row in votes_df.iterrows():
        sample_idx = row["sample_idx"]
        for method in [
            part.strip() for part in row["methods"].split(",") if part.strip()
        ]:
            if method in matrix.index:
                matrix.loc[method, sample_idx] = 1
    return matrix


def plot_detection_map(
    votes_df: pd.DataFrame,
    *,
    method_order: list[str] | None = None,
    output_path: str | Path | None = None,
):
    """Plot the recovered-anomaly detection map for the selected best case."""

    matrix = build_detection_matrix(votes_df, method_order=method_order)
    fig_width = max(12, 0.35 * max(1, matrix.shape[1]))
    fig, ax = plt.subplots(figsize=(fig_width, 3.8), constrained_layout=True)
    if sns is not None:
        sns.heatmap(
            matrix,
            cmap=ListedColormap(["#f8fafc", "#115e59"]),
            cbar=False,
            linewidths=0.5,
            linecolor="white",
            ax=ax,
        )
    else:
        ax.imshow(
            matrix.to_numpy(dtype=float),
            aspect="auto",
            cmap=ListedColormap(["#f8fafc", "#115e59"]),
        )
        ax.set_xticks(np.arange(matrix.shape[1]))
        ax.set_xticklabels(matrix.columns.tolist())
        ax.set_yticks(np.arange(matrix.shape[0]))
        ax.set_yticklabels(matrix.index.tolist())
    ax.set_title("Recovered-anomaly detection map", fontweight="bold")
    ax.set_xlabel("Recovered anomaly sample index")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=90, labelsize=9)
    ax.tick_params(axis="y", labelsize=11)
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
    return fig, ax


def plot_labeled_projection(
    embedding_2d: np.ndarray,
    labels: np.ndarray,
    *,
    ax=None,
    output_path: str | Path | None = None,
    title: str = "Feature projection",
    label_names: dict[int, str] | None = None,
):
    """Plot a 2D embedding colored by the three anomaly labels."""

    coords = np.asarray(embedding_2d, dtype=float)
    labels = np.asarray(labels, dtype=int)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("embedding_2d must have shape (n_samples, 2).")

    if label_names is None:
        label_names = {
            0: "Noise",
            1: "Anomaly-1",
            2: "Anomaly-2",
        }

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
    else:
        fig = ax.figure

    for label in sorted(np.unique(labels)):
        mask = labels == label
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            c=LABEL_COLORS.get(int(label), "#64748b"),
            marker=LABEL_MARKERS.get(int(label), "o"),
            s=95,
            alpha=0.75,
            edgecolors="white",
            linewidths=0.7,
            label=f"{label_names.get(int(label), f'Label {label}')} (n={int(mask.sum())})",
        )

    ax.legend(fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig, ax


def plot_feature_discrimination(
    summary_df: pd.DataFrame,
    *,
    top_k: int = 20,
    output_path: str | Path | None = None,
):
    """Plot the top feature effect sizes from a discrimination summary table."""

    top_df = summary_df.head(top_k).copy().iloc[::-1]
    colors = ["#e74c3c" if bool(flag) else "#95a5a6" for flag in top_df["significant"]]

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    ax.barh(top_df["feature_name"], top_df["effect_size"], color=colors, alpha=0.85)
    ax.set_xlabel("Effect Size (|rank-biserial correlation|)")
    ax.set_ylabel("")
    ax.set_title(
        f"Top {min(top_k, len(summary_df))} Physics Features", fontweight="bold"
    )

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig, ax


def save_repo_figures(
    *,
    dataset_path: str | Path,
    case_summary_path: str | Path,
    detection_votes_path: str | Path,
    output_dir: str | Path,
    method_order: list[str] | None = None,
) -> dict[str, str]:
    """Generate the three repo figures used in the README and overview notebook."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_df = load_labeled_spectrogram_dataframe(dataset_path)
    case_summary_df = load_case_summary(case_summary_path)
    votes_df = load_detection_votes(detection_votes_path)

    class_examples_path = output_dir / "repo_three_class_examples.png"
    recovery_path = output_dir / "repo_recovered_by_case.png"
    detection_map_path = output_dir / "repo_detection_map.png"

    plot_three_class_examples(dataset_df, output_path=class_examples_path)
    plt.close("all")
    plot_case_recovery_summary(case_summary_df, output_path=recovery_path)
    plt.close("all")
    plot_detection_map(
        votes_df, method_order=method_order, output_path=detection_map_path
    )
    plt.close("all")

    return {
        "class_examples": str(class_examples_path),
        "recovered_by_case": str(recovery_path),
        "detection_map": str(detection_map_path),
    }
