"""Dataset helpers for the labeled PSP anomaly-analysis artifacts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_labeled_spectrogram_dataframe(path: str | Path) -> pd.DataFrame:
    """Load the labeled spectrogram dataframe used by the anomaly notebooks."""

    return pd.read_pickle(Path(path))


def spectrogram_from_row(row: pd.Series) -> np.ndarray:
    """Extract the log-amplitude spectrogram array from a dataframe row."""

    return np.asarray(row["Log Amplitude"], dtype=np.float32)


def extract_label_examples(
    df: pd.DataFrame,
    *,
    labels: tuple[int, ...] = (0, 1, 2),
) -> list[tuple[int, pd.Series]]:
    """Return one representative dataframe row for each label in order."""

    examples: list[tuple[int, pd.Series]] = []
    for label in labels:
        match = df.loc[df["Label"] == label]
        if match.empty:
            raise ValueError(f"No examples found for label {label}.")
        examples.append((label, match.iloc[0]))
    return examples
