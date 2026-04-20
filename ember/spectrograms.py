"""Spectrogram utilities shared by the EMBER CLI and notebooks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cdflib
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal


@dataclass
class SpectrogramRecord:
    """In-memory spectrogram representation for one burst."""

    name: str
    f_log: np.ndarray
    sxx_log: np.ndarray
    t: np.ndarray
    fs: float
    timestamp: str


def _tt2000_to_utc_grid(time_tt2000: np.ndarray) -> list[np.ndarray]:
    return [cdflib.cdfepoch.to_datetime(t) for t in time_tt2000]


def _time_delta_ns(delta) -> float:
    if hasattr(delta, "total_seconds"):
        return float(delta.total_seconds()) * 1e9
    try:
        return float(np.asarray(delta, dtype="timedelta64[ns]").astype(np.int64))
    except Exception:
        return float(delta.astype(float))


def _sampling_frequency(burst_time: np.ndarray) -> float:
    return 1.0 / (_time_delta_ns(burst_time[1] - burst_time[0]) * 1e-9)


def load_voltage_bursts(
    cdf_path: str | Path,
    *,
    kind: str = "dvac",
    probes: str = "12",
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Load burst voltage traces and timestamps from a DVAC or VAC CDF file."""

    cdf_path = Path(cdf_path)
    data = cdflib.CDF(str(cdf_path))

    if kind == "dvac":
        var_name = f"psp_fld_l2_dfb_dbm_dvac{probes}"
        bursts = np.array(data.varget(var_name))
        time_tt2000 = np.array(data.varget("psp_fld_l2_dfb_dbm_dvac_time_series_TT2000"))
    elif kind == "vac":
        if probes == "12":
            vac_a = np.array(data.varget("psp_fld_l2_dfb_dbm_vac1"))
            vac_b = np.array(data.varget("psp_fld_l2_dfb_dbm_vac2"))
        else:
            vac_a = np.array(data.varget("psp_fld_l2_dfb_dbm_vac3"))
            vac_b = np.array(data.varget("psp_fld_l2_dfb_dbm_vac4"))
        bursts = vac_a - vac_b
        time_tt2000 = np.array(data.varget("psp_fld_l2_dfb_dbm_vac_time_series_TT2000"))
    else:
        raise ValueError("kind must be either 'dvac' or 'vac'")

    return bursts, _tt2000_to_utc_grid(time_tt2000)


def compute_spectrogram_record(
    burst_data: np.ndarray,
    burst_time: np.ndarray,
    *,
    name: str,
    nperseg: int = 1024,
    window: str = "hann",
) -> SpectrogramRecord:
    """Compute the EMBER-style log-frequency spectrogram for one burst."""

    fs = _sampling_frequency(burst_time)
    f, t, sxx = signal.spectrogram(
        burst_data,
        fs,
        window=window,
        nperseg=nperseg,
        scaling="spectrum",
        axis=0,
    )
    f_log = np.log10(f[1:])
    sxx_log = np.log10(np.maximum(sxx[1:, :], np.finfo(float).tiny))
    return SpectrogramRecord(
        name=name,
        f_log=f_log,
        sxx_log=sxx_log,
        t=t,
        fs=float(fs),
        timestamp=str(burst_time[0]),
    )


def iter_spectrogram_records(
    cdf_path: str | Path,
    *,
    kind: str = "dvac",
    probes: str = "12",
    nperseg: int = 1024,
    window: str = "hann",
) -> Iterable[SpectrogramRecord]:
    """Yield one spectrogram record per burst from a CDF file."""

    bursts, times = load_voltage_bursts(cdf_path, kind=kind, probes=probes)
    for burst_index, (burst_data, burst_time) in enumerate(zip(bursts, times)):
        name = f"{kind}{probes}_burst_{burst_index:03d}"
        yield compute_spectrogram_record(
            burst_data,
            burst_time,
            name=name,
            nperseg=nperseg,
            window=window,
        )


def plot_spectrogram(
    record: SpectrogramRecord,
    *,
    ax=None,
    cmap: str = "inferno",
    title: str | None = None,
):
    """Plot a spectrogram record and return the matplotlib axis."""

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))
    ax.pcolormesh(
        np.arange(len(record.t)),
        record.f_log,
        record.sxx_log,
        cmap=cmap,
        shading="auto",
    )
    ax.set_ylabel("Log Frequency [Hz]")
    ax.set_xlabel("Time Bins")
    ax.set_title(title or record.name.replace("_", " "))
    return ax


def save_spectrogram_record(
    record: SpectrogramRecord,
    output_dir: str | Path,
    *,
    fmt: str = "both",
    cmap: str = "jet",
) -> list[Path]:
    """Save a spectrogram record to disk as PNG and/or NPZ."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base = output_dir / record.name
    saved_paths: list[Path] = []

    if fmt in ("npz", "both"):
        np.savez_compressed(
            base.with_suffix(".npz"),
            f_log=record.f_log,
            Sxx_log=record.sxx_log,
            t=record.t,
            fs=np.array(record.fs),
            timestamp=record.timestamp,
        )
        saved_paths.append(base.with_suffix(".npz"))

    if fmt in ("png", "both"):
        fig, ax = plt.subplots(figsize=(20, 10))
        plot_spectrogram(record, ax=ax, cmap=cmap, title=f"{record.name} — {record.timestamp}")
        fig.savefig(base.with_suffix(".png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(base.with_suffix(".png"))

    return saved_paths


def create_spectrograms(
    cdf_path: str | Path,
    output_dir: str | Path,
    *,
    kind: str = "dvac",
    probes: str = "12",
    fmt: str = "both",
    nperseg: int = 1024,
    window: str = "hann",
) -> list[Path]:
    """Create and save spectrograms for every burst in a CDF file."""

    saved_paths: list[Path] = []
    for record in iter_spectrogram_records(
        cdf_path,
        kind=kind,
        probes=probes,
        nperseg=nperseg,
        window=window,
    ):
        saved_paths.extend(save_spectrogram_record(record, output_dir, fmt=fmt))
    return saved_paths


def create_dvac_spectrograms(
    cdf_path: str | Path,
    output_dir: str | Path,
    *,
    probes: str = "12",
    fmt: str = "both",
) -> list[Path]:
    """Save spectrograms for a DVAC CDF file."""

    return create_spectrograms(cdf_path, output_dir, kind="dvac", probes=probes, fmt=fmt)


def create_vac_spectrograms(
    cdf_path: str | Path,
    output_dir: str | Path,
    *,
    probes: str = "12",
    fmt: str = "both",
) -> list[Path]:
    """Save spectrograms for a VAC CDF file."""

    return create_spectrograms(cdf_path, output_dir, kind="vac", probes=probes, fmt=fmt)


def load_saved_spectrogram(npz_path: str | Path) -> SpectrogramRecord:
    """Load a saved spectrogram NPZ file."""

    npz_path = Path(npz_path)
    with np.load(npz_path, allow_pickle=True) as data:
        return SpectrogramRecord(
            name=npz_path.stem,
            f_log=np.array(data["f_log"]),
            sxx_log=np.array(data["Sxx_log"]),
            t=np.array(data["t"]),
            fs=float(np.array(data["fs"]).item()),
            timestamp=str(np.array(data["timestamp"]).item()),
        )
