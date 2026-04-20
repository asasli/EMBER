"""Command-line entry points for the EMBER package."""

from __future__ import annotations

import argparse

from ember.download import HOURS, download_products
from ember.spectrograms import create_dvac_spectrograms, create_vac_spectrograms


def _download_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download PSP FIELDS DBM burst data from NASA CDAWeb."
    )
    parser.add_argument("--kind", required=True, choices=["dvac", "vac"], help="Data product type")
    parser.add_argument("--output", "-o", default="data", help="Output directory (default: data)")

    single = parser.add_argument_group("single file")
    single.add_argument("--date", type=str, help="Date to download (YYYY-MM-DD)")
    single.add_argument("--hour", type=str, choices=HOURS, help="6-hour block to download")
    single.add_argument("--all-hours", action="store_true", help="Download all 6-hour blocks for the given date")

    range_group = parser.add_argument_group("date range")
    range_group.add_argument("--start", type=str, help="Start date for range download (YYYY-MM-DD)")
    range_group.add_argument("--end", type=str, help="End date for range download (YYYY-MM-DD)")
    return parser


def _spectrogram_parser(kind: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=f"Create spectrograms from PSP FIELDS {kind.upper()} burst data."
    )
    parser.add_argument("cdf_file", help=f"Path to the {kind.upper()} CDF file")
    parser.add_argument(
        "--output",
        "-o",
        default=f"output/{kind}",
        help=f"Output directory (default: output/{kind})",
    )
    parser.add_argument(
        "--probes",
        "-p",
        default="12",
        choices=["12", "34"],
        help="Antenna pair to use (default: 12)",
    )
    parser.add_argument(
        "--format",
        "-f",
        default="both",
        choices=["png", "npz", "both"],
        help="Output format (default: both)",
    )
    return parser


def main_download() -> None:
    args = _download_parser().parse_args()
    summary = download_products(
        args.kind,
        output=args.output,
        date=args.date,
        hour=args.hour,
        all_hours=args.all_hours,
        start=args.start,
        end=args.end,
    )
    print(
        f"\nDone: {summary['downloaded']} downloaded, "
        f"{summary['skipped']} skipped, {summary['failed']} failed"
    )


def main_spectrograms_dvac() -> None:
    args = _spectrogram_parser("dvac").parse_args()
    saved = create_dvac_spectrograms(
        args.cdf_file,
        args.output,
        probes=args.probes,
        fmt=args.format,
    )
    print(f"Saved {len(saved)} files")


def main_spectrograms_vac() -> None:
    args = _spectrogram_parser("vac").parse_args()
    saved = create_vac_spectrograms(
        args.cdf_file,
        args.output,
        probes=args.probes,
        fmt=args.format,
    )
    print(f"Saved {len(saved)} files")


def _anomaly_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the EMBER robust anomaly detection pipeline on a labeled "
            "spectrogram dataset (pickle file produced by prepare_anomaly_dataset)."
        )
    )
    parser.add_argument("dataset", help="Path to labeled spectrogram pickle (.pkl)")
    parser.add_argument(
        "--target-far",
        type=float,
        default=0.01,
        metavar="FAR",
        help="False alarm rate budget (default: 0.01 = 1%%)",
    )
    parser.add_argument(
        "--patchcore",
        action="store_true",
        default=False,
        help="Include PatchCore detector (requires torchvision)",
    )
    parser.add_argument(
        "--no-vae",
        dest="vae",
        action="store_false",
        default=True,
        help="Disable the β-VAE detector (faster, requires less memory)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="./robust_ad_results",
        metavar="DIR",
        help="Directory for CSV summaries and PNG plots (default: ./robust_ad_results)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output")
    return parser


def main_anomaly() -> None:
    """Entry point for the ``ember-anomaly`` CLI command."""
    import pickle
    from pathlib import Path

    args = _anomaly_parser().parse_args()

    print(f"Loading dataset from {args.dataset} …")
    with open(args.dataset, "rb") as fh:
        data = pickle.load(fh)

    # Accept either a raw DataFrame or a pre-built AnomalyDataset
    if hasattr(data, "noise_idx"):
        dataset = data
    else:
        import pandas as pd
        from ember.anomaly.pipeline import prepare_anomaly_dataset
        if not isinstance(data, pd.DataFrame):
            raise ValueError(
                f"Expected a DataFrame or AnomalyDataset in {args.dataset!r}, "
                f"got {type(data).__name__}."
            )
        dataset = prepare_anomaly_dataset(data)

    from ember.anomaly.pipeline import run_robust_anomaly_pipeline
    results = run_robust_anomaly_pipeline(
        dataset,
        target_far=args.target_far,
        include_patchcore=args.patchcore,
        include_vae=args.vae,
        seed=args.seed,
        output_dir=Path(args.output_dir),
        verbose=not args.quiet,
    )

    print("\nResults summary:")
    print(results["results_df"].to_string(index=False))
