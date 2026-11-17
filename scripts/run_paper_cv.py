#!/usr/bin/env python
"""Reproduce the class-stratified evaluation of the EMBER paper from public data.

The released catalog (catalog/ember_catalog_v1.csv) lists, for every burst,
its CDF file, burst index, UTC start, sampling rate and label. This script

1. downloads the needed FIELDS DBM CDF files from CDAWeb (``--download``) or reads
   them from ``--data-dir``;
2. recomputes every spectrogram with one chain (2048-point Hann STFT, 204-sample
   step, log10 power, frames averaged in groups of four);
3. extracts the 89-feature v2 bank and the PatchCore patch features;
4. within each sampling rate that has labeled background, runs the K-fold
   out-of-fold protocol: the flow (MAF, 3-seed bag), kNN and PCA on the feature
   bank and PatchCore are fitted on four fifths of the catalog background and
   score the held-out background and all events of that rate;
5. combines the flow and PatchCore into EMBER-U (minimum empirical p-value) and
   prints recovery at 1 %, 2 % and 5 % FAR with Clopper-Pearson intervals.

Usage:
    python scripts/run_paper_cv.py --data-dir data/cdf --download
    python scripts/run_paper_cv.py --data-dir data/cdf --k 5 --seed 42 --out results.npz
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ember.anomaly as ad  # noqa: E402
from ember.anomaly.crossval import out_of_fold_scores  # noqa: E402
from ember.anomaly.narrowband import extract_feature_bank_v2_matrix  # noqa: E402
from ember.anomaly.union import min_p_union, recovery_at_far  # noqa: E402
from ember.spectrograms import load_voltage_bursts  # noqa: E402

CDAWEB = "https://cdaweb.gsfc.nasa.gov/sp_phys/data/psp/fields/l2/dfb_dbm_{kind}/{year}/{name}"
RATE_NAME = {18750: "18.75 kS/s", 37500: "37.5 kS/s", 150000: "150 kS/s"}


def kind_of(name: str) -> str:
    return "dvac" if "_dvac_" in name else "vac"


def download(names, data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    for name in sorted(names):
        path = data_dir / name
        if path.exists() and path.stat().st_size > 0:
            continue
        year = name.split("_")[-2][:4]
        url = CDAWEB.format(kind=kind_of(name), year=year, name=name)
        print(f"downloading {name}", flush=True)
        urllib.request.urlretrieve(url, path)


def spectrogram(x: np.ndarray, fs: float, time_bin: int) -> np.ndarray:
    _, _, sxx = signal.spectrogram(
        x, fs, window="hann", nperseg=2048, noverlap=2048 - 204, scaling="spectrum"
    )
    spec = np.log10(np.maximum(sxx[1:], np.finfo(float).tiny)).astype(np.float32)
    t = (spec.shape[1] // time_bin) * time_bin
    return spec[:, :t].reshape(spec.shape[0], -1, time_bin).mean(-1)


def load_catalog_spectrograms(
    cat: pd.DataFrame, data_dir: Path, time_bin: int
) -> list[np.ndarray]:
    specs: list[np.ndarray | None] = [None] * len(cat)
    for name, rows in cat.groupby("cdf_file"):
        bursts, times = load_voltage_bursts(
            data_dir / name, kind=kind_of(name), probes="12"
        )
        for i, r in rows.iterrows():
            tt = times[int(r.burst_index)]
            fs = 1e9 / float(
                np.asarray(tt[1] - tt[0], dtype="timedelta64[ns]").astype(np.int64)
            )
            specs[i] = spectrogram(
                np.asarray(bursts[int(r.burst_index)], float), fs, time_bin
            )
        print(f"  {name}: {len(rows)} catalog bursts", flush=True)
    return specs


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--catalog",
        default=str(
            Path(__file__).resolve().parents[1] / "catalog/ember_catalog_v1.csv"
        ),
    )
    p.add_argument(
        "--data-dir",
        required=True,
        help="folder holding (or receiving) the DBM CDF files",
    )
    p.add_argument(
        "--download", action="store_true", help="download missing CDF files from CDAWeb"
    )
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--time-bin", type=int, default=4)
    p.add_argument(
        "--min-background",
        type=int,
        default=100,
        help="skip sampling rates with fewer labeled backgrounds",
    )
    p.add_argument("--device", default=None)
    p.add_argument(
        "--out",
        default=None,
        help="optional .npz with the out-of-fold scores per sampling rate",
    )
    args = p.parse_args()

    from sklearn.decomposition import PCA
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler
    import torch

    cat = pd.read_csv(args.catalog)
    cat = cat[cat.cdf_file.notna() & (cat.cdf_file != "")].reset_index(drop=True)
    data_dir = Path(args.data_dir)
    if args.download:
        download(set(cat.cdf_file), data_dir)

    print(
        f"{len(cat)} catalog bursts with a public timestamp; recomputing spectrograms …",
        flush=True,
    )
    specs = load_catalog_spectrograms(cat, data_dir, args.time_bin)
    feats = extract_feature_bank_v2_matrix(specs)
    patches = ad.PatchCoreDetector(
        pool_size=4, knn_k=5, device=args.device
    )._extract_patch_features(specs)

    def fit_and_score(train_idx: np.ndarray) -> dict[str, np.ndarray]:
        sc = StandardScaler().fit(feats[train_idx])
        xtr, xall = sc.transform(feats[train_idx]), sc.transform(feats)
        maf = []
        for s in range(3):
            torch.manual_seed(args.seed + s)
            maf.append(
                ad.flow_score(
                    ad.fit_zuko_maf(xtr, epochs=200, device=args.device),
                    xall,
                    device=args.device,
                )
            )
        d, _ = NearestNeighbors(n_neighbors=10).fit(xtr).kneighbors(xall)
        pca = PCA(n_components=min(30, xtr.shape[1] - 1), random_state=args.seed).fit(
            xtr
        )
        nn = NearestNeighbors(n_neighbors=5).fit(
            patches[train_idx].reshape(-1, patches.shape[-1])
        )
        n, pp, c = patches.shape
        dp, _ = nn.kneighbors(patches.reshape(n * pp, c))
        return {
            "MAF": np.mean(maf, axis=0),
            "kNN": d.mean(1),
            "PCA": ((xall - pca.inverse_transform(pca.transform(xall))) ** 2).mean(1),
            "PatchCore": dp.mean(1).reshape(n, pp).max(1),
        }

    saved = {}
    for rate, sub in cat.groupby("sampling_rate_Hz"):
        bg = sub.index[sub.label == 0].to_numpy()
        ev = sub.index[sub.label > 0].to_numpy()
        name = RATE_NAME.get(int(rate), f"{rate} S/s")
        if len(bg) < args.min_background:
            print(
                f"\n{name}: {len(bg)} labeled backgrounds, {len(ev)} events: no evaluation (too little background)"
            )
            continue
        print(f"\n=== {name}: {len(bg)} backgrounds, {len(ev)} events ===", flush=True)
        oof = out_of_fold_scores(
            fit_and_score, bg, ev, k=args.k, seed=args.seed, verbose=True
        )
        labels = cat.label.to_numpy()[ev]
        for det in oof.names:
            r1 = recovery_at_far(oof.background[det], oof.events[det], 0.01)
            r5 = recovery_at_far(oof.background[det], oof.events[det], 0.05)
            print(f"  {det:10s} TPR@1% = {r1['tpr']:.3f}   TPR@5% = {r5['tpr']:.3f}")
        ref = {m: oof.background[m] for m in ("MAF", "PatchCore")}
        u_bg = min_p_union(ref, ref)
        u_ev = min_p_union(ref, {m: oof.events[m] for m in ref})
        for far in (0.01, 0.02, 0.05):
            r = recovery_at_far(u_bg, u_ev, far, event_labels=labels)
            per_label = "  ".join(
                f"L{lab}={r[f'tpr_label{lab}']:.2f}"
                for lab in (1, 2)
                if f"tpr_label{lab}" in r
            )
            print(
                f"  EMBER-U @ {far:.0%}: FP {r['fp']}/{len(u_bg)}  TPR = {r['tpr']:.3f} "
                f"[{r['tpr_ci'][0]:.2f}, {r['tpr_ci'][1]:.2f}]  {per_label}"
            )
        saved[int(rate)] = oof
    if args.out:
        arrays = {}
        for rate, oof in saved.items():
            for det in oof.names:
                arrays[f"{rate}_bg_{det}"] = oof.background[det]
                arrays[f"{rate}_ev_{det}"] = oof.events[det]
            arrays[f"{rate}_bg_ids"] = cat.catalog_id.to_numpy()[oof.background_idx]
            arrays[f"{rate}_ev_ids"] = cat.catalog_id.to_numpy()[oof.event_idx]
        np.savez(args.out, **arrays)
        print("saved", args.out)


if __name__ == "__main__":
    main()
