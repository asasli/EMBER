#!/usr/bin/env python
"""Create spectrograms from PSP FIELDS DVAC burst data.

Loads a DVAC CDF file, computes log-scaled spectrograms for each burst,
and saves the resulting plots to an output directory.
"""

import argparse
import os

import cdflib
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal


def create_spectrograms(cdf_path, output_dir, probes="12", fmt="both"):
    """Generate spectrograms from a DVAC CDF file.

    Parameters
    ----------
    cdf_path : str
        Path to the DVAC CDF file.
    output_dir : str
        Directory to save spectrogram plots.
    probes : str
        Antenna pair to use, either '12' or '34'.
    fmt : str
        Output format: 'png', 'npz', or 'both'.
    """
    os.makedirs(output_dir, exist_ok=True)

    data = cdflib.CDF(cdf_path)

    var_name = f"psp_fld_l2_dfb_dbm_dvac{probes}"
    dvac = data.varget(var_name)

    time_tt2000 = data.varget("psp_fld_l2_dfb_dbm_dvac_time_series_TT2000")
    time_utc = [cdflib.cdfepoch.to_datetime(t) for t in time_tt2000]

    print(f"Found {len(dvac)} bursts with {dvac.shape[1]} samples each")

    for i in range(len(dvac)):
        burst_data = dvac[i]
        burst_time = time_utc[i]

        # Calculate sampling frequency from timestamps
        timedelta = burst_time[1] - burst_time[0]
        time_step_ns = timedelta.astype(float)
        fs = 1.0 / (time_step_ns * 1e-9)

        # Compute spectrogram
        f, t, Sxx = signal.spectrogram(
            burst_data, fs, window="hann", nperseg=1024, scaling="spectrum", axis=0
        )

        # Remove 0 Hz bin and apply log scaling
        f_log = np.log10(f[1:])
        Sxx_log = np.log10(Sxx[1:, :])

        base = os.path.join(output_dir, f"dvac{probes}_burst_{i:03d}")

        if fmt in ("npz", "both"):
            np.savez_compressed(
                f"{base}.npz",
                f_log=f_log,
                Sxx_log=Sxx_log,
                t=t,
                fs=np.array(fs),
                timestamp=str(burst_time[0]),
            )
            print(f"Saved {base}.npz")

        if fmt in ("png", "both"):
            fig, ax = plt.subplots(figsize=(20, 10))
            ax.pcolormesh(range(len(t)), f_log, Sxx_log, cmap="jet", shading="auto")
            ax.set_ylabel("Log Frequency [Hz]")
            ax.set_xlabel("Time Bins")
            ax.set_title(f"DVAC{probes} Burst {i} — {burst_time[0]}")
            fig.savefig(f"{base}.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved {base}.png")


def main():
    parser = argparse.ArgumentParser(
        description="Create spectrograms from PSP FIELDS DVAC burst data."
    )
    parser.add_argument("cdf_file", help="Path to the DVAC CDF file")
    parser.add_argument(
        "--output", "-o", default="output/dvac", help="Output directory (default: output/dvac)"
    )
    parser.add_argument(
        "--probes", "-p", default="12", choices=["12", "34"],
        help="Antenna pair to use (default: 12)"
    )
    parser.add_argument(
        "--format", "-f", default="both", choices=["png", "npz", "both"],
        help="Output format (default: both)"
    )
    args = parser.parse_args()

    create_spectrograms(args.cdf_file, args.output, args.probes, args.format)


if __name__ == "__main__":
    main()
