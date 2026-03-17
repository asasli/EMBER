#!/usr/bin/env python
"""Create spectrograms from PSP FIELDS VAC burst data.

Loads a VAC CDF file, computes differential voltages between antenna pairs,
generates log-scaled spectrograms for each burst, and saves the plots.
"""

import argparse
import os

import cdflib
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal


def create_spectrograms(cdf_path, output_dir, probes="12", fmt="both"):
    """Generate spectrograms from a VAC CDF file.

    Parameters
    ----------
    cdf_path : str
        Path to the VAC CDF file.
    output_dir : str
        Directory to save spectrogram plots.
    probes : str
        Antenna pair to use, either '12' or '34'.
    fmt : str
        Output format: 'png', 'npz', or 'both'.
    """
    os.makedirs(output_dir, exist_ok=True)

    data = cdflib.CDF(cdf_path)

    # Load individual antenna voltages and compute differential
    if probes == "12":
        vac_a = np.array(data.varget("psp_fld_l2_dfb_dbm_vac1"))
        vac_b = np.array(data.varget("psp_fld_l2_dfb_dbm_vac2"))
    else:
        vac_a = np.array(data.varget("psp_fld_l2_dfb_dbm_vac3"))
        vac_b = np.array(data.varget("psp_fld_l2_dfb_dbm_vac4"))

    diff_vac = vac_a - vac_b

    time_tt2000 = data.varget("psp_fld_l2_dfb_dbm_vac_time_series_TT2000")
    time_utc = [cdflib.cdfepoch.to_datetime(t) for t in time_tt2000]

    print(f"Found {len(diff_vac)} bursts with {diff_vac.shape[1]} samples each")

    for i in range(len(diff_vac)):
        burst_data = diff_vac[i]
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

        base = os.path.join(output_dir, f"vac{probes}_burst_{i:03d}")

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
            ax.set_title(f"VAC{probes} (differential) Burst {i} — {burst_time[0]}")
            fig.savefig(f"{base}.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved {base}.png")


def main():
    parser = argparse.ArgumentParser(
        description="Create spectrograms from PSP FIELDS VAC burst data."
    )
    parser.add_argument("cdf_file", help="Path to the VAC CDF file")
    parser.add_argument(
        "--output", "-o", default="output/vac", help="Output directory (default: output/vac)"
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
