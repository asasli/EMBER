# EMBER

ML for Core Electron Heating due to Wave Modulation in the Solar Wind

## Overview

EMBER uses machine learning to study core electron heating mechanisms driven by coupled electromagnetic wave modulation in the solar wind. The project processes data from NASA's Parker Solar Probe (PSP) FIELDS instrument, transforming burst-mode electric field measurements into spectrograms suitable for training ML models to detect coupled waves.

## Data

Parker Solar Probe FIELDS DBM (Digital Burst Memory) burst data comes in two forms:

- **DVAC**: Differential voltage between two opposing antennas (antenna pairs 1-2 or 3-4). This is the preferred data product as it is already processed into differential form.
- **VAC**: Single-ended voltages from individual antennas (vac1, vac2, vac3, vac4). VAC data must be converted to differential voltage before use (e.g., `dvac12 = vac1 - vac2`).

PSP has four antennas mounted at the corners of its heat shield, arranged diagonally — antennas 1 and 2 are opposite each other, as are antennas 3 and 4.

### Data Sources

- Berkeley SSL: https://research.ssl.berkeley.edu/data/psp/data/sci/fields/l2/
- NASA CDAWeb: https://cdaweb.gsfc.nasa.gov/sp_phys/data/psp/fields/l2/

Data is stored in CDF (Common Data Format) files. Each file contains multiple burst datasets, typically with 524,288 samples per burst.

## Processing Pipeline

1. Load CDF files using `cdflib`
2. Extract voltage data (DVAC directly, or compute differential from VAC)
3. Convert TT2000 timestamps to UTC
4. Compute spectrograms using FFT (1024-point Hann window)
5. Apply logarithmic scaling to both frequency and power axes — this is critical for observing coupled wave structures
6. Label and package spectrograms for ML training

## Scripts

- `scripts/download_data.py` — Download CDF files from NASA CDAWeb for specified dates and data products
- `scripts/create_spectrograms_dvac.py` — Process DVAC CDF files into spectrograms
- `scripts/create_spectrograms_vac.py` — Process VAC CDF files into spectrograms (computes differential voltage from single-ended measurements)

## Installation

```bash
pip install .
```

For development (includes test dependencies):

```bash
pip install .[test]
```

### Dependencies

- `cdflib` — Reading CDF files
- `numpy` — Numerical operations
- `scipy` — Spectrogram computation
- `matplotlib` — Visualization

## Usage

```bash
# Download a single DVAC file (6-hour block starting at 06 UTC)
python scripts/download_data.py --kind dvac --date 2020-09-25 --hour 06

# Download all 6-hour blocks for a day
python scripts/download_data.py --kind vac --date 2021-08-08 --all-hours

# Download a date range
python scripts/download_data.py --kind dvac --start 2020-09-25 --end 2020-09-27

# Generate spectrograms from DVAC data
python scripts/create_spectrograms_dvac.py data/psp_fld_l2_dfb_dbm_dvac_2020092506_v02.cdf --output output/dvac/

# Generate spectrograms from VAC data
python scripts/create_spectrograms_vac.py data/psp_fld_l2_dfb_dbm_vac_2021080818_v02.cdf --output output/vac/
```
