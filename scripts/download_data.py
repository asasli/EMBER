#!/usr/bin/env python
"""Download PSP FIELDS DBM burst data from NASA CDAWeb.

Downloads DVAC or VAC CDF files for specified dates. Data is organized
in 6-hour blocks (00, 06, 12, 18 UTC).

Examples
--------
    # Download a single DVAC file
    python scripts/download_data.py --kind dvac --date 2020-09-25 --hour 06

    # Download VAC data for a full day
    python scripts/download_data.py --kind vac --date 2021-08-08 --all-hours

    # Download a range of dates
    python scripts/download_data.py --kind dvac --start 2020-09-25 --end 2020-09-27
"""

import argparse
import os
import urllib.request
from datetime import datetime, timedelta

CDAWEB_URL = (
    "https://cdaweb.gsfc.nasa.gov/sp_phys/data/psp/fields/l2/"
    "dfb_dbm_{kind}/{year}/"
    "psp_fld_l2_dfb_dbm_{kind}_{year}{month}{day}{hour}_v02.cdf"
)

HOURS = ["00", "06", "12", "18"]


def build_url(kind, date, hour):
    """Build the CDAWeb download URL for a given data product and time."""
    return CDAWEB_URL.format(
        kind=kind,
        year=date.strftime("%Y"),
        month=date.strftime("%m"),
        day=date.strftime("%d"),
        hour=hour,
    )


def build_filename(kind, date, hour):
    """Build the local filename for a CDF file."""
    return f"psp_fld_l2_dfb_dbm_{kind}_{date.strftime('%Y%m%d')}{hour}_v02.cdf"


def download_file(url, output_path):
    """Download a file from a URL, skipping if it already exists."""
    if os.path.exists(output_path):
        print(f"  Already exists: {output_path}")
        return True

    print(f"  Downloading: {url}")
    try:
        urllib.request.urlretrieve(url, output_path)
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"  Saved: {output_path} ({size_mb:.1f} MB)")
        return True
    except urllib.error.HTTPError as e:
        print(f"  HTTP Error {e.code}: {url}")
        if os.path.exists(output_path):
            os.remove(output_path)
        return False
    except urllib.error.URLError as e:
        print(f"  URL Error: {e.reason}")
        return False


def date_range(start, end):
    """Yield dates from start to end inclusive."""
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def main():
    parser = argparse.ArgumentParser(
        description="Download PSP FIELDS DBM burst data from NASA CDAWeb.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--kind",
        required=True,
        choices=["dvac", "vac"],
        help="Data product type",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="data",
        help="Output directory (default: data)",
    )

    # Single file options
    single = parser.add_argument_group("single file")
    single.add_argument(
        "--date",
        type=str,
        help="Date to download (YYYY-MM-DD)",
    )
    single.add_argument(
        "--hour",
        type=str,
        choices=HOURS,
        help="6-hour block to download (00, 06, 12, or 18)",
    )
    single.add_argument(
        "--all-hours",
        action="store_true",
        help="Download all 6-hour blocks for the given date",
    )

    # Date range options
    range_group = parser.add_argument_group("date range")
    range_group.add_argument(
        "--start",
        type=str,
        help="Start date for range download (YYYY-MM-DD)",
    )
    range_group.add_argument(
        "--end",
        type=str,
        help="End date for range download (YYYY-MM-DD)",
    )

    args = parser.parse_args()

    # Determine which dates and hours to download
    downloads = []

    if args.start and args.end:
        start = datetime.strptime(args.start, "%Y-%m-%d")
        end = datetime.strptime(args.end, "%Y-%m-%d")
        for d in date_range(start, end):
            for h in HOURS:
                downloads.append((d, h))
    elif args.date:
        d = datetime.strptime(args.date, "%Y-%m-%d")
        if args.all_hours:
            for h in HOURS:
                downloads.append((d, h))
        elif args.hour:
            downloads.append((d, args.hour))
        else:
            parser.error("Specify --hour or --all-hours with --date")
    else:
        parser.error("Specify either --date or --start/--end")

    # Download
    os.makedirs(args.output, exist_ok=True)

    success = 0
    failed = 0
    skipped = 0

    for d, h in downloads:
        url = build_url(args.kind, d, h)
        filename = build_filename(args.kind, d, h)
        output_path = os.path.join(args.output, filename)

        if os.path.exists(output_path):
            skipped += 1
            print(f"  Already exists: {output_path}")
        elif download_file(url, output_path):
            success += 1
        else:
            failed += 1

    print(f"\nDone: {success} downloaded, {skipped} skipped, {failed} failed")


if __name__ == "__main__":
    main()
