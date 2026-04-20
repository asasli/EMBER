"""Download helpers for PSP FIELDS DBM burst products."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable
import urllib.error
import urllib.request

CDAWEB_URL = (
    "https://cdaweb.gsfc.nasa.gov/sp_phys/data/psp/fields/l2/"
    "dfb_dbm_{kind}/{year}/"
    "psp_fld_l2_dfb_dbm_{kind}_{year}{month}{day}{hour}_v02.cdf"
)

HOURS = ("00", "06", "12", "18")


def _coerce_datetime(value: str | datetime | date) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime.combine(value, datetime.min.time())
    return datetime.strptime(value, "%Y-%m-%d")


def build_url(kind: str, when: str | datetime | date, hour: str) -> str:
    """Build the CDAWeb URL for a PSP DBM product."""

    when_dt = _coerce_datetime(when)
    return CDAWEB_URL.format(
        kind=kind,
        year=when_dt.strftime("%Y"),
        month=when_dt.strftime("%m"),
        day=when_dt.strftime("%d"),
        hour=hour,
    )


def build_filename(kind: str, when: str | datetime | date, hour: str) -> str:
    """Build the local filename for a PSP DBM product."""

    when_dt = _coerce_datetime(when)
    return f"psp_fld_l2_dfb_dbm_{kind}_{when_dt.strftime('%Y%m%d')}{hour}_v02.cdf"


def date_range(start: str | datetime | date, end: str | datetime | date):
    """Yield dates from start to end inclusive."""

    current = _coerce_datetime(start)
    end_dt = _coerce_datetime(end)
    while current <= end_dt:
        yield current
        current += timedelta(days=1)


def build_download_plan(
    kind: str,
    *,
    date: str | datetime | date | None = None,
    hour: str | None = None,
    all_hours: bool = False,
    start: str | datetime | date | None = None,
    end: str | datetime | date | None = None,
) -> list[tuple[datetime, str]]:
    """Build the list of (date, hour) products to download."""

    if start and end:
        plan: list[tuple[datetime, str]] = []
        for day in date_range(start, end):
            for hour_code in HOURS:
                plan.append((day, hour_code))
        return plan

    if date is None:
        raise ValueError("Specify either date/hour or a start/end date range.")

    day = _coerce_datetime(date)
    if all_hours:
        return [(day, hour_code) for hour_code in HOURS]
    if hour is None:
        raise ValueError("Specify an hour or set all_hours=True.")
    if hour not in HOURS:
        raise ValueError(f"Hour must be one of {HOURS}.")
    return [(day, hour)]


def download_file(url: str, output_path: str | Path) -> bool:
    """Download a file from a URL, skipping if it already exists."""

    output_path = Path(output_path)
    if output_path.exists():
        print(f"  Already exists: {output_path}")
        return True

    print(f"  Downloading: {url}")
    try:
        urllib.request.urlretrieve(url, output_path)
        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"  Saved: {output_path} ({size_mb:.1f} MB)")
        return True
    except urllib.error.HTTPError as exc:
        print(f"  HTTP Error {exc.code}: {url}")
        if output_path.exists():
            output_path.unlink()
        return False
    except urllib.error.URLError as exc:
        print(f"  URL Error: {exc.reason}")
        return False
    except Exception:
        if output_path.exists():
            output_path.unlink()
        raise


def download_products(
    kind: str,
    *,
    output: str | Path = "data",
    date: str | datetime | date | None = None,
    hour: str | None = None,
    all_hours: bool = False,
    start: str | datetime | date | None = None,
    end: str | datetime | date | None = None,
    downloader: Callable[[str, str | Path], bool] = download_file,
) -> dict[str, int | list[str]]:
    """Download one or more PSP DBM products and return a summary."""

    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)
    plan = build_download_plan(
        kind,
        date=date,
        hour=hour,
        all_hours=all_hours,
        start=start,
        end=end,
    )

    summary = {
        "downloaded": 0,
        "skipped": 0,
        "failed": 0,
        "paths": [],
    }
    for day, hour_code in plan:
        url = build_url(kind, day, hour_code)
        filename = build_filename(kind, day, hour_code)
        output_path = output_dir / filename
        if output_path.exists():
            summary["skipped"] += 1
            print(f"  Already exists: {output_path}")
            summary["paths"].append(str(output_path))
            continue

        if downloader(url, output_path):
            summary["downloaded"] += 1
            summary["paths"].append(str(output_path))
        else:
            summary["failed"] += 1

    return summary
