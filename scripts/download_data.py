#!/usr/bin/env python
"""Backward-compatible wrapper for the packaged EMBER download CLI."""

from ember.cli import main_download as main
from ember.download import (
    HOURS,
    build_download_plan,
    build_filename,
    build_url,
    date_range,
    download_file,
    download_products,
)

__all__ = [
    "HOURS",
    "build_download_plan",
    "build_filename",
    "build_url",
    "date_range",
    "download_file",
    "download_products",
    "main",
]


if __name__ == "__main__":
    main()
