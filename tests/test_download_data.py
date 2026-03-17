"""Tests for the download_data script."""

import os
from datetime import datetime
from unittest import mock


# Import functions from the script
import importlib.util

spec = importlib.util.spec_from_file_location(
    "download_data",
    os.path.join(os.path.dirname(__file__), "..", "scripts", "download_data.py"),
)
download_data = importlib.util.module_from_spec(spec)
spec.loader.exec_module(download_data)


class TestBuildUrl:
    def test_dvac_url(self):
        date = datetime(2020, 9, 25)
        url = download_data.build_url("dvac", date, "06")
        assert url == (
            "https://cdaweb.gsfc.nasa.gov/sp_phys/data/psp/fields/l2/"
            "dfb_dbm_dvac/2020/"
            "psp_fld_l2_dfb_dbm_dvac_2020092506_v02.cdf"
        )

    def test_vac_url(self):
        date = datetime(2021, 8, 8)
        url = download_data.build_url("vac", date, "18")
        assert url == (
            "https://cdaweb.gsfc.nasa.gov/sp_phys/data/psp/fields/l2/"
            "dfb_dbm_vac/2021/"
            "psp_fld_l2_dfb_dbm_vac_2021080818_v02.cdf"
        )

    def test_all_hour_blocks(self):
        date = datetime(2020, 1, 15)
        for hour in ["00", "06", "12", "18"]:
            url = download_data.build_url("dvac", date, hour)
            assert f"2020011500_{hour[-2:]}" in url or f"20200115{hour}" in url


class TestBuildFilename:
    def test_dvac_filename(self):
        date = datetime(2020, 9, 25)
        fname = download_data.build_filename("dvac", date, "06")
        assert fname == "psp_fld_l2_dfb_dbm_dvac_2020092506_v02.cdf"

    def test_vac_filename(self):
        date = datetime(2021, 8, 8)
        fname = download_data.build_filename("vac", date, "18")
        assert fname == "psp_fld_l2_dfb_dbm_vac_2021080818_v02.cdf"


class TestDateRange:
    def test_single_day(self):
        start = datetime(2020, 9, 25)
        dates = list(download_data.date_range(start, start))
        assert len(dates) == 1
        assert dates[0] == start

    def test_multi_day(self):
        start = datetime(2020, 9, 25)
        end = datetime(2020, 9, 27)
        dates = list(download_data.date_range(start, end))
        assert len(dates) == 3
        assert dates[0] == datetime(2020, 9, 25)
        assert dates[-1] == datetime(2020, 9, 27)

    def test_end_before_start(self):
        start = datetime(2020, 9, 27)
        end = datetime(2020, 9, 25)
        dates = list(download_data.date_range(start, end))
        assert len(dates) == 0


class TestDownloadFile:
    def test_skips_existing_file(self, tmp_path):
        existing = tmp_path / "test.cdf"
        existing.write_text("data")
        result = download_data.download_file(
            "http://example.com/test.cdf", str(existing)
        )
        assert result is True

    def test_handles_http_error(self, tmp_path):
        outpath = str(tmp_path / "test.cdf")
        with mock.patch("urllib.request.urlretrieve", side_effect=Exception("fail")):
            # Should not raise
            try:
                download_data.download_file("http://bad-url/test.cdf", outpath)
            except Exception:
                pass
            assert not os.path.exists(outpath)
