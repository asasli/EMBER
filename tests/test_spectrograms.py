"""Tests for spectrogram creation scripts using synthetic data."""

import importlib.util
import os

import cdflib
import numpy as np

SCRIPTS_DIR = os.path.join(os.path.dirname(__file__), "..", "scripts")


def _load_script(name):
    """Load a script module by name from the scripts directory."""
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(SCRIPTS_DIR, f"{name}.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def create_synthetic_dvac_cdf(path, n_bursts=2, n_samples=4096):
    """Create a minimal synthetic DVAC CDF file for testing."""
    fs = 37500.0  # ~37.5 kHz like real DVAC data
    t = np.arange(n_samples) / fs
    data = np.array([np.sin(2 * np.pi * 1000 * t + i) for i in range(n_bursts)])

    base_tt2000 = cdflib.cdfepoch.compute_tt2000([2020, 9, 25, 6, 0, 0, 0, 0, 0])
    dt_ns = int(1e9 / fs)
    time_data = np.array(
        [
            [base_tt2000 + j * dt_ns + i * n_samples * dt_ns for j in range(n_samples)]
            for i in range(n_bursts)
        ],
        dtype=np.int64,
    )

    cdf = cdflib.cdfwrite.CDF(path)
    cdf.write_var(
        {
            "Variable": "psp_fld_l2_dfb_dbm_dvac12",
            "Data_Type": 8,
            "Num_Elements": 1,
            "Rec_Vary": True,
            "Dim_Sizes": [n_samples],
            "Dim_Vary": [True],
        },
        var_data=data,
    )
    cdf.write_var(
        {
            "Variable": "psp_fld_l2_dfb_dbm_dvac_time_series_TT2000",
            "Data_Type": 33,
            "Num_Elements": 1,
            "Rec_Vary": True,
            "Dim_Sizes": [n_samples],
            "Dim_Vary": [True],
        },
        var_data=time_data,
    )
    cdf.close()


def create_synthetic_vac_cdf(path, n_bursts=2, n_samples=4096):
    """Create a minimal synthetic VAC CDF file for testing."""
    fs = 18750.0  # ~18.75 kHz like real VAC data
    t = np.arange(n_samples) / fs
    vac1 = np.array([np.sin(2 * np.pi * 500 * t + i) for i in range(n_bursts)])
    vac2 = np.array([np.sin(2 * np.pi * 500 * t + i + 0.5) for i in range(n_bursts)])

    base_tt2000 = cdflib.cdfepoch.compute_tt2000([2021, 8, 8, 18, 0, 0, 0, 0, 0])
    dt_ns = int(1e9 / fs)
    time_data = np.array(
        [
            [base_tt2000 + j * dt_ns + i * n_samples * dt_ns for j in range(n_samples)]
            for i in range(n_bursts)
        ],
        dtype=np.int64,
    )

    cdf = cdflib.cdfwrite.CDF(path)
    cdf.write_var(
        {
            "Variable": "psp_fld_l2_dfb_dbm_vac1",
            "Data_Type": 8,
            "Num_Elements": 1,
            "Rec_Vary": True,
            "Dim_Sizes": [n_samples],
            "Dim_Vary": [True],
        },
        var_data=vac1,
    )
    cdf.write_var(
        {
            "Variable": "psp_fld_l2_dfb_dbm_vac2",
            "Data_Type": 8,
            "Num_Elements": 1,
            "Rec_Vary": True,
            "Dim_Sizes": [n_samples],
            "Dim_Vary": [True],
        },
        var_data=vac2,
    )
    cdf.write_var(
        {
            "Variable": "psp_fld_l2_dfb_dbm_vac_time_series_TT2000",
            "Data_Type": 33,
            "Num_Elements": 1,
            "Rec_Vary": True,
            "Dim_Sizes": [n_samples],
            "Dim_Vary": [True],
        },
        var_data=time_data,
    )
    cdf.close()


class TestDvacSpectrograms:
    def test_creates_pngs(self, tmp_path):
        """Test that DVAC script produces expected number of PNG files."""
        mod = _load_script("create_spectrograms_dvac")
        cdf_path = str(tmp_path / "test_dvac.cdf")
        output_dir = str(tmp_path / "output")
        create_synthetic_dvac_cdf(cdf_path, n_bursts=3, n_samples=4096)

        mod.create_spectrograms(cdf_path, output_dir, fmt="png")

        pngs = [f for f in os.listdir(output_dir) if f.endswith(".png")]
        assert len(pngs) == 3

    def test_creates_npz(self, tmp_path):
        """Test that DVAC script produces npz files with correct arrays."""
        mod = _load_script("create_spectrograms_dvac")
        cdf_path = str(tmp_path / "test_dvac.cdf")
        output_dir = str(tmp_path / "output")
        create_synthetic_dvac_cdf(cdf_path, n_bursts=2, n_samples=4096)

        mod.create_spectrograms(cdf_path, output_dir, fmt="npz")

        npzs = [f for f in os.listdir(output_dir) if f.endswith(".npz")]
        assert len(npzs) == 2
        # No PNGs should be created
        pngs = [f for f in os.listdir(output_dir) if f.endswith(".png")]
        assert len(pngs) == 0

        # Verify npz contents
        data = np.load(
            os.path.join(output_dir, "dvac12_burst_000.npz"), allow_pickle=True
        )
        assert "f_log" in data
        assert "Sxx_log" in data
        assert "t" in data
        assert "fs" in data
        assert "timestamp" in data
        assert data["Sxx_log"].shape[0] == data["f_log"].shape[0]

    def test_creates_both(self, tmp_path):
        """Test that fmt='both' produces both PNG and npz files."""
        mod = _load_script("create_spectrograms_dvac")
        cdf_path = str(tmp_path / "test_dvac.cdf")
        output_dir = str(tmp_path / "output")
        create_synthetic_dvac_cdf(cdf_path, n_bursts=2, n_samples=4096)

        mod.create_spectrograms(cdf_path, output_dir, fmt="both")

        pngs = [f for f in os.listdir(output_dir) if f.endswith(".png")]
        npzs = [f for f in os.listdir(output_dir) if f.endswith(".npz")]
        assert len(pngs) == 2
        assert len(npzs) == 2

    def test_output_filenames(self, tmp_path):
        """Test that output filenames follow expected pattern."""
        mod = _load_script("create_spectrograms_dvac")
        cdf_path = str(tmp_path / "test_dvac.cdf")
        output_dir = str(tmp_path / "output")
        create_synthetic_dvac_cdf(cdf_path, n_bursts=2, n_samples=4096)

        mod.create_spectrograms(cdf_path, output_dir)

        assert os.path.exists(os.path.join(output_dir, "dvac12_burst_000.png"))
        assert os.path.exists(os.path.join(output_dir, "dvac12_burst_001.png"))
        assert os.path.exists(os.path.join(output_dir, "dvac12_burst_000.npz"))
        assert os.path.exists(os.path.join(output_dir, "dvac12_burst_001.npz"))


class TestVacSpectrograms:
    def test_creates_pngs(self, tmp_path):
        """Test that VAC script produces expected number of PNG files."""
        mod = _load_script("create_spectrograms_vac")
        cdf_path = str(tmp_path / "test_vac.cdf")
        output_dir = str(tmp_path / "output")
        create_synthetic_vac_cdf(cdf_path, n_bursts=3, n_samples=4096)

        mod.create_spectrograms(cdf_path, output_dir, fmt="png")

        pngs = [f for f in os.listdir(output_dir) if f.endswith(".png")]
        assert len(pngs) == 3

    def test_creates_npz(self, tmp_path):
        """Test that VAC script produces npz files with correct arrays."""
        mod = _load_script("create_spectrograms_vac")
        cdf_path = str(tmp_path / "test_vac.cdf")
        output_dir = str(tmp_path / "output")
        create_synthetic_vac_cdf(cdf_path, n_bursts=2, n_samples=4096)

        mod.create_spectrograms(cdf_path, output_dir, fmt="npz")

        npzs = [f for f in os.listdir(output_dir) if f.endswith(".npz")]
        assert len(npzs) == 2

        data = np.load(
            os.path.join(output_dir, "vac12_burst_000.npz"), allow_pickle=True
        )
        assert "f_log" in data
        assert "Sxx_log" in data
        assert "t" in data
        assert "fs" in data
        assert "timestamp" in data

    def test_creates_both(self, tmp_path):
        """Test that fmt='both' produces both PNG and npz files."""
        mod = _load_script("create_spectrograms_vac")
        cdf_path = str(tmp_path / "test_vac.cdf")
        output_dir = str(tmp_path / "output")
        create_synthetic_vac_cdf(cdf_path, n_bursts=2, n_samples=4096)

        mod.create_spectrograms(cdf_path, output_dir, fmt="both")

        pngs = [f for f in os.listdir(output_dir) if f.endswith(".png")]
        npzs = [f for f in os.listdir(output_dir) if f.endswith(".npz")]
        assert len(pngs) == 2
        assert len(npzs) == 2

    def test_differential_voltage_computed(self, tmp_path):
        """Test that VAC script runs without error (differential computed internally)."""
        mod = _load_script("create_spectrograms_vac")
        cdf_path = str(tmp_path / "test_vac.cdf")
        output_dir = str(tmp_path / "output")
        create_synthetic_vac_cdf(cdf_path, n_bursts=2, n_samples=4096)

        mod.create_spectrograms(cdf_path, output_dir, fmt="npz")

        npzs = [f for f in os.listdir(output_dir) if f.endswith(".npz")]
        assert len(npzs) == 2
