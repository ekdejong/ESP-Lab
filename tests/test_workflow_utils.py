import os
import time
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
import cftime

from esp_lab.leadtime_validation import check_remove_drift_sample
from esp_lab import env_paths
from esp_lab.utils.netcdf_utils import (
    atomic_to_netcdf,
    cleanup_netcdf_temp_files,
    load_netcdf,
)
from esp_lab.utils.resource_utils import ResourceTracker
from esp_lab.utils.unit_conversion import (
    convert_kelvin_to_celsius,
    convert_pa_to_hpa,
    convert_precip_mmps_to_mmday,
    convert_precip_mps_to_mmday,
    no_unit_conversion,
)


class _Resource:
    def __init__(self, name, closed):
        self.name = name
        self.closed = closed

    def close(self):
        self.closed.append(self.name)


def test_resource_tracker_closes_unique_resources_in_reverse_order():
    closed = []
    first = _Resource("first", closed)
    second = _Resource("second", closed)
    tracker = ResourceTracker(first, second)

    assert tracker.track(first) is first
    assert tracker.track(None) is None
    tracker.close()
    tracker.close()

    assert closed == ["first", "second"]
    with pytest.raises(RuntimeError, match="closed"):
        tracker.track(_Resource("late", closed))


def test_samudrace_output_roots_are_private_and_benchmark_is_separate(monkeypatch, tmp_path):
    monkeypatch.setenv("PSCRATCH", str(tmp_path / "scratch"))
    monkeypatch.delenv("ESP_LAB_S2D_DIAG_ROOT", raising=False)
    monkeypatch.delenv("ESP_LAB_FIGURE_ROOT", raising=False)
    monkeypatch.delenv("ESP_LAB_CESM_SMYLE_BENCHMARK_ROOT", raising=False)
    assert env_paths.samudrace_diagnostic_root() == tmp_path / "scratch/esp-lab-output/diagnostics"
    assert env_paths.samudrace_figure_root() == tmp_path / "scratch/esp-lab-output/figures"
    assert env_paths.cesm_smyle_benchmark_root() == Path("/global/cfs/cdirs/e3sm/S2S2D/s2d_diag/CESM-SMYLE")
    monkeypatch.setenv("ESP_LAB_S2D_DIAG_ROOT", str(tmp_path / "custom-cache"))
    monkeypatch.setenv("ESP_LAB_FIGURE_ROOT", str(tmp_path / "custom-figures"))
    monkeypatch.setenv("ESP_LAB_CESM_SMYLE_BENCHMARK_ROOT", str(tmp_path / "benchmark"))
    assert env_paths.samudrace_diagnostic_root() == tmp_path / "custom-cache"
    assert env_paths.samudrace_figure_root() == tmp_path / "custom-figures"
    assert env_paths.cesm_smyle_benchmark_root() == tmp_path / "benchmark"


def test_atomic_netcdf_round_trip_and_temp_cleanup(tmp_path):
    destination = tmp_path / "cache.nc"
    stale = tmp_path / ".cache.nc.tmp.stale"
    stale.write_text("incomplete")
    old_time = time.time() - 7200
    os.utime(stale, (old_time, old_time))

    removed = cleanup_netcdf_temp_files(destination, max_age_hours=1)
    assert removed == [stale]

    dataset = xr.Dataset({"value": ("x", [1.0, 2.0])})
    assert atomic_to_netcdf(dataset, destination) == destination
    xr.testing.assert_identical(load_netcdf(destination), dataset)
    assert not list(tmp_path.glob(".cache.nc.tmp.*"))


def test_unit_conversions_preserve_lazy_xarray_behavior():
    kelvin = xr.DataArray([273.15, 274.15], dims="x", attrs={"units": "K"})
    precip = xr.DataArray([1.0e-3], dims="x", attrs={"units": "m/s"})
    pressure = xr.DataArray([100000.0], dims="x", attrs={"units": "Pa"})

    np.testing.assert_allclose(convert_kelvin_to_celsius(kelvin), [0.0, 1.0])
    np.testing.assert_allclose(convert_precip_mps_to_mmday(precip), [86400.0])
    np.testing.assert_allclose(convert_precip_mmps_to_mmday(precip), [86.4])
    np.testing.assert_allclose(convert_pa_to_hpa(pressure), [1000.0])
    assert convert_kelvin_to_celsius(kelvin).attrs["units"] == r"$^\circ$C"
    assert convert_precip_mps_to_mmday(precip).attrs["units"] == "mm/day"
    assert convert_precip_mmps_to_mmday(precip).attrs["units"] == "mm/day"
    assert convert_pa_to_hpa(pressure).attrs["units"] == "hPa"
    assert no_unit_conversion(kelvin, units="standardized").attrs["units"] == "standardized"


def test_sampled_drift_validation_accepts_consistent_anomaly(capsys):
    data = xr.DataArray(
        np.arange(8.0).reshape(2, 2, 2),
        dims=("Y", "M", "L"),
        coords={"Y": [2000, 2001], "M": [0, 1], "L": [1, 2]},
    )
    verification_time = xr.DataArray(
        [
            [cftime.DatetimeNoLeap(2000, 1, 15), cftime.DatetimeNoLeap(2000, 4, 15)],
            [cftime.DatetimeNoLeap(2001, 1, 15), cftime.DatetimeNoLeap(2001, 4, 15)],
        ],
        dims=("Y", "L"),
        coords={"Y": data.Y, "L": data.L},
    )
    climatology = data.mean(("Y", "M"))
    anomaly = data - climatology

    check_remove_drift_sample(
        data, verification_time, anomaly, climatology, 2000, 2001,
        sample={"Y": 0, "M": 0, "L": 0}, name="test",
    )

    assert "sample reconstruction PASS" in capsys.readouterr().out


def test_atomic_to_netcdf_computes_dask_data_before_writing(tmp_path):
    import dask.array as da
    import numpy as np
    import pytest
    import xarray as xr

    from esp_lab.utils.netcdf_utils import atomic_to_netcdf

    lazy = xr.Dataset({"x": (("t",), da.arange(10, chunks=3).astype("float32"))})
    out = atomic_to_netcdf(lazy, tmp_path / "small.nc")
    with xr.open_dataset(out) as ds:
        np.testing.assert_array_equal(ds["x"].values, np.arange(10))

    # Above the in-memory limit the write stays lazy and says so.
    with pytest.warns(RuntimeWarning, match="lazily"):
        atomic_to_netcdf(lazy, tmp_path / "big.nc", max_in_memory_bytes=1)
    assert (tmp_path / "big.nc").is_file()
