"""Tests for the SamudrACE monthly lead-time adapter."""

import cftime
import numpy as np
import pytest
import xarray as xr

from esp_lab.data_access_samudrace import get_monthly_data, preprocess_monthly
from esp_lab.utils.calendar_utils import mon_to_seas_dask


def monthly_dataset(year=2000):
    dates = [cftime.DatetimeNoLeap(year + (10 + lead) // 12, (10 + lead) % 12 + 1, 15) for lead in range(25)]
    return xr.Dataset(
        {
            "Tat2m": (("sample", "time", "lat", "lon"), np.ones((2, 25, 1, 1))),
            "valid_time": (("sample", "time"), np.tile(dates, (2, 1))),
            "init_time": ("sample", [cftime.DatetimeNoLeap(year, 11, 2)] * 2),
            "counts": (("sample", "time"), np.tile([115] + [120] * 23 + [1], (2, 1))),
        },
        coords={"lat": [0.0], "lon": [0.5]},
    )


def test_monthly_adapter_preserves_initial_month_and_excludes_partial_final_month():
    data = monthly_dataset().set_coords(["init_time", "counts", "valid_time"])
    result = preprocess_monthly(data, "Tat2m")
    assert result.Tat2m.dims == ("Y", "L", "M", "lat", "lon")
    assert result.sizes == {"Y": 1, "L": 24, "M": 2, "lat": 1, "lon": 1}
    assert result.Y.item() == "2000110100"
    assert result.time.values[0, 0] == cftime.DatetimeNoLeap(2000, 11, 15)
    assert result.time.values[0, -1] == cftime.DatetimeNoLeap(2002, 10, 15)
    assert result.init_time.values[0, 0] == cftime.DatetimeNoLeap(2000, 11, 2)
    assert result.counts.values[0, 0, 0] == 115
    assert "valid_time" not in result


def test_last_centered_season_is_incomplete_without_25th_month():
    monthly = preprocess_monthly(monthly_dataset(), "Tat2m")
    seasonal = mon_to_seas_dask(monthly[["Tat2m", "time"]])
    assert seasonal.L.values.tolist() == [3, 6, 9, 12, 15, 18, 21, 24]
    assert seasonal.Tat2m.isel(L=-2).notnull().all().item()
    assert seasonal.Tat2m.isel(L=-1).isnull().all().item()


def test_monthly_adapter_rejects_bad_coverage_and_time_alignment():
    data = monthly_dataset()
    with pytest.raises(ValueError, match="between 1 and 24"):
        preprocess_monthly(data, "Tat2m", nlead=25)
    data.valid_time.values[1, 0] = cftime.DatetimeNoLeap(2000, 12, 15)
    with pytest.raises(ValueError, match="share initialization and valid times"):
        preprocess_monthly(data, "Tat2m")
    data = monthly_dataset()
    data["counts"] = data.counts.astype(float)
    data.counts.values[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite and positive"):
        preprocess_monthly(data, "Tat2m")


def write_run(root, case, year, data=None):
    path = root / case / f"{case}-{year}-nov" / "atmosphere" / "monthly_mean_predictions.nc"
    path.parent.mkdir(parents=True)
    (monthly_dataset(year) if data is None else data).set_coords(
        ["init_time", "counts", "valid_time"]
    ).to_netcdf(path)
    return path


def test_load_multiple_runs_preserves_times_counts_and_lazy_data(tmp_path):
    case = "C1A1-01mo-EN00"
    for year in (2000, 2001):
        write_run(tmp_path, case, year)
    tags = ["2001110100", "2000110100"]
    with get_monthly_data(tmp_path, case, tags, "Tat2m", chunks={"Y": 1}) as result:
        assert result.Tat2m.dims == ("Y", "L", "M", "lat", "lon")
        assert result.Y.values.tolist() == tags
        assert result.time.values[:, 0].tolist() == [
            cftime.DatetimeNoLeap(2001, 11, 15), cftime.DatetimeNoLeap(2000, 11, 15)
        ]
        assert result.init_time.values[:, 0].tolist() == [
            cftime.DatetimeNoLeap(2001, 11, 2), cftime.DatetimeNoLeap(2000, 11, 2)
        ]
        assert result.counts.values[:, 0, 0].tolist() == [115, 115]
        assert result.Tat2m.chunks is not None
        assert result.Tat2m.isel(Y=0, L=0, M=0).compute().item() == 1


def test_load_rejects_missing_runs_and_invalid_requests(tmp_path):
    case = "C1A1-01mo-EN00"
    write_run(tmp_path, case, 2000)
    with pytest.raises(FileNotFoundError, match="2001110100"):
        get_monthly_data(tmp_path, case, ["2000110100", "2001110100"], "Tat2m")
    with pytest.raises(ValueError, match="no duplicates"):
        get_monthly_data(tmp_path, case, ["2000110100", "2000110100"], "Tat2m")
    with pytest.raises(ValueError, match="Invalid initialization tag"):
        get_monthly_data(tmp_path, case, ["2000130100"], "Tat2m")
    with pytest.raises(ValueError, match="Unsupported SamudrACE realm"):
        get_monthly_data(tmp_path, case, ["2000110100"], "Tat2m", realm="land")


def test_load_rejects_mismatched_initialization(tmp_path):
    case = "C1A1-01mo-EN00"
    write_run(tmp_path, case, 2000)
    write_run(tmp_path, case, 2001, monthly_dataset(2000))
    with pytest.raises(ValueError, match="do not match requested tags"):
        get_monthly_data(tmp_path, case, ["2000110100", "2001110100"], "Tat2m")


@pytest.mark.parametrize("changed", ["grid", "members"])
def test_load_rejects_inconsistent_run_dimensions(tmp_path, changed):
    case = "C1A1-01mo-EN00"
    write_run(tmp_path, case, 2000)
    data = monthly_dataset(2001)
    if changed == "grid":
        data = data.assign_coords(lat=[1.0])
    else:
        data = data.isel(sample=slice(0, 1))
    write_run(tmp_path, case, 2001, data)
    with pytest.raises(ValueError, match="align|index|size"):
        get_monthly_data(tmp_path, case, ["2000110100", "2001110100"], "Tat2m")


@pytest.mark.parametrize("chunks", [None, {"Y": 1}])
def test_load_closes_source_datasets(tmp_path, monkeypatch, chunks):
    case = "C1A1-01mo-EN00"
    write_run(tmp_path, case, 2000)
    original_open = xr.open_mfdataset
    closed = []

    def tracked_open(*args, **kwargs):
        source = original_open(*args, **kwargs)
        close_source = source._close

        def tracked_close():
            closed.append(True)
            close_source()

        source.set_close(tracked_close)
        return source

    monkeypatch.setattr(xr, "open_mfdataset", tracked_open)
    with get_monthly_data(tmp_path, case, ["2000110100"], "Tat2m", chunks=chunks):
        assert not closed
    assert closed == [True]