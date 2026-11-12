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


def write_run(root, case, year, data=None, realm="atmosphere"):
    path = root / case / f"{case}-{year}-nov" / realm / "monthly_mean_predictions.nc"
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


def test_prepare_atmospheric_anomaly_converts_and_restarts_from_cache(tmp_path, monkeypatch):
    from workflows.leadtime_skill import samudrace

    case = "C1A1-01mo-EN00"
    source_paths = []
    for year in (2000, 2001):
        data = monthly_dataset(year)
        data["surface_precipitation_rate"] = data.Tat2m * (year - 1999) * 0.001
        source_paths.append(write_run(tmp_path, case, year, data))
    calls = []

    def identity_regridder(source, target, **kwargs):
        calls.append(kwargs)
        return lambda data: data

    monkeypatch.setattr(samudrace.regrid_utils, "make_regridder", identity_regridder)
    args = dict(
        data_dir=tmp_path, case=case, field="PRECT", init_month=11,
        years=[2000, 2001], climatology_years=(2000, 2003),
        diagnostic_root=tmp_path / "diagnostics", target_grid=xr.Dataset(),
        target_grid_identity="test-grid", regridding_method="conservative",
        periodic=True, chunks={"Y": 1, "L": 7, "M": 1, "lat": 1, "lon": 1},
        ensemble_member_count=2,
    )
    prepared, identity = samudrace.prepare_atmospheric_anomaly(**args)
    try:
        assert prepared.L.values.tolist() == [3, 6, 9, 12, 15, 18, 21]
        np.testing.assert_allclose(
            prepared.anomaly.isel(L=0, M=0, lat=0, lon=0).values, [-43.2, 43.2]
        )
        assert prepared.anomaly.attrs["units"] == "mm/day"
        assert prepared.attrs["source_data_identity"] == identity
    finally:
        prepared.close()
    assert len(calls) == 1

    for path in source_paths:
        path.unlink()
    cached, cached_identity = samudrace.prepare_atmospheric_anomaly(
        **args, source_identity_mode="snapshot", prepared_mode="require"
    )
    try:
        assert cached_identity == identity
        np.testing.assert_allclose(
            cached.anomaly.isel(L=0, M=0, lat=0, lon=0).values, [-43.2, 43.2]
        )
    finally:
        cached.close()
    assert len(calls) == 1


def test_prepare_atmospheric_anomaly_rejects_unsupported_field(tmp_path):
    from workflows.leadtime_skill.samudrace import prepare_atmospheric_anomaly

    with pytest.raises(ValueError, match="Unsupported SamudrACE atm field"):
        prepare_atmospheric_anomaly(
            data_dir=tmp_path, case="case", field="PSL", init_month=11,
            years=[2000], climatology_years=(2000, 2001), diagnostic_root=tmp_path,
            target_grid=xr.Dataset(), target_grid_identity="test-grid",
            regridding_method="conservative", periodic=True, chunks={},
        )


def test_prepare_ocean_sst_uses_kelvin_surface_field_and_retains_mask(tmp_path, monkeypatch):
    from workflows.leadtime_skill import samudrace

    case = "C1A1-01mo-EN00"
    for year in (2000, 2001):
        data = monthly_dataset(year).reindex(lon=[0.5, 1.5])
        data["sst"] = 273.15 + data.Tat2m * (year - 1999)
        write_run(tmp_path, case, year, data, realm="ocean")
    monkeypatch.setattr(
        samudrace.regrid_utils, "make_regridder", lambda *args, **kwargs: lambda data: data
    )
    prepared, _ = samudrace.prepare_seasonal_anomaly(
        data_dir=tmp_path, case=case, field="SST", realm="ocn", init_month=11,
        years=[2000, 2001], climatology_years=(2000, 2003),
        diagnostic_root=tmp_path / "diagnostics", target_grid=xr.Dataset(),
        target_grid_identity="test-grid", regridding_method="conservative",
        periodic=True, chunks={"Y": 1, "L": 7, "M": 1, "lat": 1, "lon": 2},
        ensemble_member_count=2,
    )
    try:
        np.testing.assert_allclose(
            prepared.anomaly.isel(L=0, M=0, lat=0, lon=0).values, [-0.5, 0.5]
        )
        assert prepared.anomaly.isel(lon=1).isnull().all().compute().item()
        assert prepared.anomaly.attrs["units"] == r"$^\circ$C"
    finally:
        prepared.close()


@pytest.mark.parametrize("field,native_field", [("TREFHT", "Tat2m"), ("TS", "TS")])
def test_prepare_atmospheric_temperature_converts_kelvin(tmp_path, monkeypatch, field, native_field):
    from workflows.leadtime_skill import samudrace

    case = "C1A1-01mo-EN00"
    for year in (2000, 2001):
        data = monthly_dataset(year)
        data[native_field] = 273.15 + data.Tat2m * (year - 1999)
        write_run(tmp_path, case, year, data)
    monkeypatch.setattr(
        samudrace.regrid_utils, "make_regridder", lambda *args, **kwargs: lambda data: data
    )
    prepared, _ = samudrace.prepare_atmospheric_anomaly(
        data_dir=tmp_path, case=case, field=field, init_month=11,
        years=[2000, 2001], climatology_years=(2000, 2003),
        diagnostic_root=tmp_path / "diagnostics", target_grid=xr.Dataset(),
        target_grid_identity="test-grid", regridding_method="conservative",
        periodic=True, chunks={"Y": 2, "L": 7, "M": 2, "lat": 1, "lon": 1},
        ensemble_member_count=2,
    )
    try:
        np.testing.assert_allclose(
            prepared.climatology.isel(L=0, lat=0, lon=0).values, 1.5
        )
        assert prepared.anomaly.attrs["units"] == r"$^\circ$C"
    finally:
        prepared.close()


def test_prepared_precipitation_runs_seven_lead_rmse_skill(tmp_path, monkeypatch):
    from esp_lab.leadtime_realm_skill import compute_and_cache_rmse_skill
    from workflows.leadtime_skill import samudrace

    case = "C1A1-01mo-EN00"
    years = [2000, 2001, 2002, 2003]
    for year in years:
        data = monthly_dataset(year)
        data["surface_precipitation_rate"] = data.Tat2m * (year - 1999) * 0.001
        write_run(tmp_path, case, year, data)
    monkeypatch.setattr(
        samudrace.regrid_utils, "make_regridder", lambda *args, **kwargs: lambda data: data
    )
    prepared, _ = samudrace.prepare_seasonal_anomaly(
        data_dir=tmp_path, case=case, field="PRECT", realm="atm", init_month=11,
        years=years, climatology_years=(2000, 2005),
        diagnostic_root=tmp_path / "diagnostics", target_grid=xr.Dataset(),
        target_grid_identity="test-grid", regridding_method="conservative",
        periodic=True, chunks={"Y": 4, "L": 7, "M": 2, "lat": 1, "lon": 1},
        ensemble_member_count=2,
    )
    obs_times = sorted(set(prepared.time.values.ravel()))
    obs_values = [(date.year - 1999) ** 2 + date.month / 12 for date in obs_times]
    obs = xr.DataArray(
        np.array(obs_values).reshape(-1, 1, 1), dims=("time", "lat", "lon"),
        coords={"time": obs_times, "lat": [0.0], "lon": [0.5]},
    )
    try:
        skill = compute_and_cache_rmse_skill(
            prepared.anomaly, prepared.time.load(), obs,
            cftime.DatetimeNoLeap(2000, 1, 1), cftime.DatetimeNoLeap(2005, 12, 31),
            1, 7, outfile=tmp_path / "skill.nc", expected_attrs={},
            netcdf_write_options={}, detrend=False,
        )
        assert skill.L.values.tolist() == [3, 6, 9, 12, 15, 18, 21]
        assert np.isfinite(skill.rmse.values).all()
    finally:
        prepared.close()