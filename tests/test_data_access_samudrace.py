"""Tests for the SamudrACE monthly lead-time adapter."""

import cftime
import numpy as np
import pytest
import xarray as xr

from esp_lab.data_access_samudrace import preprocess_monthly


def monthly_dataset():
    dates = [cftime.DatetimeNoLeap(2000 + (10 + lead) // 12, (10 + lead) % 12 + 1, 15) for lead in range(25)]
    return xr.Dataset(
        {
            "Tat2m": (("sample", "time", "lat", "lon"), np.ones((2, 25, 1, 1))),
            "valid_time": (("sample", "time"), np.tile(dates, (2, 1))),
            "init_time": ("sample", [cftime.DatetimeNoLeap(2000, 11, 2)] * 2),
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


def test_monthly_adapter_rejects_bad_coverage_and_time_alignment():
    data = monthly_dataset()
    with pytest.raises(ValueError, match="between 1 and 24"):
        preprocess_monthly(data, "Tat2m", nlead=25)
    data.valid_time.values[1, 0] = cftime.DatetimeNoLeap(2000, 12, 15)
    with pytest.raises(ValueError, match="share initialization and valid times"):
        preprocess_monthly(data, "Tat2m")