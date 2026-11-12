"""Access helpers for monthly SamudrACE hindcast predictions."""

import numpy as np
import xarray as xr


def preprocess_monthly(ds: xr.Dataset, field: str, nlead: int = 24) -> xr.Dataset:
    """Convert one initialized prediction dataset to the lead-time contract.

    The first month is retained even when initialization occurs after its
    beginning. The 25th archived entry is excluded because it can be partial.
    Counts and actual initialization times are retained for coverage checks.
    Times must be decoded before calling this function.
    """
    if not 1 <= nlead <= 24:
        raise ValueError("nlead must be between 1 and 24")
    for name in (field, "valid_time", "init_time", "counts"):
        if name not in ds:
            raise ValueError(f"Missing required SamudrACE variable: {name}")
    if ds.sizes.get("time", 0) < nlead or ds.sizes.get("sample", 0) == 0:
        raise ValueError("Insufficient monthly leads or ensemble samples")
    if ds[field].dims[:2] != ("sample", "time"):
        raise ValueError(f"{field} must start with sample and time dimensions")
    if ds.valid_time.dims != ("sample", "time") or ds.counts.dims != ("sample", "time"):
        raise ValueError("valid_time and counts must have sample and time dimensions")
    if ds.init_time.dims != ("sample",):
        raise ValueError("init_time must have a sample dimension")

    selected = ds.isel(time=slice(0, nlead))
    valid_time = selected.valid_time.values
    init_time = selected.init_time.values
    if not np.all(valid_time == valid_time[0]) or not np.all(init_time == init_time[0]):
        raise ValueError("Ensemble samples must share initialization and valid times")
    if np.any(selected.counts.values <= 0):
        raise ValueError("Monthly sample counts must be positive")
    try:
        months = np.array([date.year * 12 + date.month for date in valid_time[0]])
        init_month = init_time[0].year * 12 + init_time[0].month
    except AttributeError as exc:
        raise ValueError("init_time and valid_time must contain decoded dates") from exc
    if months[0] != init_month or not np.all(np.diff(months) == 1):
        raise ValueError("Valid times must start in the initialization month and advance monthly")

    init_tag = f"{init_time[0].year:04d}{init_time[0].month:02d}0100"
    model = selected[field].reset_coords(drop=True).rename({"sample": "M", "time": "L"})
    counts = selected.counts.reset_coords(drop=True).rename({"sample": "M", "time": "L"})
    actual_init = selected.init_time.reset_coords(drop=True).rename({"sample": "M"})
    result = xr.Dataset({field: model, "counts": counts, "init_time": actual_init})
    result = result.assign_coords(M=np.arange(selected.sizes["sample"]), L=np.arange(1, nlead + 1))
    result = result.expand_dims(Y=[init_tag])
    result["time"] = xr.DataArray(valid_time[0][None, :], dims=("Y", "L"))
    return result.transpose("Y", "L", "M", ...)