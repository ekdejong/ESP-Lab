"""Access helpers for monthly SamudrACE hindcast predictions."""

from datetime import datetime
from functools import partial
from pathlib import Path

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
    counts_values = selected.counts.values
    if not np.all(np.isfinite(counts_values) & (counts_values > 0)):
        raise ValueError("Monthly sample counts must be finite and positive")
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


def get_monthly_data(
    data_dir: str | Path,
    case: str,
    init_tags,
    field: str,
    nlead: int = 24,
    realm: str = "atm",
    chunks: dict | None = None,
    engine: str = "netcdf4",
) -> xr.Dataset:
    """Open monthly predictions for the requested initializations lazily.

    ``data_dir`` contains the experiment directory ``case``, which contains
    run directories named ``<case>-<year>-<month>``. ``init_tags`` use the
    E3SM convention ``YYYYMM0100``; actual initialization dates remain in
    ``init_time``. All requested runs must exist and agree on their grid and
    member count. Close the returned dataset when it is no longer needed.

    ``chunks`` applies to the returned dimensions, such as ``Y``, ``L``,
    ``M``, ``lat`` and ``lon``. The first 24 lead months include a potentially
    partial initialization month; sample counts are returned for auditing.
    Process each initialization month separately when using seasonal helpers.
    """
    realms = {"atm": "atmosphere", "ocn": "ocean", "atmosphere": "atmosphere", "ocean": "ocean"}
    if realm not in realms:
        raise ValueError(f"Unsupported SamudrACE realm: {realm!r}")
    if not isinstance(case, str) or not case or Path(case).name != case or case in {".", ".."}:
        raise ValueError("case must be a single experiment directory name")
    if isinstance(init_tags, (str, bytes)):
        raise ValueError("init_tags must be a sequence of initialization tags")
    tags = list(init_tags)
    if not tags or len(set(tags)) != len(tags):
        raise ValueError("init_tags must be nonempty and contain no duplicates")
    if not isinstance(nlead, int) or isinstance(nlead, bool) or not 1 <= nlead <= 24:
        raise ValueError("nlead must be between 1 and 24")

    month_names = ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec")
    paths = []
    for tag in tags:
        if not isinstance(tag, str) or len(tag) != 10 or not tag.isascii() or not tag.isdigit() or tag[6:] != "0100":
            raise ValueError(f"Invalid initialization tag: {tag!r}")
        try:
            start = datetime.strptime(tag, "%Y%m%d%H")
        except ValueError as exc:
            raise ValueError(f"Invalid initialization tag: {tag!r}") from exc
        run = f"{case}-{start.year:04d}-{month_names[start.month - 1]}"
        path = Path(data_dir) / case / run / realms[realm] / "monthly_mean_predictions.nc"
        if not path.is_file():
            raise FileNotFoundError(f"Missing SamudrACE monthly predictions for {tag}: {path}")
        paths.append(path)

    opened = xr.open_mfdataset(
        paths,
        combine="nested",
        concat_dim="Y",
        preprocess=partial(preprocess_monthly, field=field, nlead=nlead),
        data_vars="all",
        coords="minimal",
        compat="equals",
        join="exact",
        chunks={},
        engine=engine,
        parallel=False,
    )
    try:
        if opened.Y.values.tolist() != tags:
            raise ValueError("SamudrACE initialization times do not match requested tags")
        result = opened.transpose("Y", "L", "M", ...)
        if chunks:
            result = result.unify_chunks()
            existing_chunks = result.chunksizes
            requested_chunks = {
                dim: chunks.get(dim, existing_chunks.get(dim, result.sizes[dim]))
                for dim in result.dims
            }
            unknown = set(chunks) - set(result.dims)
            if unknown:
                raise ValueError(f"Unknown chunk dimensions: {sorted(unknown)}")
            result = result.chunk(requested_chunks)
        result.set_close(opened.close)
        return result
    except Exception:
        opened.close()
        raise