"""Prepare SamudrACE inputs for lead-time skill notebooks."""

from pathlib import Path

from esp_lab import data_access_e3sm, data_access_samudrace, leadtime_realm_skill
from esp_lab.leadtime_prepared_cache import (
    expected_prepared_skill_attrs,
    prepared_skill_cache_status,
    prepared_skill_path,
)
from esp_lab.leadtime_skill_cache import source_fingerprint
from esp_lab.leadtime_workflow import resolve_source_revision
from esp_lab.utils import calendar_utils, regrid_utils
from esp_lab.utils.unit_conversion import (
    convert_kelvin_to_celsius,
    convert_precip_mmps_to_mmday,
)


ATMOSPHERIC_FIELDS = {
    "TREFHT": ("Tat2m", convert_kelvin_to_celsius),
    "TS": ("TS", convert_kelvin_to_celsius),
    "PRECT": ("surface_precipitation_rate", convert_precip_mmps_to_mmday),
}
OCEAN_FIELDS = {"SST": ("sst", convert_kelvin_to_celsius)}
FIELD_CONFIG = {"atm": ATMOSPHERIC_FIELDS, "ocn": OCEAN_FIELDS}


def prepare_seasonal_anomaly(
    *, data_dir, case, field, init_month, years, climatology_years,
    diagnostic_root, target_grid, target_grid_identity, regridding_method,
    periodic, chunks, source_identity_mode="inventory", prepared_mode="auto",
    force_recompute=False, ensemble_member_count=10, realm="atm",
):
    """Prepare one case/month's seven complete seasonal anomaly leads.

    Input files are inventoried unless snapshot or revision mode is selected.
    Source monthly values retain their own units until after regridding.
    Returns the opened prepared dataset and its source fingerprint; callers
    must close the dataset when finished.
    """
    if realm not in FIELD_CONFIG or field not in FIELD_CONFIG[realm]:
        raise ValueError(f"Unsupported SamudrACE {realm} field: {field!r}")
    if prepared_mode not in {"auto", "rebuild", "require"}:
        raise ValueError("prepared_mode must be auto, rebuild, or require")
    if source_identity_mode == "snapshot" and prepared_mode != "require":
        raise ValueError("Snapshot mode requires prepared_mode='require'")
    if prepared_mode == "require" and force_recompute:
        raise ValueError("Cannot force recomputation in require mode")

    years = [int(year) for year in years]
    if not years:
        raise ValueError("years must not be empty")
    native_field, convert = FIELD_CONFIG[realm][field]
    unit_conversion_version = f"samudrace_{realm}_units_v1"
    tags = data_access_e3sm.build_init_tags(years, int(init_month))
    identity = {
        "archive_root": str(data_dir), "case": case, "realm": realm,
        "native_field": native_field, "init_tags": tags, "monthly_leads": 24,
        "ensemble_member_count": int(ensemble_member_count),
        "preparation": "samudrace_seven_complete_seasons_v1",
        "unit_conversion_version": unit_conversion_version,
    }
    paths = []
    if source_identity_mode == "inventory":
        paths = data_access_samudrace.monthly_prediction_paths(data_dir, case, tags, realm)
    revision = resolve_source_revision(
        source_identity_mode, "samudrace_monthly_v1", identity=identity,
        snapshot_dir=Path(diagnostic_root) / "tmp" / "source_inventory_snapshots",
        paths=paths, root=data_dir,
    )
    source_identity = source_fingerprint(identity, source_revision=revision)
    source = f"SamudrACE-{case}"
    provenance = {
        "source_data_identity": source_identity, "case_prefix": case,
        "requested_years": years, "target_grid": target_grid_identity,
        "regridding_method": regridding_method,
        "ensemble_member_count": int(ensemble_member_count), "lead_count": 7,
        "unit_conversion_version": unit_conversion_version,
    }
    path = prepared_skill_path(
        source, realm, field, int(init_month), climatology_years,
        root=Path(diagnostic_root), **provenance,
    )
    expected = expected_prepared_skill_attrs(
        source=source, component=realm, variable=field, init_month=int(init_month),
        climatology_years=climatology_years, **provenance,
    )
    compatible, reason = prepared_skill_cache_status(path, expected_attrs=expected)
    if prepared_mode == "require" and not compatible:
        raise RuntimeError(f"Required SamudrACE prepared input unavailable: {path}: {reason}")

    monthly = None
    prepared_input = None

    def load_input():
        nonlocal monthly, prepared_input
        if prepared_input is None:
            monthly = data_access_samudrace.get_monthly_data(
                data_dir, case, tags, native_field, realm=realm, chunks=chunks,
            )
            if monthly.sizes["M"] != ensemble_member_count:
                raise ValueError(
                    f"Expected {ensemble_member_count} SamudrACE members, got {monthly.sizes['M']}"
                )
            seasonal = calendar_utils.mon_to_seas_dask(monthly[[native_field, "time"]])
            seasonal = seasonal.isel(L=slice(0, 7))
            mapper = regrid_utils.make_regridder(
                seasonal, target_grid, method=regridding_method, periodic=periodic,
            )
            values = convert(mapper(seasonal[native_field])).rename(field)
            prepared_input = (values, seasonal["time"].load())
        return prepared_input

    try:
        prepared = leadtime_realm_skill.prepare_drift_removed_anomaly(
            lambda: load_input()[0], lambda: load_input()[1],
            prepared_path=path, expected_attrs=expected, source=source,
            component=realm, field=field, init_month=int(init_month),
            climatology_years=climatology_years, requested_years=years,
            prepared_chunks=chunks, reuse=prepared_mode != "rebuild",
            force_recompute=force_recompute,
        )
    finally:
        if monthly is not None:
            monthly.close()
    return prepared, source_identity


def prepare_atmospheric_anomaly(**kwargs):
    """Prepare an atmospheric field using the shared seasonal workflow."""
    return prepare_seasonal_anomaly(realm="atm", **kwargs)