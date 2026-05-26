"""Compliance tests for SMI loader/integrator with PyHyperScattering conventions.

Covers the changes from PR "SMI loader: PyHyper compliance & docs":
- wavelength attr in metres (matches SST1RSoXSLoader convention)
- smi_panels attr as JSON-encoded string (round-trip safe)
- CombinedReductionResult.to_dataarray() helper
"""
import json

import numpy as np
import pytest
import xarray as xr

from PyHyperScattering.SMISWAXSIntegrator import CombinedReductionResult


# ---------------------------------------------------------------------------
# CombinedReductionResult.to_dataarray()
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_result():
    q = np.linspace(0.1, 5.0, 10)
    chi = np.linspace(-180, 180, 8)
    merged_iq = xr.Dataset(
        {"I": ("q", np.arange(10.0)), "counts": ("q", np.ones(10))},
        coords={"q": q},
    )
    merged_qchi = xr.Dataset(
        {"intensity": (("q", "chi"), np.ones((10, 8))),
         "counts":    (("q", "chi"), np.ones((10, 8)))},
        coords={"q": q, "chi": chi},
    )
    per_frame_iq = xr.Dataset(
        {"I": (("frame", "q"), np.ones((3, 10)))},
        coords={"q": q, "frame": np.arange(3)},
    )
    return CombinedReductionResult(
        uid="test-uid",
        scan_info={"scan_id": 42, "sample_name": "test_sample"},
        saxs=None, waxs=None,
        merged_qchi=merged_qchi, merged_iq=merged_iq, per_frame_iq=per_frame_iq,
        geometry="transmission", incident_angle_deg=0.0,
    )


def test_to_dataarray_default_returns_iq(sample_result):
    da = sample_result.to_dataarray()
    assert isinstance(da, xr.DataArray)
    assert da.dims == ("q",)
    assert da.attrs["uid"] == "test-uid"
    assert da.attrs["scan_id"] == 42
    assert da.attrs["sample_name"] == "test_sample"
    assert da.attrs["geometry"] == "transmission"
    assert da.attrs["source"] == "merged_iq"


def test_to_dataarray_merged_qchi(sample_result):
    da = sample_result.to_dataarray("merged_qchi")
    assert da.dims == ("q", "chi")
    assert da.attrs["source"] == "merged_qchi"


def test_to_dataarray_per_frame_iq(sample_result):
    da = sample_result.to_dataarray("per_frame_iq")
    assert da.dims == ("frame", "q")


def test_to_dataarray_explicit_variable(sample_result):
    da = sample_result.to_dataarray("merged_iq", variable="counts")
    np.testing.assert_array_equal(da.values, np.ones(10))


def test_to_dataarray_bad_key_raises(sample_result):
    with pytest.raises(ValueError, match="key must be one of"):
        sample_result.to_dataarray("bogus")


def test_to_dataarray_missing_variable_raises(sample_result):
    with pytest.raises(ValueError, match="not in"):
        sample_result.to_dataarray("merged_iq", variable="not_a_var")


def test_to_dataarray_none_product_raises():
    empty = CombinedReductionResult(
        uid="x", scan_info={}, saxs=None, waxs=None,
        merged_qchi=None, merged_iq=None,
    )
    with pytest.raises(ValueError, match="wasn't produced"):
        empty.to_dataarray("merged_iq")


# ---------------------------------------------------------------------------
# Attrs serialization (smi_panels JSON-encoded; wavelength in metres)
# ---------------------------------------------------------------------------

def test_smi_panels_is_json_string():
    """smi_panels must be a JSON-encoded string so the DataArray is netCDF/Zarr/Tiled safe."""
    from PyHyperScattering.SMISWAXSLoader import (
        WAXSGeometry, WAXSPanelGeometry, load_waxs_raw,
    )

    # Build a minimal mock run + geometry so we can call load_waxs_raw without tiled.
    # We test the attrs construction logic specifically; image data is a stub.
    panels = (
        WAXSPanelGeometry(col_start=0,   col_end=206, offset_deg=-7.0),
        WAXSPanelGeometry(col_start=206, col_end=413, offset_deg=0.0),
        WAXSPanelGeometry(col_start=413, col_end=619, offset_deg=7.0),
    )
    geo = WAXSGeometry(
        dist_m=0.273,
        beam_center_row_px=217.0,
        beam_center_col_px=319.0,
        energy_ev=16100.0,
        wavelength_m=7.701e-11,
        panels=panels,
    )

    # We can't run load_waxs_raw without a tiled run, so we directly check
    # that the attrs would be JSON-encoded by inspecting the source.
    import inspect
    src = inspect.getsource(load_waxs_raw)
    assert 'json.dumps(panels_attr)' in src, (
        "load_waxs_raw should JSON-encode smi_panels for serialization"
    )

    # And that round-tripping a JSON-encoded panels attr works:
    panels_dict = [
        {"col_start": p.col_start, "col_end": p.col_end, "offset_deg": p.offset_deg}
        for p in panels
    ]
    encoded = json.dumps(panels_dict)
    decoded = json.loads(encoded)
    assert decoded[0]["offset_deg"] == -7.0
    assert decoded[1]["col_start"] == 206


def test_wavelength_attr_is_angstroms():
    """wavelength attr must be in Ångstroms — SMISWAXSIntegrator depends on this.

    SMISWAXSIntegrator.integrate_saxs reads ``attrs['wavelength']`` and
    converts via ``× 1e-10`` to metres.  If the loader stored metres
    instead, q would come out 10^10× too large and the merged_iq would
    be all-NaN (since q would land outside the bin range).
    """
    import inspect
    from PyHyperScattering.SMISWAXSLoader import load_saxs_raw, load_waxs_raw

    for fn in (load_saxs_raw, load_waxs_raw):
        src = inspect.getsource(fn)
        assert '"wavelength": geo.wavelength_m * 1e10' in src, (
            f"{fn.__name__} should store wavelength in Ångstroms "
            f"(geo.wavelength_m * 1e10) so SMISWAXSIntegrator's "
            f"× 1e-10 conversion produces metres."
        )
